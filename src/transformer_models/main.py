import os
import ssl
import shutil
import yaml
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.api_cals import vectorization_request, decoding_request, fetch_data_from_db
from datetime import datetime

# Установка SSL контекста
ssl._create_default_https_context = ssl._create_stdlib_context

# Настройка директорий
home_path = os.getcwd()
home_path = f"{home_path}/src/transformer_models"
experiments_path = f"{home_path}/experiments"
dir_name = datetime.now().strftime("exp_%Y-%m-%d_%H-%M-%S")
BASE_PATH = f"{experiments_path}/{dir_name}"
os.makedirs(BASE_PATH, exist_ok=True)
params_file = f'{home_path}/params.yaml'
cur_running_path = f"{home_path}/main.py"

# Загрузка параметров
params_path = os.path.join(home_path, params_file)
params = yaml.load(open(params_path, 'r'), Loader=yaml.SafeLoader)

lag = params['lag']
points_per_call = params['points_per_call']
epochs = params['epochs']
batch_size = params['batch_size']
activation = params['activation']

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)

# Разделение последовательности
def split_sequence(sequence, n_steps, horizon):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        out_end_ix = end_ix + horizon
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix:out_end_ix, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Позиционное кодирование
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000., (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        if isinstance(inputs, tf.SparseTensor):
            inputs = tf.sparse.to_dense(inputs)
        seq_len = tf.shape(inputs)[1]
        pos_encoding_slice = self.pos_encoding[:, :seq_len, :]
        return inputs + tf.cast(pos_encoding_slice, dtype=inputs.dtype)

# Слой энкодера
def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = Input(shape=(None, d_model), name="inputs")
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="attention")(
        query=inputs, value=inputs, key=inputs
    )
    attention = Dropout(dropout)(attention)
    attention = LayerNormalization(epsilon=1e-6)(inputs + attention)
    outputs = Dense(units=d_model, activation='relu')(attention)
    outputs = Dropout(dropout)(outputs)
    outputs = LayerNormalization(epsilon=1e-6)(attention + outputs)
    return Model(inputs=inputs, outputs=outputs, name=name)

# Слой декодера
def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = Input(shape=(None, d_model), name="inputs")
    enc_outputs = Input(shape=(None, d_model), name="encoder_outputs")
    attention1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="attention_1")(
        query=inputs, value=inputs, key=inputs
    )
    attention1 = Dropout(dropout)(attention1)
    attention1 = LayerNormalization(epsilon=1e-6)(inputs + attention1)
    attention2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="attention_2")(
        query=attention1, value=enc_outputs, key=enc_outputs
    )
    attention2 = Dropout(dropout)(attention2)
    attention2 = LayerNormalization(epsilon=1e-6)(attention1 + attention2)
    outputs = Dense(units=d_model, activation='relu')(attention2)
    outputs = Dropout(dropout)(outputs)
    outputs = LayerNormalization(epsilon=1e-6)(attention2 + outputs)
    return Model(inputs=[inputs, enc_outputs], outputs=outputs, name=name)

# Модель Transformer
def transformer(input_dim, num_layers, units, d_model, num_heads, dropout, points_per_call, name="transformer"):
    inputs = Input(shape=(None, input_dim), name="inputs")
    dec_inputs = Input(shape=(None, 1), name="dec_inputs")
    
    embeddings = Dense(d_model)(inputs)
    embeddings = PositionalEncoding(1000, d_model)(embeddings)
    
    enc_outputs = embeddings
    for i in range(num_layers):
        enc_outputs = encoder_layer(units=units, d_model=d_model, num_heads=num_heads, dropout=dropout, name=f"encoder_layer_{i}")(enc_outputs)
    
    dec_embeddings = Dense(d_model)(dec_inputs)
    dec_embeddings = PositionalEncoding(1000, d_model)(dec_embeddings)
    
    outputs = dec_embeddings
    for i in range(num_layers):
        outputs = decoder_layer(units=units, d_model=d_model, num_heads=num_heads, dropout=dropout, name=f"decoder_layer_{i}")([outputs, enc_outputs])
    
    outputs = outputs[:, -1, :]
    outputs = Dense(points_per_call, name="outputs_dense")(outputs)
    return Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

# Цикл обучения
def train_transformer(df_train, df_test, lag, points_per_call, epochs, batch_size):
    # Преобразование даты в числовой формат
    for col in df_train.columns:
        if df_train[col].dtype == 'object' or str(df_train[col].dtype).startswith('datetime'):
            try:
                df_train[col] = pd.to_datetime(df_train[col]).astype(int) / 10**9
                df_test[col] = pd.to_datetime(df_test[col]).astype(int) / 10**9
            except Exception:
                df_train = df_train.drop(columns=[col])
                df_test = df_test.drop(columns=[col])
    
    # Обработка пропущенных значений
    df_train = df_train.ffill().bfill()
    df_test = df_test.ffill().bfill()
    
    # Нормализация данных
    scaler = MinMaxScaler()
    df_train = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns)
    df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)
    
    values = df_train.values
    X, y = split_sequence(values, lag, points_per_call)
    
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    dec_inputs = X[:, :-1, 0]
    dec_inputs = np.expand_dims(dec_inputs, axis=-1)
    
    input_dim = X.shape[2]
    num_layers = 4
    units = 512
    d_model = 128
    num_heads = 8
    dropout = 0.1
    
    model = transformer(input_dim, num_layers, units, d_model, num_heads, dropout, points_per_call)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    history = model.fit([X, dec_inputs], y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model, history

if __name__ == "__main__":
    df_all_data = fetch_data_from_db()
    df_all_data['datetime'] = pd.to_datetime(df_all_data['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
    json_list_df = df_all_data.to_dict(orient='records')
    df_vectorized, min_val, max_val = vectorization_request(
        col_time='datetime',
        col_target="load_consumption",
        json_list_df=json_list_df
    )
    
    train_index = int(len(df_vectorized) * 0.8)
    df_train = df_vectorized.iloc[:train_index]
    df_test = df_vectorized.iloc[train_index:]
    
    df_train = df_train.select_dtypes(include=[np.number])
    df_test = df_test.select_dtypes(include=[np.number])
    
    df_train = df_train.ffill().bfill().astype(float)
    df_test = df_test.ffill().bfill().astype(float)
    
    model, history = train_transformer(df_train, df_test, lag, points_per_call, epochs, batch_size)
    
    model_save_path = f"{BASE_PATH}/transformer_model.keras"
    model.save(model_save_path)

    # --- Подготовка входных данных для предсказания ---
    encoder_input = df_test.values[:lag].reshape((1, lag, df_test.shape[1]))
    decoder_input = np.zeros((1, lag, 1))
    predictions = []

    for _ in range(len(df_test) - lag):
        pred = model.predict([encoder_input, decoder_input], verbose=0)
        predictions.append(pred.flatten())
        
        next_encoder_input = df_test.values[lag + len(predictions) - 1].reshape((1, 1, df_test.shape[1]))
        encoder_input = np.concatenate([encoder_input[:, 1:, :], next_encoder_input], axis=1)
        
        next_decoder_input = pred[:, -1].reshape((1, 1, 1))
        decoder_input = np.concatenate([decoder_input[:, 1:, :], next_decoder_input], axis=1)

    predict_values = np.concatenate(predictions, axis=0)
    df_test_lagged = df_test.iloc[lag:].copy()
    df_test_lagged['load_consumption'] = predict_values
    
    df_comparative = decoding_request(
        col_time='datetime',
        col_target="load_consumption",
        json_list_norm_df=df_test_lagged.to_dict(orient='records'),
        min_val=min_val,
        max_val=max_val
    )
    
    y_true = decoding_request(
        col_time='datetime',
        col_target="load_consumption",
        json_list_norm_df=df_test.iloc[lag:].to_dict(orient='records'),
        min_val=min_val,
        max_val=max_val
    )['load_consumption']
    y_pred = df_comparative['load_consumption']

    # Вычисление метрик
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    
    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R-squared": r2,
        "MAPE": mape,
        "WMAPE": wmape
    }
    
    df_metrics = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    df_metrics.to_csv(f"{BASE_PATH}/metrics.csv", index=False)
    
    df_comparative.to_csv(f"{BASE_PATH}/predictions.csv", index=False)
    
    destination_snapshot = os.path.join(BASE_PATH, 'snapshot_main.py')
    shutil.copy(cur_running_path, destination_snapshot)
    print("Обучение завершено. Результаты сохранены.")