import os
import ssl
import math
import torch
import ephem
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go

from tqdm import tqdm
from datetime import datetime
from scipy.fftpack import fft
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader, Dataset


class Time2Vec:

    def __init__(self, col_time, col_target):
        self.min_year = 1900
        self.max_year = 2100
        self.min_month = 1
        self.max_month = 12
        self.min_day = 1
        self.max_day = 31
        self.min_week = 1
        self.max_week = 52
        self.min_day_of_week = 0
        self.max_day_of_week = 6
        self.min_minute = 0
        self.max_minute = 59
        self.min_second = 0
        self.max_second = 59
        self.min_hour = 0
        self.max_hour = 23
        self.scaler = MinMaxScaler()
        self.col_time = col_time
        self.col_target = col_target

    def get_part_of_day(self, hour):
        if 6 <= hour < 12:
            return 0
        elif 12 <= hour < 18:
            return 1
        elif 18 <= hour < 22:
            return 2
        else:
            return 3

    def check_different_years(self):
        return self.min_year != self.max_year

    def get_season(self, month):
        if month in [12, 1, 2]:
            return 0  # Зима
        elif month in [3, 4, 5]:
            return 1  # Весна
        elif month in [6, 7, 8]:
            return 2  # Лето
        else:
            return 3

    def meta_date(self, df):
        df_with_meta = df.copy()
        df_with_meta[self.col_time] = pd.to_datetime(df_with_meta[self.col_time])
        df_with_meta.set_index(self.col_time, inplace=True)
        df_with_meta[self.col_time] = df[self.col_time]
        df_with_meta['year'] = df_with_meta.index.year
        df_with_meta['month'] = df_with_meta.index.month
        df_with_meta['day'] = df_with_meta.index.day
        df_with_meta['week'] = df_with_meta.index.isocalendar().week
        df_with_meta['day_of_week'] = df_with_meta.index.dayofweek
        df_with_meta['hour'] = df_with_meta.index.hour
        df_with_meta['minute'] = df_with_meta.index.minute
        df_with_meta['second'] = df_with_meta.index.second
        df_with_meta['hour_sin'] = np.sin(2 * np.pi * df_with_meta['hour'] / 24)
        df_with_meta['hour_cos'] = np.cos(2 * np.pi * df_with_meta['hour'] / 24)
        df_with_meta['day_of_week_sin'] = np.sin(2 * np.pi * df_with_meta['day_of_week'] / 7)
        df_with_meta['day_of_week_cos'] = np.cos(2 * np.pi * df_with_meta['day_of_week'] / 7)
        df_with_meta['week_sin'] = np.sin(2 * np.pi * df_with_meta['week'] / 52)
        df_with_meta['week_cos'] = np.cos(2 * np.pi * df_with_meta['week'] / 52)
        df_with_meta['month_sin'] = np.sin(2 * np.pi * df_with_meta['month'] / 12)
        df_with_meta['month_cos'] = np.cos(2 * np.pi * df_with_meta['month'] / 12)
        df_with_meta['part_of_day'] = df_with_meta['hour'].apply(self.get_part_of_day)
        df_with_meta['is_night'] = df_with_meta['hour'].apply(lambda x: 1 if x >= 22 or x < 6 else 0)
        df_with_meta['is_weekend'] = df_with_meta['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df_with_meta['day_of_year'] = df_with_meta.index.dayofyear

        df_with_meta['is_working_hours'] = df_with_meta.apply(lambda row: 1 if 9 <= row['hour'] < 18 and row['is_weekend'] == 0 else 0, axis=1)
        df_with_meta['season'] = df_with_meta['month'].apply(self.get_season)
        df_with_meta['season_sin'] = np.sin(2 * np.pi * df_with_meta['season'] / 4)
        df_with_meta['season_cos'] = np.cos(2 * np.pi * df_with_meta['season'] / 4)
        df_with_meta['quarter'] = df_with_meta.index.quarter
        df_with_meta['quarter_sin'] = np.sin(2 * np.pi * df_with_meta['quarter'] / 4)
        df_with_meta['quarter_cos'] = np.cos(2 * np.pi * df_with_meta['quarter'] / 4)
        df_with_meta['moon_phase'] = df_with_meta.index.to_series().apply(lambda x: ephem.Moon(x).phase / 29.53)

        df_with_meta['time_trend'] = (df_with_meta.index - df_with_meta.index.min()).total_seconds()
        df_with_meta['fourier_time'] = np.abs(fft(df_with_meta['hour_sin'].astype(float).to_numpy()))

        return df_with_meta

    def normalize_column(self, column, min_val, max_val):
        return (column - min_val) / (max_val - min_val)

    def inverse_normalize_column(self, column, min_val, max_val):
        return column * (max_val - min_val) + min_val


    def vectorization(self, df):
        all_col = df.columns
        col_vec = [
            self.col_time,
            self.col_target,
            "year",
            "month",
            "day",
            "week",
            "day_of_week",
            "hour",
            "minute",
            "second",
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "week_sin",
            "week_cos",
            "month_sin",
            "month_cos",
            "part_of_day",
            "is_night",
            "is_weekend",
            "day_of_year",
            "is_working_hours",
            "season",
            "season_sin",
            "season_cos",
            "quarter",
            "quarter_sin",
            "quarter_cos",
            "moon_phase",
            "time_trend",
            "fourier_time"
        ]

        diff_cols = list(all_col.difference(col_vec))

        df[self.col_target] = df[self.col_target].astype(float)
        min_val = df[self.col_target].min() * 1.2
        max_val = df[self.col_target].max() * 1.2

        df_with_meta = self.meta_date(df)
        normalized_dates = []

        for index, date in df_with_meta.iterrows():
            time = index
            diff_col_values = date[diff_cols].values.tolist()

            year_norm = (date['year'] - self.min_year) / (self.max_year - self.min_year) if self.check_different_years() else 1
            month_norm = (date['month'] - self.min_month) / (self.max_month - self.min_month)
            day_norm = (date['day'] - self.min_day) / (self.max_day - self.min_day)
            week_norm = (date['week'] - self.min_week) / (self.max_week - self.min_week)
            day_of_week_norm = (date['day_of_week'] - self.min_day_of_week) / (self.max_day_of_week - self.min_day_of_week)
            hour_norm = (date['hour'] - self.min_hour) / (self.max_hour - self.min_hour)
            minute_norm = (date['minute'] - self.min_minute) / (self.max_minute - self.min_minute)
            second_norm = (date['second'] - self.min_second) / (self.max_second - self.min_second)
            part_of_day_norm = (date['part_of_day'] - 0) / 3
            is_night_norm = date['is_night']
            is_weekend_norm = date['is_weekend']
            day_of_year_norm = (date['day_of_year'] - 1) / 365

            is_working_hours = date["is_working_hours"]
            season = date["season"]
            season_sin = date["season_sin"]
            season_cos = date["season_cos"]
            quarter = date["quarter"]
            quarter_sin = date["quarter_sin"]
            quarter_cos = date["quarter_cos"]
            moon_phase = date["moon_phase"]
            time_trend = date["time_trend"]
            fourier_time = date["fourier_time"]

            normalized_date = [
                                  time, date[self.col_target], year_norm, month_norm, day_norm, week_norm, day_of_week_norm,
                                  hour_norm, minute_norm, second_norm,
                                  date['hour_sin'], date['hour_cos'], date['day_of_week_sin'], date['day_of_week_cos'],
                                  date['week_sin'], date['week_cos'], date['month_sin'], date['month_cos'],
                                  part_of_day_norm, is_night_norm, is_weekend_norm, day_of_year_norm, is_working_hours,
                                  season, season_sin, season_cos, quarter, quarter_sin, quarter_cos, moon_phase, time_trend, fourier_time
                              ] + diff_col_values
            normalized_dates.append(normalized_date)

        normalized_df = pd.DataFrame(normalized_dates, columns=col_vec + diff_cols)
        normalized_df[self.col_target] = self.normalize_column(normalized_df[self.col_target], min_val, max_val)
        normalized_df = normalized_df.fillna("None")

        return normalized_df, min_val, max_val

    def reverse_vectorization(self, df, min_val, max_val):
        df = df.sort_values(by=['year', 'month', 'day', 'hour', 'minute'], ascending=True)

        denormalized_dates = []
        for index, date in df.iterrows():
            year_denorm = date['year'] * (self.max_year - self.min_year) + self.min_year
            month_denorm = date['month'] * (self.max_month - self.min_month) + self.min_month
            day_denorm = date['day'] * (self.max_day - self.min_day) + self.min_day
            week_denorm = date['week'] * (self.max_week - self.min_week) + self.min_week
            day_of_week_denorm = date['day_of_week'] * (self.max_day_of_week - self.min_day_of_week) + self.min_day_of_week
            hour_denorm = date['hour'] * (self.max_hour - self.min_hour) + self.min_hour
            minute_denorm = date['minute'] * (self.max_minute - self.min_minute) + self.min_minute
            second_denorm = date['second'] * (self.max_second - self.min_second) + self.min_second

            denormalized_date = [
                date[self.col_target], year_denorm, month_denorm, day_denorm, week_denorm,
                day_of_week_denorm, hour_denorm, minute_denorm, second_denorm
            ]
            denormalized_dates.append(denormalized_date)

        for i in range(len(denormalized_dates)):
            denormalized_dates[i][0] = self.inverse_normalize_column(denormalized_dates[i][0], min_val, max_val)

        denormalized_df = pd.DataFrame(denormalized_dates, columns=[
            self.col_target, 'year', 'month', 'day', 'week', 'day_of_week',
            'hour', 'minute', 'second'
        ])

        denormalized_df['hour'] = denormalized_df['hour'].apply(lambda x: math.ceil(x))
        denormalized_df['minute'] = denormalized_df['minute'].apply(lambda x: math.ceil(x))
        denormalized_df['second'] = denormalized_df['second'].apply(lambda x: math.ceil(x))
        denormalized_df['month'] = denormalized_df['month'].apply(lambda x: math.ceil(x))
        denormalized_df['day'] = denormalized_df['day'].apply(lambda x: math.ceil(x))
        denormalized_df['year'] = denormalized_df['year'].apply(lambda x: math.ceil(x))

        denormalized_df[self.col_time] = pd.to_datetime({
            'year': denormalized_df['year'],
            'month': denormalized_df['month'],
            'day': denormalized_df['day'],
            'hour': denormalized_df['hour'],
            'minute': denormalized_df['minute'],
            'second': denormalized_df['second']
        })

        return denormalized_df



ssl._create_default_https_context = ssl._create_stdlib_context

home_path = os.getcwd()
home_path = f"{home_path}/src/transformer_model"
os.makedirs(home_path, exist_ok=True)
experiments_path = f"{home_path}/experiments"
dir_name = datetime.now().strftime("exp_%Y-%m-%d_%H-%M-%S")
BASE_PATH = f"{experiments_path}/{dir_name}"
os.makedirs(BASE_PATH, exist_ok=True)
params_file = f'{home_path}/params.yaml'
cur_running_path = f"{home_path}/main.py"

destination_snapshot = os.path.join(BASE_PATH, 'snapshot_main.py')

home_path = os.getcwd()
path_to_save = BASE_PATH

print('Start!!')

LAG = 5
HORIZON = 288
BATCH_SIZE = 1
EPOCHS = 1

LR = 0.1
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 4
DROPOUT = 0.2
points_per_call = LAG*4

measurement = 'load_consumption'

home_path = os.getcwd()

url_backend = os.getenv("BACKEND_URL", 'http://77.37.136.11:7070')

# col_for_train = [measurement, 'month', 'day', 'week', 'day_of_week',
#                  'hour', 'minute', 'hour_cos', 'day_of_week_cos', 'week_cos', 'month_cos',
#                  'part_of_day', 'is_night', 'is_weekend', 'day_of_year']

col_for_train = [measurement, "year", "month", "day", "week", "day_of_week", "hour", "minute", "hour_sin", "hour_cos",
                 "day_of_week_sin", "day_of_week_cos", "week_sin", "week_cos", "month_sin", "month_cos", "part_of_day",
                 "is_night", "is_weekend", "day_of_year"]

""" Possible columns for train

["year", "month", "day", "week", "day_of_week", "hour", "minute", "second", "hour_sin", "hour_cos",
 "day_of_week_sin", "day_of_week_cos", "week_sin", "week_cos", "month_sin", "month_cos", "part_of_day",
  "is_night", "is_weekend", "day_of_year"]
"""


def cast_logger(message):
    count = len(message) + 4
    if count > 150:
        count = 150
    print('='*count)
    print(f'>>> {message}')
    print('='*count)


def make_predictions(x_input, x_future, points_per_call, model, device="cuda"):
    model.eval()
    predict_values = []
    x_future_len = len(x_future)
    remaining_horizon = x_future_len

    while remaining_horizon > 0:
        current_points_to_predict = min(remaining_horizon, points_per_call)

        x_input_tensor = torch.tensor(x_input, dtype=torch.float32).to(device)
        x_input_tensor = x_input_tensor.unsqueeze(0)

        with torch.no_grad():
            y_predict = model(x_input_tensor)

        y_predict = y_predict.cpu().numpy().flatten()

        y_predict = y_predict[:current_points_to_predict]
        predict_values.extend(y_predict)

        for i in range(current_points_to_predict):
            cur_val = y_predict[i]
            x_input = np.delete(x_input, 0, axis=0)
            future_lag = x_future[0]
            x_future = np.delete(x_future, 0, axis=0)
            future_lag[0] = cur_val
            x_input = np.append(x_input, future_lag.reshape(1, -1), axis=0)

        remaining_horizon -= current_points_to_predict

    return predict_values


def create_x_input(df_train, n_steps):
    df_input = df_train.iloc[len(df_train) - n_steps:]
    x_input = df_input.values
    return x_input


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def split_sequence(sequence, n_steps, points_per_call):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        out_end_ix = end_ix + points_per_call
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix:out_end_ix, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


class TimeSeriesDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def generate_causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf')).to(device)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x, return_weights=False):
        attn_weights = torch.softmax(self.attn(x), dim=1)
        pooled = (x * attn_weights).sum(dim=1)
        return (pooled, attn_weights) if return_weights else pooled


class TimeSeriesTransformer(nn.Module):

    def __init__(self, input_dim, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT, output_seq_len=points_per_call):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_pool = AttentionPooling(d_model)
        self.fc = nn.Linear(d_model, output_seq_len)

    def generate_causal_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)  # Верхнетреугольная матрица с нулями на главной диагонали
        return mask.bool().to(next(self.parameters()).device)  # Перевод в bool и на тот же девайс, что и модель

    def forward(self, x, return_attn=False):
        assert x.shape[-1] == self.input_dim, f"Expected input dim {self.input_dim}, but got {x.shape[-1]}"

        x = self.embedding(x)
        x = self.positional_encoding(x)

        causal_mask = self.generate_causal_mask(x.shape[1])
        x = self.transformer_encoder(x, mask=causal_mask)

        if return_attn:
            x, attn_weights = self.attn_pool(x, return_weights=True)
            return self.fc(x), attn_weights

        return self.fc(x)


# df_init = fetch_data_from_db()

df_init = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQCkT-fxfA54I_bwXI7fJ1n2cphUQJXilT35uQj9hR64HTUlq3Oc2E5IAluygAFQCtAB4tsCYT6vh72/pub?gid=569315859&single=true&output=csv")


message = f'прочитали данные'
cast_logger(message=message)

print(df_init)

df_init = df_init.rename(columns={"Datetime": "datetime"})

df_init['datetime'] = pd.to_datetime(df_init['datetime'])  # Преобразование в datetime
df_init['datetime'] = df_init['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')  # Теперь можно форматировать

t2v = Time2Vec(col_time='datetime', col_target='load_consumption')


message = 'Vectorizing the data'
cast_logger(message=message)

df_general_norm_df, min_val, max_val = t2v.vectorization(df_init)

df_general_norm_df = df_general_norm_df.drop(columns=['datetime'])
all_col = df_general_norm_df.columns

# =============== Preparing data for training ================

df = df_general_norm_df

diff_cols = all_col.difference(col_for_train)

train_index = int(len(df) - HORIZON)
df_train_all_col = df.iloc[:train_index]
df_test_all_col = df.iloc[train_index:]

df_true_all_col = df_test_all_col.copy()
df = df_general_norm_df[col_for_train]
df_train = df.iloc[:train_index]
values = df_train[col_for_train].values

X, y = split_sequence(values, LAG, 1)

df_test = df.iloc[train_index:]

df_for_comparison = df_init.iloc[train_index:]

df_true = df_test.copy()
df_forecast = df_test.copy()
x_input = create_x_input(df_train, LAG)
df_test = df_test.copy()
x_future = df_test.values
n_features = values.shape[1]

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f">>> device = {device}")
model = TimeSeriesTransformer(input_dim=X.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

message = 'Started training'
cast_logger(message=message)

progress_bar_epochs = tqdm(range(EPOCHS), desc=f"Epoch")

for epoch in progress_bar_epochs:
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for X_batch, y_batch in progress_bar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        progress_bar.set_postfix(loss=train_loss / len(train_loader))

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss/len(train_loader):.4f}")


# model.eval()

save_path = f"{path_to_save}/model_weights.pth"
torch.save(model.state_dict(), save_path)
torch.save(model, f"{path_to_save}/model_full.pth")

future_predictions = make_predictions(x_input=x_input, x_future=x_future, points_per_call=points_per_call, model=model)

df_forecast[diff_cols] = df_true_all_col[diff_cols]

df_forecast[measurement] = future_predictions

json_list_df_forecast = df_forecast.to_dict(orient='records')
cast_logger("Normalizing the data.")

message = 'Vector decoding'
cast_logger(message=message)


df_predict = t2v.reverse_vectorization(df_forecast, min_val, max_val)


future_predictions = df_predict[measurement]
real_values = df_for_comparison[measurement]

mape_value = mean_absolute_percentage_error(real_values, future_predictions)

fig_consumption = make_subplots(rows=1, cols=1, subplot_titles=['consumption_real vs consumption_predict'])

fig_consumption.add_trace(
    go.Scatter(x=df_for_comparison['datetime'], y=df_for_comparison[measurement], mode='lines', name='consumption_real', line=dict(color='blue')), row=1,
    col=1)
fig_consumption.add_trace(go.Scatter(x=df_predict['datetime'], y=df_predict[measurement], mode='lines', name='consumption_predict',
                                     line=dict(color='orange')), row=1, col=1)

fig_consumption.add_trace(
    go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=True,
        name=f'📌 MAPE = {round(mape_value, 2)} %'
    )
)

fig_consumption.add_trace(
    go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=True,
        name=f'Transformer'
    )
)

template = "presentation"

fig_consumption.update_layout(template="presentation")

output_path = f"{path_to_save}/real_vs_predict.html"

fig_consumption.write_html(output_path)

fig_consumption.show()
