import io
import os
import ssl
import shap
import yaml
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go

from datetime import datetime
from plotly.subplots import make_subplots
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.api_cals import vectorization_request, decoding_request, fetch_data_from_db
from tensorflow.keras.layers import (LSTM, Dense, Bidirectional, Dropout, Input, MaxPooling1D, Conv1D, Embedding,
                                     BatchNormalization, Reshape, Embedding, TimeDistributed, Flatten, Conv2D, GlobalAveragePooling2D)
from tensorflow.keras import regularizers

# tf.keras.backend.clear_session()

home_path = os.getcwd()
home_path = f"{home_path}/src/gru_models"
experiments_path = f"{home_path}/experiments"
dir_name = datetime.now().strftime("exp_%Y-%m-%d_%H-%M-%S")
BASE_PATH = f"{experiments_path}/{dir_name}"
os.makedirs(BASE_PATH, exist_ok=True)
params_file = f"{home_path}/params.yaml"
cur_running_path = f"{home_path}/main.py"

ssl._create_default_https_context = ssl._create_stdlib_context


def replace_zeros_with_average(df, column):
    values = df[column].values
    for i in range(len(values)):
        if values[i] == 0:
            prev_value = values[i - 1] if i > 0 else None
            next_value = values[i + 1] if i < len(values) - 1 else None

            neighbors = [v for v in [prev_value, next_value] if v is not None]
            values[i] = np.mean(neighbors) if neighbors else 0

    df[column] = values
    return df


class SaveBestWeights(Callback):
    def __init__(self):
        super(SaveBestWeights, self).__init__()
        self.best_weights = None
        self.best_loss = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("loss")
        if current_loss is None:
            return
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_weights = self.model.get_weights()


def calculate_metrics(y_true, y_pred):
    y_true_mean = y_true.mean()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true_mean) ** 2)

    r2 = 1 - (ss_res / ss_tot)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

    return rmse, r2, mae, mape, wmape


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


def create_x_input(df_train, n_steps):
    df_input = df_train.iloc[len(df_train) - n_steps:]
    x_input = df_input.values
    return x_input


def make_predictions(x_input, x_future, points_per_call):
    predict_values = []
    x_future_len = len(x_future)
    remaining_horizon = x_future_len

    while remaining_horizon > 0:
        current_points_to_predict = min(remaining_horizon, points_per_call)
        x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, x_input.shape[1], x_input.shape[2])), dtype=tf.float32)
        y_predict = model.predict(x_input_tensor, verbose=0)

        if len(y_predict.shape) == 2 and y_predict.shape[0] == 1:
            y_predict = y_predict[0]

        y_predict = y_predict[:current_points_to_predict]
        predict_values.extend(y_predict)

        for i in range(current_points_to_predict):
            cur_val = y_predict[i]
            x_input = np.delete(x_input, (0), axis=1)
            future_lag = x_future[0]
            x_future = np.delete(x_future, 0, axis=0)
            future_lag[0] = cur_val
            x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)

        remaining_horizon -= current_points_to_predict

    return predict_values


def calc_lcr(previous_val, cur_val):
    if previous_val == 0:
        return cur_val

    percentage_change = ((cur_val - previous_val) / abs(previous_val))
    return percentage_change


def make_predictions_lcr(x_input, x_future, points_per_call):
    """
    Выполняет рекурсивный прогноз на основе x_input и x_future.

    :param x_input: Входные данные для модели (numpy array с размерностью (1, lag, n_features)).
    :param x_future: Дополнительные данные, используемые для обновления lag (numpy array с размерностью (x_future_len, n_features)).
    :param points_per_call: Количество точек, которое модель возвращает за один вызов.
    :return: Список прогнозов длиной x_future_len.
    """
    predict_values = []
    x_future_len = len(x_future)  # Общее количество точек для предсказания
    remaining_horizon = x_future_len

    while remaining_horizon > 0:
        current_points_to_predict = min(remaining_horizon, points_per_call)

        x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, x_input.shape[1], x_input.shape[2])), dtype=tf.float32)

        y_predict = model.predict(x_input_tensor, verbose=0)

        if len(y_predict.shape) == 2 and y_predict.shape[0] == 1:
            y_predict = y_predict[0]

        y_predict = y_predict[:current_points_to_predict]
        predict_values.extend(y_predict)


        for i in range(current_points_to_predict):
            privios_val = x_input[0, -1, 0]
            cur_val = y_predict[i]

            lcr = calc_lcr(previous_val=privios_val, cur_val=cur_val)

            x_input = np.delete(x_input, (0), axis=1)

            future_lag = x_future[0]
            x_future = np.delete(x_future, 0, axis=0)

            future_lag[0] = cur_val
            x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)

            x_input[0, -1, -1] = lcr

        remaining_horizon -= current_points_to_predict

    return predict_values


params_path = os.path.join(home_path, params_file)
params = yaml.load(open(params_path, "r"), Loader=yaml.SafeLoader)
lstm0_units = params["lstm0_units"]
lstm1_units = params["lstm1_units"]
lstm2_units = params["lstm2_units"]
regularizers_l2 = params["regularizers_l2"]
recurrent_dropout_rate = params["recurrent_dropout_rate"]
cnn0_units = params["cnn0_units"]
cnn1_units = params["cnn1_units"]

lag = params["lag"]
activation = params["activation"]
optimizer = params["optimizer"]
epochs = params["epochs"]
points_per_call = params["points_per_call"]
points_to_predict = params["points_to_predict"]
target_date_str = params["target_date_str"]

model_type_chitecture = ["LSTM",]

# model_type_chitecture = ["LSTM", "Bi-LSTM", "CNN-BI-LSTM", "CNN-LSTM"]
# model_type_chitecture = ["CNN-BI-LSTM", "CNN-LSTM"]

# model_type_chitecture = ["Bi-LSTM"]


# case_A = ["consumption","year", "month", "day","hour", "minute"]
# case_A_temperature = ["consumption","year", "month", "day","hour", "minute", "temperature"]
# case_A_lag_3h_temperature = ["consumption","year", "month", "day","hour", "minute", "temperature_lag_3h"]
# case_A_lag_6h_temperature = ["consumption","year", "month", "day","hour", "minute", "temperature_lag_6h"]
# case_A_lag_12h_temperature = ["consumption","year", "month", "day","hour", "minute", "temperature_lag_12h"]
# case_A_lag_15h_temperature = ["consumption","year", "month", "day","hour", "minute", "temperature_lag_15h"]

# Case_A = [
#     "consumption",
#     "hour",
#     "hour_cos",
#     "hour_sin",
#     "minute",
# ]

# case_A = ["consumption", "year", "month", "day","hour", "minute"]
#
# case_B_temperature = ["consumption","year", "month", "day","hour", "minute", "temperature"]
#
# case_C = ["consumption", "year", "month", "day", "week", "day_of_week", "hour", "minute", "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "week_sin", "week_cos", "month_sin", "month_cos"]
#
# case_D = ["consumption", "year", "month", "day", "week", "day_of_week", "hour", "minute", "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "week_sin", "week_cos", "month_sin", "month_cos", "temperature"]

case_E = ["consumption", "year", "month", "day", "week", "day_of_week", "hour", "minute", "part_of_day", "is_night", "is_weekend", "day_of_year"]

case_F = ["consumption", "year", "month", "day", "week", "day_of_week", "hour", "minute", "part_of_day", "is_night", "is_weekend", "day_of_year", "temperature"]

case_G = ["consumption", "year", "month", "day", "week", "day_of_week", "hour", "minute", "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "week_sin", "week_cos", "month_sin", "month_cos", "part_of_day", "is_night", "is_weekend", "day_of_year"]

case_H = ["consumption", "year", "month", "day", "week", "day_of_week", "hour", "minute", "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "week_sin", "week_cos", "month_sin", "month_cos", "part_of_day", "is_night", "is_weekend", "day_of_year","temperature"]

case_I = ["consumption", "year", "month", "day", "week", "day_of_week", "hour", "minute", "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "week_sin", "week_cos", "month_sin", "month_cos", "part_of_day", "is_night", "is_weekend", "day_of_year", "is_working_hours", "season", "season_sin", "season_cos", "quarter", "quarter_sin", "quarter_cos", "moon_phase"]

case_J = ["consumption", "year", "month", "day", "week", "day_of_week", "hour", "minute", "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "week_sin", "week_cos", "month_sin", "month_cos", "part_of_day", "is_night", "is_weekend", "day_of_year", "is_working_hours", "season", "season_sin", "season_cos", "quarter", "quarter_sin", "quarter_cos", "moon_phase","temperature"]

case_K = ["consumption", "day", "day_of_week", "day_of_year", "hour", "hour_cos", "hour_sin", "is_night", "is_working_hours", "minute", "month", "month_cos", "month_sin", "moon_phase", "season", "season_cos", "season_sin", "week", "week_cos", "year"]

case_L = ["consumption", "day", "day_of_week", "day_of_year", "hour", "hour_cos", "hour_sin", "is_night", "is_working_hours", "minute", "month", "month_cos", "month_sin", "moon_phase", "season", "season_cos", "season_sin", "week", "week_cos", "year", "temperature"]

"""
"year", "month", "day", "week",
       "day_of_week", "hour", "minute", "second", "hour_sin", "hour_cos",
       "day_of_week_sin", "day_of_week_cos", "week_sin", "week_cos",
       "month_sin", "month_cos", "part_of_day", "is_night", "is_weekend",
       "day_of_year", "is_working_hours", "season", "season_sin", "season_cos",
       "quarter", "quarter_sin", "quarter_cos", "moon_phase",
"""

Case_A_new = ["consumption", "year", "week", "day_of_week", "hour", "minute", "second"]

case_previous_better_res = ['consumption', 'week', 'day_of_week', 'hour', 'minute', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'week_sin', 'week_cos']
case_C_new = ["consumption", "year", "week", "day_of_week", "hour", "minute", "hour_cos", "week_cos"]
Case_G_new = ["consumption", "year", "week", "day_of_week", "hour", "minute", "hour_cos", "week_cos", "part_of_day", "is_night", "is_weekend", "day_of_year"]
Case_M = ["consumption", "year", "week", "day_of_week", "hour", "minute", "hour_cos", "week_cos", "is_working_hours"]
case_C_old = ['consumption',  'week', 'day_of_week', 'hour', 'minute', 'hour_cos', 'week_cos']

Case_0 = ["consumption"]
Case_A = ["consumption", "year", "month", "day","hour", "minute"]
Case_B = ["consumption", "year", "week", "day_of_week", "hour"]
Case_C = ["consumption", "year", "week", "day_of_week", "hour", "is_working_hours"]
Case_D = ["consumption", "year", "week", "day_of_week", "hour", "part_of_day", "is_night", "is_weekend", "day_of_year"]


# my_case = ["consumption", "year", "month", "day", "week", "day_of_week", "hour", "minute", "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "week_sin", "week_cos", "month_sin", "month_cos", "part_of_day", "is_night", "is_weekend", "day_of_year", "is_working_hours", "season"]
my_case = ["consumption", "year", "month", "day", "week", "day_of_week", "hour", "minute", "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "week_sin", "week_cos", "month_sin", "month_cos", "part_of_day", "is_night", "is_weekend", "day_of_year", "is_working_hours"]
# my_case = ["consumption", "hour", "hour_cos","hour_sin","day", "minute", "is_working_hours"]

train_col_dict = {
    "my_case": my_case,
    # "Case_A": Case_A,
    # "Case_B": Case_B,
    # "Case_C": Case_C,
    # "Case_D": Case_D,
    # "case_previous_better_res": case_previous_better_res,
    # "case_C_old": case_C_old,
    # "case_F": case_F,
    # "case_G": case_G,
    # "case_H": case_H,
    # "case_I": case_I,
    # "case_J": case_J,
    # "case_K": case_K,
    # "case_L": case_L,


}
#
# random_seed_dict = {
#     "1": 1,
#     "2": 2,
# }

# random_seed_dict = {str(i): i for i in range(61, 101)}


experements = {
    # "Australia Bundoora": "https://docs.google.com/spreadsheets/d/e/2PACX-1vTl3ZMKUEqYeXJe1b8A4IbfYIKjWlm0lR61glDoXOEfHxsmDUv1ZZ2IK2GpjkH2fZ6fvX3NaCOryqzW/pub?gid=751874949&single=true&output=csv",
    # "Australia Albury-Wodonga": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQmRJCXCBp-qsY4LQrf8x_zJax_5FAnZDl6-sv1zje9m0pCM7hore-cjS3zlzJezgHIm6h81KY1hsEz/pub?gid=1184660391&single=true&output=csv",
    # "Australia Bendigo": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQgXCJsm0V7ylsqvzRzK_LHZzky0lABeXvRuiqRWzDumN1Y8i8xul-Ih1ERIU1v-C46AKISnOOzBmtb/pub?gid=1902219272&single=true&output=csv",
    # "Morocco Zone 1": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSgwB47qVFZcr1Aq--UWxZ6fDi9CGLZm-1i8QoMgfdaHUbV8EqSli3ayPxYYxD8kqfYYHD41uuNxbjZ/pub?gid=1952392108&single=true&output=csv",
    # "Morocco Zone 2": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQT1DfqAB5Yec8MIQ_E5A8w-SXNcRmTwbXsv2W-ZT1ZcXN_G83BHlb6QBgnWkO-MpH3oVgfLoE0SnLx/pub?gid=1952392108&single=true&output=csv",
    # "Morocco Zone 3": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQSHw5k7n3_RM6ksGbvdQJsa1i9-zF-18CFLCFnXFkCxQwqLcQ4Wu2_8EF2H1lF02ih2NLL9BDecFzQ/pub?gid=1952392108&single=true&output=csv",
    # "load_consumption_temp": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRFF6SXvGbQgQG1bh0hwbXgVWpUU_UG8OQAVhHcNAcAFT5x-XoIYxMAeF-goym6_wNhJEwQd3iGHp9b/pub?gid=791211028&single=true&output=csv",
    "load_consumption_real": "load_consumption_real"
}

# df_all_data.to_csv('/Users/nikitasavvin/Desktop/PhD/experiments-on-models/src/arima_model/experiments/b.csv')



os.makedirs(BASE_PATH, exist_ok=True)

res_dict = {}

for model_type in model_type_chitecture:

    model_dir = os.path.join(BASE_PATH, model_type)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for dir_name, col_for_train in train_col_dict.items():

        dir = os.path.join(model_dir, dir_name)
        if not os.path.exists(dir):
            os.makedirs(dir)

        destination_params = os.path.join(dir, "params.yaml")
        shutil.copy(params_file, destination_params)

        destination_snapshot = os.path.join(dir, "snapshot_main.py")
        shutil.copy(cur_running_path, destination_snapshot)

        for experement_name, csv_train_data in experements.items():
            if experement_name == "load_consumption_real":
                df_all_data = fetch_data_from_db()
                df_all_data = df_all_data.rename(columns={"datetime": "Datetime"})
                df_all_data.to_csv('/Users/nikitasavvin/Desktop/PhD/experiments-on-models/src/arima_model/experiments/b.csv')
                last = 288*62
                df_all_data = df_all_data.iloc[-last:]
                print(df_all_data)
            else:
                df_all_data = pd.read_csv(csv_train_data)

                # df_all_data = csv_train_data

            if experement_name == "load_consumption_temp":
                df_all_data = df_all_data.rename(columns={"time": "Datetime"})
                # df_all_data = fetch_data_from_db()


                df_index = df_all_data.copy()
                target_date = pd.to_datetime("2023-12-18 08:20:00")
                df_index["Datetime"] = pd.to_datetime(df_index["Datetime"])

                df_index["time_diff"] = (df_index["Datetime"] - target_date).abs()

                nearest_index = df_index["time_diff"].idxmin() - 1

                date_start = pd.to_datetime("2024-10-01 00:00:00")

                df_index["time_diff"] = (df_index["Datetime"] - date_start).abs()

                start_index = df_index["time_diff"].idxmin()

                df_all_data = df_all_data.iloc[: nearest_index]


            # df_all_data = df_all_data.iloc[start_index: nearest_index]
            #
            # df_all_data = df_all_data[
            #     (df_all_data["datetime"].dt.time >= pd.to_datetime("17:30").time()) &
            #     (df_all_data["datetime"].dt.time <= pd.to_datetime("22:00").time())
            #     ]


            # df_index = df_all_data.copy()
            # df_index["time"] = pd.to_datetime(df_index["time"])
            # target_date = pd.to_datetime(target_date_str)
            # df_index["time_diff"] = (df_index["time"] - target_date).abs()
            # nearest_index = df_index["time_diff"].idxmin()
            # print(f">>> nearest_index = {nearest_index}")
            # df_all_data = df_all_data.iloc[:nearest_index+1]
            # print(df_all_data)


            df_all_data["Datetime"] = pd.to_datetime(df_all_data["Datetime"]).apply(lambda x: x.replace(second=0))

            df_all_data = df_all_data.sort_values(by="Datetime")

            df_all_data["Datetime"] = df_all_data["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
            df_all_data = df_all_data.rename(columns={"load_consumption": "consumption"})

            df_all_data = df_all_data[["Datetime", "consumption"]]

            json_list_df = df_all_data.to_dict(orient="records")

            df_all_data_norm, min_val, max_val = vectorization_request(
                col_time="Datetime",
                col_target="consumption",
                json_list_df=json_list_df
            )

            print(df_all_data_norm.head(3))
            print(f"cols = {df_all_data_norm.columns}")

            # TODO Дата с которой делаем прогноз на сутки вперед ------------------------------------------------------------------

            start_day_index = len(df_all_data) - points_to_predict
            df_all_data_norm = df_all_data_norm[:start_day_index + points_to_predict]

            all_col = df_all_data_norm.columns

            # TODO Расчет LCR ------------------------------------------------------------------------------------------------------

            # df_all_data_norm["lcr"] = (df_all_data_norm["consumption"].shift(1) - df_all_data_norm["consumption"]) / df_all_data_norm["consumption"]
            # df_all_data_norm["lcr"] = df_all_data_norm["lcr"].shift(7)
            # df_all_data_norm = df_all_data_norm[8:]
            # df_all_data_norm = df_all_data_norm.reset_index()
            #
            # df_all_data_norm["temperature_lag_3h"] = df_all_data_norm["temperature"].shift(-180)
            # df_all_data_norm["temperature_lag_6h"] = df_all_data_norm["temperature"].shift(-360)
            # df_all_data_norm["temperature_lag_12h"] = df_all_data_norm["temperature"].shift(-720)
            # df_all_data_norm["temperature_lag_15h"] = df_all_data_norm["temperature"].shift(-900)
            #
            # df_all_data_norm = df_all_data_norm.dropna()
            #
            # df_all_data_norm = df_all_data_norm[:-(288*21 + 130)]

            # tf.keras.utils.set_random_seed(random_seed)
            tf.keras.utils.set_random_seed(91)

            tf.config.experimental.enable_op_determinism()

            dir_name_experiment = f"{experement_name}"

            experiment_dir = os.path.join(dir, dir_name_experiment)
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)

            # tf.keras.backend.clear_session()

            col_for_train_dir = os.path.join(dir, "col_for_train.txt")

            with open(col_for_train_dir, "w") as file:
                file.write(str(col_for_train))

            flag = f">>> Current model - {model_type} Case: {dir_name} <<<"
            print("-"*len(flag))
            print(flag)
            print("-"*len(flag))

            # TODO Здесь создаем данные для обучение и прогноз ---------------------------------------------------------------------

            diff_cols = all_col.difference(col_for_train)
            # Данные На Тренировку
            df = df_all_data_norm
            train_index = len(df) - points_to_predict
            df_train_all_col = df.iloc[:train_index]
            df_test_all_col = df.iloc[train_index + 1:]
            df_true_all_col = df_test_all_col.copy()
            df = df_all_data_norm[col_for_train]
            df_train = df.iloc[:train_index]
            values = df_train[col_for_train].values

            X, y = split_sequence(values, lag, points_per_call)


            df_test = df.iloc[train_index + 1:]
            df_true = df_test.copy()
            df_test["consumption"] = None
            df_forecast = df_test.copy()
            x_input = create_x_input(df_train, lag)
            df_test_no_lcr = df_test.copy()
            if "lcr" in col_for_train:
                df_test_no_lcr["lcr"] = None
            x_future = df_test_no_lcr.values
            n_features = values.shape[1]

            save_best_weights_callback = SaveBestWeights()


            # TODO Здесь задаем конфигурацию модели слои и тд --------------------------------------------------------------------


            # TODO  ---------------------------------Построение разновидности моделей-----------------------------------------------


            # TODO ---------BI-LSTM model------------------------------------------------------------------------------------------


            if model_type == "Bi-LSTM":
                bi_lstm_model = Sequential()

                bi_lstm_model.add(Input(shape=(lag, n_features)))

                bi_lstm_model.add(Bidirectional(LSTM(lstm0_units, activation="softplus",recurrent_dropout=recurrent_dropout_rate, return_sequences=True)))
                bi_lstm_model.add(Bidirectional(LSTM(lstm1_units, activation=activation, recurrent_dropout=recurrent_dropout_rate, return_sequences=True)))
                bi_lstm_model.add(Bidirectional(LSTM(lstm2_units, activation=activation, recurrent_dropout=recurrent_dropout_rate)))
                bi_lstm_model.add(Dense(points_per_call, activation="linear", kernel_regularizer=regularizers.l2(regularizers_l2)))

                # learning_rate = 0.01
                #
                # optimizer = Adam(learning_rate=learning_rate)
                bi_lstm_model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
                model = bi_lstm_model

            # TODO ---------LSTM model----------------------------------------------------------------------------------------------

            if model_type == "LSTM":
                lstm_model = Sequential()

                lstm_model.add(LSTM(lstm0_units, activation="softplus", return_sequences=True, recurrent_dropout=recurrent_dropout_rate))
                lstm_model.add(LSTM(lstm1_units, activation=activation, return_sequences=True, recurrent_dropout=recurrent_dropout_rate))
                lstm_model.add(LSTM(lstm2_units, activation=activation, recurrent_dropout=recurrent_dropout_rate))

                lstm_model.add(Dense(points_per_call, activation="linear", kernel_regularizer=regularizers.l2(regularizers_l2)))

                lstm_model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])

                model = lstm_model


            # TODO ---------CNN-LSTM model------------------------------------------------------------------------------------------

            if model_type == "CNN-LSTM":

                cnn_lstm_model = Sequential()

                cnn_lstm_model.add(Conv1D(filters=cnn0_units, kernel_size=1, activation=activation, input_shape=(lag, n_features)))
                cnn_lstm_model.add(MaxPooling1D(pool_size=1))

                cnn_lstm_model.add(Conv1D(filters=cnn1_units, kernel_size=1,  activation=activation))
                cnn_lstm_model.add(MaxPooling1D(pool_size=1))

                cnn_lstm_model.add(LSTM(lstm0_units, activation="softplus", return_sequences=True, recurrent_dropout=recurrent_dropout_rate))
                cnn_lstm_model.add(LSTM(lstm1_units, activation=activation, return_sequences=True, recurrent_dropout=recurrent_dropout_rate))
                cnn_lstm_model.add(LSTM(lstm2_units, activation=activation, recurrent_dropout=recurrent_dropout_rate))

                cnn_lstm_model.add(Dense(points_per_call, activation="linear", kernel_regularizer=regularizers.l2(regularizers_l2)))

                cnn_lstm_model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
                model = cnn_lstm_model


            # # TODO ---------CNN-BI-LSTM model---------------------------------------------------------------------------------------

            if model_type == "CNN-BI-LSTM":
                cnn_bi_lstm_model = Sequential()
                cnn_lstm_model = Sequential()
                cnn_lstm_model.add(Conv1D(filters=cnn0_units, kernel_size=1, activation=activation, input_shape=(lag, n_features)))
                cnn_lstm_model.add(MaxPooling1D(pool_size=1))

                cnn_lstm_model.add(Conv1D(filters=cnn1_units, kernel_size=1,  activation=activation))
                cnn_lstm_model.add(MaxPooling1D(pool_size=1))

                cnn_bi_lstm_model.add(Bidirectional(LSTM(lstm0_units, activation="softplus",recurrent_dropout=recurrent_dropout_rate, return_sequences=True)))
                cnn_bi_lstm_model.add(Bidirectional(LSTM(lstm1_units, activation=activation, recurrent_dropout=recurrent_dropout_rate, return_sequences=True)))
                cnn_bi_lstm_model.add(Bidirectional(LSTM(lstm2_units, activation=activation, recurrent_dropout=recurrent_dropout_rate)))

                cnn_bi_lstm_model.add(Dense(points_per_call, activation="linear", kernel_regularizer=regularizers.l2(regularizers_l2)))

                cnn_bi_lstm_model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

                model = cnn_bi_lstm_model


            # TODO Обучение --------------------------------------------------------------------------------------------------------
            history = model.fit(X, y, epochs=epochs, verbose=1)
            model_name = "italy_case_model_2025_test.keras"
            model_save_path = f"{experiment_dir}/{model_name}"

            model.save(model_save_path)


            # lstm_layer = model.layers[1]  # Берем первый LSTM-слой (индекс может отличаться)
            # weights = lstm_layer.get_weights()
            # # weights, recurrent_weights, biases = lstm_layer.get_weights()
            # weights_fwd, recurrent_weights_fwd, biases_fwd, weights_bwd, recurrent_weights_bwd, biases_bwd = weights
            #
            #
            # print("="*100)
            #
            # print(f"weights_fwd = {weights_fwd}")
            # print(f"recurrent_weights_fwd = {recurrent_weights_fwd}")
            # print(f"biases_fwd = {biases_fwd}")
            # print(f"weights_bwd = {weights_bwd}")
            # print(f"recurrent_weights_bwd = {recurrent_weights_bwd}")
            # print(f"biases_bwd = {biases_bwd}")


            # print("="*100)

            # feature_idx = 3  # Задаем индекс признака, который хотим изменить (0, 1, ..., features-1)
            # new_value = 1  # Задаем новое значение веса
            #
            # weights[feature_idx, :] = new_value
            #
            # lstm_layer.set_weights([weights, recurrent_weights, biases])
            #
            # print(f"Веса для фичи {feature_idx} успешно изменены на {new_value}!")


            # TODO Прогноз ---------------------------------------------------------------------------------------------------------


            print(df_train.head())

            x_input = create_x_input(df_train, lag)
            x_input = x_input.reshape((1, lag, n_features))
            if "lcr" in col_for_train:
                predict_values = make_predictions_lcr(x_input, x_future, points_per_call)
            else:
                predict_values = make_predictions(x_input, x_future, points_per_call)

            predict_values = np.array(predict_values).flatten()

            df_forecast["consumption"] = predict_values
            df_forecast = replace_zeros_with_average(df_forecast, "consumption")

            if len(diff_cols) > 0:
                for col in diff_cols:
                    df_forecast[col] = df_true_all_col[col]

            df_forecast[col] = df_true_all_col[col]

            json_list_df = df_forecast.to_dict(orient="records")
            df_comparative = decoding_request(
                col_time="Datetime",
                col_target="consumption",
                json_list_norm_df=json_list_df,
                min_val=min_val,
                max_val=max_val
            )

            df_predict = df_comparative.copy()
            df_predict = df_predict[["consumption", "Datetime"]]
            path = f"{experiment_dir}/predict.xlsx"
            df_predict.to_excel(path, index=False)

            json_list_df = df_true_all_col.to_dict(orient="records")
            df_true = decoding_request(
                col_time="Datetime",
                col_target="consumption",
                json_list_norm_df=json_list_df,
                min_val=min_val,
                max_val=max_val
            )

            # df_comparative["consumption"] = df_comparative["consumption"] ** 1.2
            # df_comparative["consumption"] = np.sign(df_comparative["consumption"]) * (np.abs(df_comparative["consumption"]) ** 1.5)


            y_true = df_true["consumption"]
            y_pred = df_comparative["consumption"]

            # TODO Метрики ---------------------------------------------------------------------------------------------------------

            if y_pred.isna().any().any():
                rmse, r2, mae, mape, wmape = None, None, None, None, None
                metrix_dict = {
                    "RMSE": rmse,
                    "R-squared": r2,
                    "MAE": mae,
                    "MAPE": mape,
                    "WMAPE": wmape
                }
            else:
                rmse, r2, mae, mape, wmape = calculate_metrics(y_true=y_true, y_pred=y_pred)

                print(f"MAPE = {mape}")

                metrix_dict = {
                    "RMSE": rmse,
                    "R-squared": r2,
                    "MAE": mae,
                    "MAPE": mape,
                    "WMAPE": wmape
                }

            res_dict[experiment_dir] = mape

            df_metrics = pd.DataFrame(list(metrix_dict.items()), columns=["Metric", "Value"])

            output_path = f"{experiment_dir}/metrics.xlsx"
            df_metrics.to_excel(output_path, index=False)

            output_path = f"{experiment_dir}/model_summary.txt"

            with open(output_path, "w") as f:
                with io.StringIO() as buf:
                    model.summary(print_fn=lambda x: buf.write(x + "\n"))
                    f.write(buf.getvalue())


            path = f"{experiment_dir}/model_architecture.png"

            plot_model(model, show_shapes=True, to_file=path)

            # TODO Отрисовка -------------------------------------------------------------------------------------------------------

            title = f"Model - {model_type} | Case - {dir_name} | Dataset - {experement_name}"

            fig_consumption = make_subplots(rows=1, cols=1, subplot_titles=[title])

            fig_consumption.add_trace(
                go.Scatter(x=df_true["Datetime"], y=df_true["consumption"], mode="lines", name="consumption_real", line=dict(color="blue")), row=1,
                col=1)
            fig_consumption.add_trace(go.Scatter(x=df_comparative["Datetime"], y=df_comparative["consumption"], mode="lines", name="consumption_predict",
                                                 line=dict(color="orange")), row=1, col=1)

            fig_consumption.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=True,
                    name=f"MAPE = {round(mape, 3)} %"
                )
            )
            rmse, r2, mae, mape, wmape
            fig_consumption.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=True,
                    name=f" R = {round(r2, 3)} %"
                )
            )

            fig_consumption.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=True,
                    name=f" RMSE = {round(rmse, 3)} %"
                )
            )

            fig_consumption.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=True,
                    name=f" MAE = {round(mae, 3)} %"
                )
            )

            fig_consumption.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=True,
                    name=f" WMAPE = {round(wmape, 3)} %"
                )
            )

            fig_consumption.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=True,
                    name=f"-------------------"
                )
            )
            fig_consumption.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=True,
                    name=f">> Cols for train <<:"
                )
            )
            for col in col_for_train:
                fig_consumption.add_trace(
                    go.Scatter(
                        x=[None], y=[None],
                        mode="lines",
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=True,
                        name=f" - {col}"
                    )
                )


            template = "presentation"

            fig_consumption.update_layout(template="presentation")

            output_path = f"{experiment_dir}/real_vs_predict.html"

            fig_consumption.write_html(output_path)


df = pd.DataFrame(list(res_dict.items()), columns=["path", "mape"])

df_new = df.copy()
cases = []
datasets = []
paths_norm = []
R_squareds = []
RMSEs = []
MAEs = []
WMAPEs = []

for path in df_new["path"]:
    path_list = path.split('/')
    path_norm = path_list[9:]
    path_relative = path_list[10:]
    case = path_norm[2]
    cases.append(case)
    dataset = path_norm[3]
    datasets.append(dataset)
    path_new = "/".join(path_norm)
    path_relative = "/".join(path_relative)
    path_to_metrix = f"{BASE_PATH}/{path_relative}/metrics.xlsx"
    df_metrix = pd.read_excel(path_to_metrix)
    R_squared = df_metrix[df_metrix["Metric"] == "R-squared"]["Value"].values[0]
    RMSE = df_metrix[df_metrix["Metric"] == "RMSE"]["Value"].values[0]
    MAE = df_metrix[df_metrix["Metric"] == "MAE"]["Value"].values[0]
    WMAPE = df_metrix[df_metrix["Metric"] == "WMAPE"]["Value"].values[0]
    R_squareds.append(R_squared)
    RMSEs.append(RMSE)
    MAEs.append(MAE)
    WMAPEs.append(WMAPE)

    paths_norm.append(path_new)

df_new["path"] = paths_norm
df_new["case"] = cases
df_new["dataset"] = datasets
df_new["R-squared"] = R_squareds
df_new["RMSE"] = RMSEs
df_new["MAE"] = MAEs
df_new["WMAPE"] = WMAPEs


df_new.to_excel(f"{BASE_PATH}/results.xlsx", index=False)
df_new.to_csv(f"{BASE_PATH}/results.csv", index=False)

