import os
import ssl
import yaml
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go

from config import logger
from datetime import datetime
from xgboost import XGBRegressor
from plotly.subplots import make_subplots
from utils.api_cals import vectorization_request, decoding_request, fetch_data_from_db

ssl._create_default_https_context = ssl._create_stdlib_context



home_path = os.getcwd()
home_path = f"{home_path}/src/xgboost_model"
experiments_path = f"{home_path}/experiments"
dir_name = datetime.now().strftime("exp_%Y-%m-%d_%H-%M-%S")
BASE_PATH = f"{experiments_path}/{dir_name}"
res_dir = BASE_PATH
os.makedirs(BASE_PATH, exist_ok=True)
params_file = f'{home_path}/params.yaml'
cur_running_path = f"{home_path}/main.py"

destination_snapshot = os.path.join(BASE_PATH, 'snapshot_main.py')
shutil.copy(cur_running_path, destination_snapshot)
horizon = 288


def cast_logger(message):
    count = len(message) + 4
    if count > 150:
        count = 150
    print('='*count)
    print(f'>>> {message}')
    print('='*count)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def split_sequence(sequence, n_steps):
    """
    Split a univariate sequence into samples for supervised learning.

    Parameters:
        sequence (np.ndarray): Input sequence.
        n_steps (int): Number of steps to look back.

    Returns:
        tuple: Arrays of input samples (X) and targets (y).
    """
    X, y = [], []
    for i in range(len(sequence) - n_steps):
        seq_x, seq_y = sequence[i:i + n_steps, :], sequence[i + n_steps, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def create_x_input(df_train, n_steps):
    """
    Create the input array for predictions from the training DataFrame.

    Parameters:
        df_train (pd.DataFrame): Training data.
        n_steps (int): Number of steps to look back.

    Returns:
        np.ndarray: Input array for predictions.
    """
    return df_train.iloc[-n_steps:].values


def make_predictions(x_input, x_future, n_features, model, lag):
    """
    Generate predictions for a future horizon using an iterative approach.

    Parameters:
        x_input (np.ndarray): Initial input data.
        x_future (np.ndarray): Future data.
        n_features (int): Number of features in the data.
        model (tf.keras.Model): Trained prediction model.
        lag (int): Number of time steps used for predictions.

    Returns:
        list: Predicted values.
    """
    predict_values = []
    for _ in range(len(x_future)):
        x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, -1)), dtype=tf.float32)
        y_predict = model.predict(x_input_tensor)
        predict_values.append(y_predict)

        x_input = np.delete(x_input, 0, axis=1)
        future_lag = x_future[0]
        x_future = np.delete(x_future, 0, axis=0)
        future_lag[0] = y_predict
        x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)
        x_input = x_input.reshape((1, lag, n_features))

    return predict_values


def forecast_XGBoost(
        col_target, df_all_data_norm, evaluation_index, last_known_index, lag,
        model_architecture_params, norm_values, col_for_train
):

    """
    Perform forecasting using XGBoost for regression.

    Parameters:
        col_target (str): Target column name.
        df_all_data_norm (pd.DataFrame): Normalized data.
        evaluation_index (int): Start index for evaluation.
        last_known_index (int): Last known data index.
        lag (int): Number of time steps for prediction.
        model_architecture_params (dict): Parameters for XGBoost model.
        forecast_type (str): Type of forecast ('predictions' or other).
        norm_values (bool): Whether to normalize input values.

    Returns:
        dict: DataFrames containing evaluation, true values, and predictions.
    """
    if norm_values:

        possible_cols = [
            col_target, 'year', 'month', 'day', 'week', 'day_of_week',
            'hour', 'minute', 'second', 'hour_sin', 'hour_cos',
            'day_of_week_sin', 'day_of_week_cos', 'week_sin', 'week_cos',
            'month_sin', 'month_cos', 'part_of_day', 'is_night', 'is_weekend', 'day_of_year'
        ]

    model_architecture_params = model_architecture_params[0]

    df_all_data_norm = df_all_data_norm[possible_cols]

    df_true_all_col = df_all_data_norm.iloc[evaluation_index: last_known_index]
    df_true_all_col_skip = df_all_data_norm.iloc[last_known_index:]
    df_all_data_norm[col_target] = df_all_data_norm[col_target].replace('None', None)
    df_all_data_norm[col_target] = df_all_data_norm[col_target].astype(float)

    all_columns = df_all_data_norm.columns
    diff_cols = all_columns.difference(col_for_train)
    columns = col_for_train
    train_index = evaluation_index

    df_true_all_col = df_true_all_col.iloc[:last_known_index + 1]

    df = df_all_data_norm[col_for_train]
    df_test = df.iloc[train_index + 1: last_known_index + 1]
    df_test.loc[:, col_target] = np.nan

    xgb_model = XGBRegressor(**model_architecture_params)

    df_evaluetion = pd.DataFrame({
        'column1': [1, 2, 3],
        'column2': ['a', 'b', 'c']
    })
    df_evaluetion['minute'] = 0
    df_evaluetion['second'] = 0

    df_all_data_norm = df_all_data_norm[col_for_train]

    train_index = last_known_index

    df_train = df_all_data_norm[:last_known_index + 1]


    df_test = df_all_data_norm.iloc[train_index + 1:]
    df_test.loc[:, col_target] = np.nan
    df_real_predict = df_test.copy()
    values = df_train[columns].values
    x_input = create_x_input(df_train, lag)
    x_future = df_test.values
    X, y = split_sequence(values, lag)
    n_features = values.shape[1]

    X_reshaped = X.reshape(X.shape[0], -1)
    xgb_model.fit(X_reshaped, y)

    x_input = x_input.reshape((1, lag, n_features))

    predict_values = make_predictions(x_input, x_future, n_features, xgb_model, lag)
    predict_values = np.array(predict_values).flatten()

    df_real_predict[col_target] = predict_values
    if len(diff_cols) > 0:
        for col in diff_cols:
            df_real_predict[col] = df_true_all_col_skip[col]

    df_real_predict[col_target] = predict_values

    for df in [df_evaluetion, df_true_all_col, df_real_predict]:
        df['second'] = df['second'].fillna(0)
        df['minute'] = df['minute'].fillna(method='ffill')
        df['second'] = df['second'].fillna(method='ffill')

        for col in ['year', 'hour', 'hour_sin', 'hour_cos']:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')

    loss_list = [1]

    df_evaluetion.fillna(method='ffill', inplace=True)

    dataframes = {
        'df_evaluation': df_evaluetion,
        'df_true_all_col': df_true_all_col,
        'df_real_predict': df_real_predict
    }
    for name, df in dataframes.items():
        none_indices = df[df.isnull().any(axis=1)].index.tolist()
        if none_indices:
            logger.error(f"В DataFrame '{name}' есть None на строках: {none_indices}")

    response_code, response_message = 200, 'The training was successful'

    return df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_message

message = "Getting data from the database"
cast_logger(message=message)

df = fetch_data_from_db()
df_index = df.copy()

target_date = pd.to_datetime("2025-03-18 06:00:00")

df_index["time_diff"] = (df_index["datetime"] - target_date).abs()

nearest_index = df_index["time_diff"].idxmin()

# df = df.iloc[:nearest_index+1]
print(df)

df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

df_for_evaluation = df.iloc[-horizon:]


json_list_df = df.to_dict(orient='records')
message = "Vectorizing the data"
cast_logger(message=message)
df_all_data_norm, min_val, max_val = vectorization_request(
    col_time='datetime',
    col_target="load_consumption",
    json_list_df=json_list_df
)

df_train = df_all_data_norm.iloc[:-horizon]
df_test = df_all_data_norm.iloc[-horizon:]
df_test_none = df_test.copy()
df_test_none['load_consumption'] = None

df_all_data_norm = pd.concat([df_train, df_test_none], ignore_index=True)

last_known_index = len(df_all_data_norm) - horizon - 1


col_target="load_consumption"
evaluation_index = 1

lag = 5
# model_architecture_params = [{
#     "objective": "reg:squarederror",
#     "n_estimators": 500,
#     "learning_rate": 0.1,
#     "max_depth": 15,
#     "subsample": .9,
#     "colsample_bytree": .9,
#     "min_child_weight": 5,
#     "booster": "gbtree"
#
# }]

objectives = [
    "reg:squarederror",  # Среднеквадратичная ошибка (MSE)
    "reg:squaredlogerror",  # Среднеквадратичная логарифмическая ошибка (MSLE)
    "reg:pseudohubererror",  # Псевдо-Huber потеря (устойчива к выбросам)
    "reg:logistic",  # Логистическая регрессия для регрессии
    "binary:logistic",  # Логистическая регрессия (возвращает вероятность)
    "binary:logitraw",  # Логистическая регрессия (возвращает логиты)
    "count:poisson",  # Пуассоновская регрессия (для подсчета событий)
    "reg:gamma",  # Гамма-регрессия (для положительных данных)
    "reg:tweedie",  # Tweedie-регрессия (для моделирования редких событий)
]

model_architecture_params = [{
    "objective": "reg:squarederror",
    "n_estimators": 500,
    "learning_rate": 0.1,
    "max_depth": 15,
    "subsample": .9,
    "colsample_bytree": .9,
    "min_child_weight": 5,
    "booster": "gbtree"
}]
norm_values = True

"""
Возможные колонки ["year", "month", "day", "week", "day_of_week", "hour", "minute", "second",
 "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "week_sin", "week_cos", 
 "month_sin", "month_cos", "part_of_day", "is_night", "is_weekend", "day_of_year"]"
"""

#
# Case_A = ["month", "day", "week", "day_of_week", "hour", "minute",
#                  "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "week_sin", "week_cos",
#                  "month_sin", "month_cos", "part_of_day", "is_night", "is_weekend", "day_of_year"]
#
Case_B = [
    "year", "month", "day", "day_of_year", "week", "day_of_week", "hour", "hour_cos", "day_of_week_sin", "day_of_week_cos",  "minute", "part_of_day", "is_night",
]

Case_A = [
    "year", "month", "day", "day_of_year", "week", "day_of_week", "hour", "hour_sin", "hour_cos", "minute", "part_of_day", "is_night",
]

cases = {
    "Case_A": Case_A,
    "Case_B": Case_B,
    # "Case_C": Case_C,
    # "Case_D": Case_D,
    # "Case_E": Case_E,
}

res_dict = {}

for case_name, col_for_train in cases.items():

    BASE_PATH = f"{experiments_path}/{dir_name}/{case_name}"

    os.makedirs(BASE_PATH, exist_ok=True)

    col_for_train.insert(0, col_target)

    message = "Learn model and do predict"
    cast_logger(message=message)
    df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_message = (
        forecast_XGBoost(
            col_target=col_target,
            df_all_data_norm=df_all_data_norm,
            evaluation_index=evaluation_index,
            last_known_index=last_known_index,
            lag=lag,
            model_architecture_params=model_architecture_params,
            norm_values=norm_values,
            col_for_train=col_for_train
    ))



    json_list_df = df_real_predict.to_dict(orient='records')

    message = 'Decoding the data'
    cast_logger(message=message)

    df_decoding = decoding_request(
        col_time='datetime',
        col_target="load_consumption",
        json_list_norm_df=json_list_df,
        min_val=min_val,
        max_val=max_val
    )


    future_predictions = df_decoding[col_target]
    real_values = df_for_evaluation[col_target]

    mape_value = mean_absolute_percentage_error(real_values, future_predictions)

    fig_consumption = make_subplots(rows=1, cols=1, subplot_titles=['consumption_real vs consumption_predict'])

    fig_consumption.add_trace(
        go.Scatter(x=df_for_evaluation['datetime'], y=df_for_evaluation[col_target], mode='lines', name='consumption_real', line=dict(color='blue')), row=1,
        col=1)
    fig_consumption.add_trace(go.Scatter(x=df_decoding['datetime'], y=df_decoding[col_target], mode='lines', name='consumption_predict',
                                         line=dict(color='orange')), row=1, col=1)

    MAPE = f'📌 MAPE = {round(mape_value, 2)} %'

    res_dict[BASE_PATH] = MAPE

    fig_consumption.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=True,
            name=MAPE
        )
    )

    fig_consumption.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=True,
            name=f'XGBoost'
        )
    )

    template = "presentation"

    fig_consumption.update_layout(template="presentation")

    output_path = f"{BASE_PATH}/real_vs_predict.html"

    fig_consumption.write_html(output_path)

    cast_logger(message=MAPE)

    fig_consumption.show()

df = pd.DataFrame(list(res_dict.items()), columns=['path', 'mape'])

df.to_csv(f'{res_dir}/results.csv', index=False)
