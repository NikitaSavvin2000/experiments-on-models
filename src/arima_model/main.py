import os
import ssl
import shutil

from datetime import datetime
from utils.api_cals import vectorization_request, decoding_request, fetch_data_from_db
import plotly.graph_objects as go

import pandas as pd
import statsmodels.api as sm


def cast_logger(message):
    count = len(message) + 4
    if count > 150:
        count = 150
    print('='*count)
    print(f'>>> {message}')
    print('='*count)


ssl._create_default_https_context = ssl._create_stdlib_context

home_path = os.getcwd()
home_path = f"{home_path}/src/arima_model"
experiments_path = f"{home_path}/experiments"
dir_name = datetime.now().strftime("exp_%Y-%m-%d_%H-%M-%S")
BASE_PATH = f"{experiments_path}/{dir_name}"
os.makedirs(BASE_PATH, exist_ok=True)
params_file = f'{home_path}/params.yaml'
cur_running_path = f"{home_path}/main.py"


df = fetch_data_from_db()


df_to_eval = df.iloc[-288:]

df_to_exog = df_to_eval.copy()
df_to_exog['datetime'] = df_to_exog['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
json_list_df = df_to_exog.to_dict(orient='records')
message = "vectorization_request"
cast_logger(message=message)
df_vectorized_to_exog, min_val, max_val = vectorization_request(
    col_time='datetime',
    col_target="load_consumption",
    json_list_df=json_list_df
)

df_vectorized_to_exog['datetime'] = pd.to_datetime(df_vectorized_to_exog['datetime'])
df_vectorized_to_exog.set_index('datetime', inplace=True)



df = df.iloc[:-288]
df_features = df.copy()
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)
# df = df.asfreq('5T')


df_features['datetime'] = df_features['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')


json_list_df = df_features.to_dict(orient='records')

message = "vectorization_request"
cast_logger(message=message)

df_vectorized, min_val, max_val = vectorization_request(
    col_time='datetime',
    col_target="load_consumption",
    json_list_df=json_list_df
)

df_vectorized['datetime'] = pd.to_datetime(df_vectorized['datetime'])
df_vectorized.set_index('datetime', inplace=True)

step = 288
train = df['load_consumption']

print(df_vectorized.describe)

col_for_train = [
    "year", "month", "day", "day_of_year", "week", "day_of_week", "hour", "hour_cos", "day_of_week_sin", "day_of_week_cos",  "minute", "part_of_day", "is_night",
]
exog = df_vectorized[col_for_train]

from statsmodels.tsa.stattools import adfuller

def find_best_d(train, max_d=2):
    for d in range(max_d + 1):
        test_series = train.diff(d).dropna() if d > 0 else train
        p_value = adfuller(test_series)[1]
        if p_value < 0.05:
            print(f"Оптимальное d: {d}")
            return d  # Считаем, что ряд стационарный при p-value < 0.05
    return max_d  # Если не нашли, возвращаем максимум


d = find_best_d(train)

print(d)
model = sm.tsa.ARIMA(train, order=(d,5,4), exog=exog)

message = "Learn model and do predict"
cast_logger(message=message)
arima_fit = model.fit()

future_exog = df_vectorized_to_exog[col_for_train]

# forecast = arima_fit.forecast(steps=step)
forecast = arima_fit.forecast(steps=step, exog=future_exog)

future_dates = pd.date_range(df.index[-1], periods=step+1, freq='5T')[1:]

forecast_df = pd.DataFrame({'datetime': future_dates, 'forecast': forecast})
forecast_df.set_index('datetime', inplace=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_to_eval["datetime"], y=df_to_eval['load_consumption'], mode='lines', name="Исторические данные"))
fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['forecast'], mode='lines', name="Прогноз", line=dict(color='red')))

fig.update_layout(title="Прогноз потребления с помощью ARIMA", xaxis_title="Дата", yaxis_title="Потребление")
fig.show()



#
# json_list_df = df.to_dict(orient='records')
#
# df_vectorized, min_val, max_val = vectorization_request(
#     col_time='datetime',
#     col_target="load_consumption",
#     json_list_df=json_list_df
# )
#
# print(df_vectorized)
#
# json_list_df = df_vectorized.to_dict(orient='records')
# df_decoding = decoding_request(
#     col_time='datetime',
#     col_target="load_consumption",
#     json_list_norm_df=json_list_df,
#     min_val=min_val,
#     max_val=max_val
# )
# print(df_decoding)
#
# destination_snapshot = os.path.join(BASE_PATH, 'snapshot_main.py')
# shutil.copy(cur_running_path, destination_snapshot)