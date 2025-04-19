import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from utils.api_cals import vectorization_request, decoding_request, fetch_data_from_db

# upload
df = fetch_data_from_db()
print(df)
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df.sort_values('datetime', inplace=True)
df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")


# vector
json_list_df = df.to_dict(orient='records')

df_vectorized, min_val, max_val = vectorization_request(
    col_time='datetime',
    col_target="load_consumption",
    json_list_df=json_list_df
)

#  train, evaluate
train_size = len(df_vectorized) - 288
df_train = df_vectorized.iloc[:train_size]
df_evaluate = df_vectorized.iloc[train_size:]

# ARIMA - train
external_features_train = df_train.drop(columns=['load_consumption', 'datetime'])
model = ARIMA(df_train['load_consumption'], exog=external_features_train, order=(5,1,0))
model_fit = model.fit()


external_features_eval = df_evaluate.drop(columns=['load_consumption', 'datetime'])
predictions = model_fit.forecast(steps=288, exog=external_features_eval)

# metrics
mape = mean_absolute_percentage_error(df_evaluate['load_consumption'], predictions)
r2 = r2_score(df_evaluate['load_consumption'], predictions)

# predict
fig_eval = go.Figure()
fig_eval.add_trace(go.Scatter(x=df_evaluate['datetime'], y=df_evaluate['load_consumption'], mode='lines', name='Actual'))
fig_eval.add_trace(go.Scatter(x=df_evaluate['datetime'], y=predictions, mode='lines', name='Forecast'))
fig_eval.update_layout(title='Evaluation Forecast', xaxis_title='Time', yaxis_title='Load Consumption')
fig_eval.show()

# real
external_features_real = df_vectorized.drop(columns=['load_consumption', 'datetime']).iloc[-288:]
predictions_real = model_fit.forecast(steps=288, exog=external_features_real)

# decode
json_list_pred = pd.DataFrame({
    'datetime': pd.date_range(start=df['datetime'].iloc[-1], periods=288, freq='H'),
    'load_consumption': predictions_real
}).to_dict(orient='records')
df_decoded = decoding_request(col_time='datetime', col_target="load_consumption", json_list_norm_df=json_list_pred, min_val=min_val, max_val=max_val)

# draw
fig_real = go.Figure()
fig_real.add_trace(go.Scatter(x=df_decoded['datetime'], y=df_decoded['load_consumption'], mode='lines', name='Forecast'))
fig_real.update_layout(title='Real Forecast', xaxis_title='Time', yaxis_title='Load Consumption')
fig_real.show()

print(f'MAPE: {mape:.4f}, R²: {r2:.4f}')
fig_eval.write_image("results/evaluation_forecast.png")
fig_real.write_image("results/real_forecast.png")