import os
import ssl
import torch
import shutil
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go

from tqdm import tqdm
from config import logger
from datetime import datetime
from plotly.subplots import make_subplots
from torch.utils.data import TensorDataset, DataLoader, Dataset
from utils.api_cals import vectorization_request, decoding_request, fetch_data_from_db


ssl._create_default_https_context = ssl._create_stdlib_context

home_path = os.getcwd()
home_path = f"{home_path}/src/transformer_model"
experiments_path = f"{home_path}/experiments"
dir_name = datetime.now().strftime("exp_%Y-%m-%d_%H-%M-%S")
BASE_PATH = f"{experiments_path}/{dir_name}"
os.makedirs(BASE_PATH, exist_ok=True)
params_file = f'{home_path}/params.yaml'
cur_running_path = f"{home_path}/main.py"

destination_snapshot = os.path.join(BASE_PATH, 'snapshot_main.py')
shutil.copy(cur_running_path, destination_snapshot)

home_path = os.getcwd()
path_to_save = BASE_PATH


LAG = 20
HORIZON = 288
BATCH_SIZE = 64
EPOCHS = 3

LR = 0.00001
D_MODEL = 8
NHEAD = 8
NUM_LAYERS = 8
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


def make_predictions(x_input, x_future, points_per_call, model, device="cpu"):
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


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x):
        attn_weights = torch.softmax(self.attn(x), dim=1)  # Вычисляем веса
        return (x * attn_weights).sum(dim=1)


class TimeSeriesTransformer(nn.Module):

    def __init__(self, input_dim, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT, output_seq_len=points_per_call):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.attn_pool = AttentionPooling(d_model)
        self.fc = nn.Linear(d_model, output_seq_len)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x_residual = x

        x = self.layer_norm(x)
        x = self.transformer_encoder(x)

        x = x + x_residual
        x = self.attn_pool(x)
        return self.fc(x)


SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)


df_init = fetch_data_from_db()

df_init['datetime'] = df_init['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

json_list_general_norm_df = df_init.to_dict(orient='records')
logger.info("Normalizing the data.")

message = 'Vectorizing the data'
cast_logger(message=message)

df_general_norm_df, min_val, max_val = vectorization_request(
    col_time='datetime',
    col_target=measurement,
    json_list_df=json_list_general_norm_df
)

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
logger.info("Normalizing the data.")

message = 'Vector decoding'
cast_logger(message=message)

df_predict = decoding_request(
    col_time='datetime',
    col_target=measurement,
    json_list_norm_df=json_list_df_forecast,
    min_val=min_val,
    max_val=max_val
)

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
