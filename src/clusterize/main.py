import os
import ssl
import shutil

from datetime import datetime
from utils.api_cals import vectorization_request, decoding_request, fetch_data_from_db
from tslearn.utils import to_time_series_dataset
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans


ssl._create_default_https_context = ssl._create_stdlib_context

home_path = os.getcwd()
home_path = f"{home_path}/src/template"
experiments_path = f"{home_path}/experiments"
dir_name = datetime.now().strftime("exp_%Y-%m-%d_%H-%M-%S")
BASE_PATH = f"{experiments_path}/{dir_name}"
os.makedirs(BASE_PATH, exist_ok=True)
params_file = f'{home_path}/params.yaml'
cur_running_path = f"{home_path}/main.py"


df = fetch_data_from_db()
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
pivot_df = df.pivot_table(index=['dayofweek', 'hour'], values='load_consumption', aggfunc='mean').reset_index()
time_series = to_time_series_dataset(pivot_df[['load_consumption']].values)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(pivot_df[['load_consumption']])
time_series_scaled = to_time_series_dataset(scaled_features)
km = TimeSeriesKMeans(n_clusters=6, metric="dtw", random_state=42)
pivot_df['cluster'] = km.fit_predict(time_series_scaled)
df = df.merge(pivot_df[['dayofweek', 'hour', 'cluster']], on=['dayofweek', 'hour'], how='left')


cluster_intervals = (
    pivot_df.groupby("cluster")["hour"]
    .agg(["min", "max"])
    .reset_index()
    .sort_values(by="min")
)

cluster_dict = {
    row["cluster"]: [f"{row['min']:02d}:00", f"{row['max'] + 1:02d}:00"]
    for _, row in cluster_intervals.iterrows()
}

print(cluster_dict)