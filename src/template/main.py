import os
import ssl
import shutil

from datetime import datetime
from utils.api_cals import vectorization_request, decoding_request, fetch_data_from_db


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

df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

json_list_df = df.to_dict(orient='records')

df_vectorized, min_val, max_val = vectorization_request(
    col_time='datetime',
    col_target="load_consumption",
    json_list_df=json_list_df
)

print(df_vectorized)

json_list_df = df_vectorized.to_dict(orient='records')
df_decoding = decoding_request(
    col_time='datetime',
    col_target="load_consumption",
    json_list_norm_df=json_list_df,
    min_val=min_val,
    max_val=max_val
)
print(df_decoding)

destination_snapshot = os.path.join(BASE_PATH, 'snapshot_main.py')
shutil.copy(cur_running_path, destination_snapshot)