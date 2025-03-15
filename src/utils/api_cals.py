import os
import psycopg2
import requests
import pandas as pd

from config import logger


url_backend = os.getenv("BACKEND_URL", 'http://77.37.136.11:7070')


def vectorization_request(col_time, col_target, json_list_df):
    """
    Пример вызова:

    json_list_df = df_init.to_dict(orient='records')

    df_vectorized, min_val, max_val = vectorization_request(
        col_time='datetime',
        col_target="load_consumption",
        json_list_df=json_list_df
    )
    """
    url = f'{url_backend}/backend/v1/normalization'
    json = {"col_time": col_time, "col_target": col_target, "json_list_df": json_list_df}
    try:
        req = requests.post(url=url, json=json)
        if req.status_code == 200:
            response_json = req.json()
            norm_df = pd.DataFrame.from_dict(response_json['df_all_data_norm'])
            min_val = float(response_json['min_val'])
            max_val = float(response_json['max_val'])
            return norm_df, min_val, max_val
        else:
            logger.error(f'Status code backend server: {req.status_code}')
            return None, None, None
    except Exception as e:
        logger.error(e)
        return None, None, None


def decoding_request(col_time, col_target, json_list_norm_df, min_val, max_val):
    """
    Пример вызова:

    json_list_df = df.to_dict(orient='records')
    df_decoding = decoding_request(
        col_time='datetime',
        col_target='load_consumption',
        json_list_norm_df=json_list_df,
        min_val=min_val,
        max_val=max_val
    )
    """

    url = f'{url_backend}/backend/v1/reverse_normalization'
    json = {
        "col_time": col_time,
        "col_target": col_target,
        "min_val": min_val,
        "max_val": max_val,
        "json_list_norm_df": json_list_norm_df
    }
    try:
        req = requests.post(url=url, json=json)
        if req.status_code == 200:
            reverse_de_norm_data_json = req.json()
            reverse_norm_df = pd.DataFrame.from_dict(reverse_de_norm_data_json['df_all_data_reverse_norm'])
            return reverse_norm_df
        else:
            logger.error(f'Status code backend server: {req.status_code}')
            return None
    except Exception as e:
        logger.error(e)


def fetch_data_from_db():
    table_name = 'load_consumption'
    measurement = 'load_consumption'

    DB_PARAMS = {
        "dbname": "mydb",
        "user": "myuser",
        "password": "mypassword",
        "host": "77.37.136.11",
        "port": 8083
    }


    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    select_query = f"""
    SELECT * FROM {table_name} ORDER BY datetime;
    """

    cur.execute(select_query)
    rows = cur.fetchall()

    df_result = pd.DataFrame(rows, columns=["datetime", measurement])
    df_result["datetime"] = df_result["datetime"].dt.tz_localize(None)

    cur.close()
    conn.close()
    return df_result
