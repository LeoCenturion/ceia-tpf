import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from collections import namedtuple
import os

Variable = namedtuple("Variable", ["column", "label", "type"])

def prepare_df(df):
    df['open_date'] = pd.to_datetime(df['open_time'], unit='ms')
    df['diff'] =  df['close'] - df['open']
    df['dow'] = df['open_date'].dt.weekday

def read_values() -> dict[int, dict[int, pd.DataFrame]]:
    data_path = '../data/binance/python/data/spot/monthly/klines/BTCUSDT/1h/'
    data_dict = {}

    for filename in os.listdir(data_path):
        if filename.endswith('.csv'):
            parts = filename.split('-')
            year = int(parts[2])
            month = int(parts[3].split('.')[0])

            file_path = os.path.join(data_path, filename)
            df = pd.read_csv(file_path)
            df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time","quote_assets_volume", "number_of_trades", "taker_buy_base_asset_vol", "taker_buy_quote_asset_vol", "ignore"]
            prepare_df(df)
            if year not in data_dict:
                data_dict[year] = {}
            data_dict[year][month] = df

    return data_dict

def compute_runs(df):
    start_prices = []
    end_prices = []
    run_vectors = []
    run_lengths = []
    start_times = []
    end_times = []

    start_price = None
    start_time = None
    current_run_length = 0
    increasing = None  # Track if the run is increasing or decreasing
    open_date = "open_date"
    # Iterate over the rows in the dataframe
    for i in range(1, len(df)):
        # Calculate daily difference in closing price
        prev_close = df.loc[i-1, 'close']
        current_close = df.loc[i, 'close']
        # Determine if the current row continues the run
        if increasing is None:
            # Set initial direction of the run
            start_price = prev_close
            start_time = df.loc[i-1, open_date]
            increasing = current_close > prev_close
            current_run_length = 1
        elif (increasing and current_close > prev_close) or (not increasing and current_close < prev_close):
            # Continue the run
            current_run_length += 1
        else:
            # End of the run, record data
            start_prices.append(start_price)
            end_prices.append(prev_close)
            run_vectors.append(current_run_length if increasing else -current_run_length)
            run_lengths.append(current_run_length)
            start_times.append(start_time)
            end_times.append(df.loc[i-1, open_date])

            # Reset run variables for a new run
            start_price = prev_close
            start_time = df.loc[i-1, open_date]
            increasing = current_close > prev_close
            current_run_length = 1

    # Capture the last run if it reaches the end of the dataset
    start_prices.append(start_price)
    end_prices.append(df.loc[len(df)-1, 'close'])
    run_vectors.append(current_run_length if increasing else -current_run_length)
    run_lengths.append(current_run_length)
    start_times.append(start_time)
    end_times.append(df.loc[len(df)-1, open_date])

    # Create the new DataFrame
    result = pd.DataFrame(
        {'start_price': start_prices,
         'end_price': end_prices,
         'run_vector': run_vectors,
         'run_len': run_lengths,
         'start_time': start_times,
         'end_time': end_times})
    result['run_diff'] = result['end_price'] - result['start_price']
    return result
