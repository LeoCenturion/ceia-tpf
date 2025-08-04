from types import LambdaType
from typing import Callable, Optional, Sequence
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from collections import namedtuple
import os
import itertools as it

Variable = namedtuple("Variable", ["column", "label", "type"])

def prepare_df(df):
    df['open_date'] = pd.to_datetime(df['open_time'], unit='ms')
    df['diff'] =  df['close'] - df['open']
    df['dow'] = df['open_date'].dt.weekday

def read_data(data_path: str ) -> pd.DataFrame:
    # data_path = '../data/binance/python/data/spot/monthly/klines/BTCUSDT/1h/'

    if os.path.isdir(data_path):
        # data_dict = {}
        final_df = pd.DataFrame()
        for filename in os.listdir(data_path):
            if filename.endswith('.csv'):
                # parts = filename.split('-')
                # year = int(parts[2])
                # month = int(parts[3].split('.')[0])
                file_path = os.path.join(data_path, filename)
                final_df = pd.concat([final_df, pd.read_csv(file_path)])
                # df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time","quote_assets_volume", "number_of_trades", "taker_buy_base_asset_vol", "taker_buy_quote_asset_vol", "ignore"]
                # prepare_df(df)
                # if year not in data_dict:
                #     data_dict[year] = {}
                #     data_dict[year][month] = df
        return final_df
    else:
        return pd.read_csv(data_path)

def preprocess_data(
        data: pd.DataFrame,
        time_column: Optional[str] = None,
        inplace: bool=False
) -> pd.DataFrame:

    df = data if inplace else data.copy()
    if time_column != None:
        df[time_column] = pd.to_datetime(df[time_column])
        df.set_index(time_column, inplace=True)
    return df

def describe_instrument(
        df: pd.DataFrame,
        figsize=(11,5),
        ncols=2,
        time_variable = "date"
):
    def plot_variable_history(df: pd.DataFrame, ax: Axes, variable: str):
        sns.lineplot(df, x = time_variable, y = variable, ax=ax)
        ax.set_title(f"{variable} history")

    plot_price_history = lambda ax: plot_variable_history(df, ax, "close")
    plot_returns_history = lambda ax: plot_variable_history(df, ax, "returns")
    plot_volatility_history = lambda ax: plot_variable_history(df, ax, "volatility")
    plot_return_std_history = lambda ax: plot_variable_history(df, ax, "return_std")

    plots = [
        plot_price_history, plot_returns_history ,
        plot_volatility_history, plot_return_std_history
    ]
    plots = list(it.batched(plots, n=ncols))

    return plot_grid(
        plotters=plots,
        figsize=figsize,
    )


def plot_grid(
        plotters: Sequence[Sequence[Callable[[Axes], None]]],
        figsize=(11,5),
):
    ncols=len(plotters[0])
    nrows=len(plotters)
    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=figsize,
        layout="constrained"
    )

    for row in range(nrows):
        for col in range(ncols):
            ax = axs[row, col]
            plotters[row][col](ax)
    return fig, axs

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

