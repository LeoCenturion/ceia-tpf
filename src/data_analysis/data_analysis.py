"""Data fetching and processing utilities."""

import ccxt
import pandas as pd
import numpy as np
import time
from functools import wraps
from statsmodels.tsa.stattools import adfuller
from joblib import Parallel, delayed, cpu_count
from src.constants import (
    OPEN_COL,
    HIGH_COL,
    LOW_COL,
    CLOSE_COL,
    VOLUME_COL,
    TIMESTAMP_COL,
)


def timer(func):
    """Decorator that prints the execution time of the function it decorates."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"\n>>> Function '{func.__name__}' executed in {duration:.4f} seconds")
        return result

    return wrapper


def fetch_historical_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_date: str = None,
    end_date: str = None,
    data_path: str = None,
) -> pd.DataFrame:
    """
    Fetch historical price data for backtesting.

    The `backtesting` library requires column names: 'Open', 'High', 'Low', 'Close'.

    :param symbol: Trading pair
    :param timeframe: Candle timeframe
    :param start_date: Start date for data fetch
    :param end_date: End date for data fetch
    :param data_path: Path to local CSV file. If provided, data is loaded from here.
    :return: DataFrame with OHLCV data
    """
    if data_path:
        df = pd.read_csv(data_path)
        df.rename(columns={"date": "timestamp", "Volume BTC": "volume"}, inplace=True)
        # s_timestamp = str(int(df['unix'].iloc[0]))
        # unit = 'us' if len(s_timestamp) > 13 else 'ms'
        # df["timestamp"] = pd.to_datetime(df["unix"], unit=unit)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        if start_date:
            start_date = pd.to_datetime(start_date)
            if start_date.tzinfo is not None:
                start_date = start_date.tz_convert(None)
            df = df[df.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            if end_date.tzinfo is not None:
                end_date = end_date.tz_convert(None)
            df = df[df.index <= end_date]

    else:
        exchange = ccxt.binance(
            {"enableRateLimit": True, "options": {"defaultType": "future"}}
        )
        start_timestamp = None
        if start_date:
            start_timestamp = exchange.parse8601(start_date)
        else:
            start_timestamp = None
        end_timestamp = None
        if end_date:
            end_timestamp = exchange.parse8601(end_date)
        else:
            end_timestamp = None

        ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe,
            since=start_timestamp,
            limit=end_timestamp,
        )

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

    # The backtesting library requires uppercase column names
    df.rename(
        columns={
            "open": OPEN_COL,
            "high": HIGH_COL,
            "low": LOW_COL,
            "close": CLOSE_COL,
            "volume": VOLUME_COL,
        },
        inplace=True,
    )
    df.sort_index(inplace=True)
    return df


# Fractional Differentiation functions
def get_weights_ffd(d, thres):
    """
    Get weights for fractional differentiation.
    """
    w, k = [1.0], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def _frac_diff_ffd_batch(df_chunk, w, width):
    """
    Helper to apply fractional differentiation on a batch of columns.
    """
    df_batch = {}
    for name in df_chunk.columns:
        series_f = df_chunk[name].ffill().dropna()
        df_ = pd.Series(0.0, index=series_f.index, name=name)

        for i in range(width, series_f.shape[0]):
            loc0, loc1 = series_f.index[i - width], series_f.index[i]
            if not np.isfinite(df_chunk.loc[loc1, name]):
                continue
            df_[loc1] = np.dot(w.T, series_f.loc[loc0:loc1])[0]
        df_batch[name] = df_.copy(deep=True)

    if not df_batch:
        return pd.DataFrame()

    return pd.concat(df_batch, axis=1)


@timer
def frac_diff_ffd(series, d, thres=1e-5):
    """
    Fractional differentiation with fixed-width window.
    Parallelized with column batching.
    """
    w = get_weights_ffd(d, thres)
    width = len(w) - 1

    n_jobs = cpu_count()
    columns = series.columns
    n_batches = min(n_jobs, len(columns))

    if n_batches < 1:
        return pd.DataFrame(index=series.index)

    col_chunks = np.array_split(columns, n_batches)
    # Use backend='multiprocessing' to avoid ResourceTracker/loky cleanup errors
    batch_results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_frac_diff_ffd_batch)(series[chunk], w, width)
        for chunk in col_chunks
    )
    df = pd.concat(batch_results, axis=1)
    return df


def _check_stationarity_batch(df_chunk):
    """
    Helper to check stationarity for a batch of columns.
    Returns a list of booleans (True if stationary).
    """
    results = []
    for col in df_chunk.columns:
        series = df_chunk[col]
        try:
            if series.nunique() <= 1:
                results.append(False)
                continue
            p_val = adfuller(series, maxlag=1, regression="c", autolag=None)[1]
            results.append(p_val < 0.05)
        except Exception:
            results.append(False)
    return results


@timer
def find_minimum_d(series):
    """
    Find minimum d for stationarity using ADF test.
    Returns optimal d and the differentiated series.
    Uses batched parallel execution for efficiency.
    """
    n_jobs = cpu_count()

    for d in np.linspace(0, 1, 11):
        d_series_df = frac_diff_ffd(series, d, thres=1e-5).dropna()
        if d_series_df.empty:
            continue

        # Split columns into batches for efficient parallel processing
        columns = d_series_df.columns
        # Handle case where n_jobs > n_columns
        n_batches = min(n_jobs, len(columns))
        col_chunks = np.array_split(columns, n_batches)

        # Execute batches in parallel
        # We pass the full dataframe subset to avoid slicing/pickling overhead repeatedly
        # Use backend='multiprocessing' to avoid ResourceTracker/loky cleanup errors
        batch_results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            delayed(_check_stationarity_batch)(d_series_df[chunk])
            for chunk in col_chunks
        )
        # Flatten results
        is_stationary = [item for sublist in batch_results for item in sublist]

        if all(is_stationary):
            return d, d_series_df  # Found a d that makes all columns stationary

    # Fallback to d=1.0
    d_series_df = frac_diff_ffd(series, 1.0, thres=1e-5).dropna()
    return 1.0, d_series_df


# Indicator functions to be used with `self.I`
def pct_change(series):
    """Calculate the percentage change of a series."""
    return pd.Series(series).pct_change()


def sma(series, n):
    """Calculate the simple moving average of a series."""
    return pd.Series(series).rolling(n).mean()


def ewm(series, span):
    """Calculate the exponential moving average of a series."""
    return pd.Series(series).ewm(span=span, adjust=False).mean()


def std(series, n):
    """Calculate the rolling standard deviation of a series."""
    return pd.Series(series).rolling(n).std()


def adjust_data_to_ubtc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusts the price data from BTC to micro-BTC (uBTC).
    1 BTC = 1,000,000 uBTC.
    This function divides the OHLC prices by 1,000,000.
    """
    df_copy = df.copy()
    for col in [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col] / 1_000_000
    return df_copy

