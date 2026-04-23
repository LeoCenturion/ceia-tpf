# src/data_preprocessing/feature_engineering.py

import pandas as pd
from ta import add_all_ta_features

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to the data.

    Args:
        df (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The data with technical indicators.
    """
    df = add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="volume"
    )
    return df

def add_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds time-based features to the data.

    Args:
        df (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The data with time-based features.
    """
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df

def add_lagged_features(df: pd.DataFrame, col: str, lags: list) -> pd.DataFrame:
    """
    Adds lagged features to the data.

    Args:
        df (pd.DataFrame): The input data.
        col (str): The column to lag.
        lags (list): A list of integers representing the lags.

    Returns:
        pd.DataFrame: The data with lagged features.
    """
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df
