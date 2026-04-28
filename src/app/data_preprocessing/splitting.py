# src/data_preprocessing/splitting.py

import pandas as pd


def split_data(df: pd.DataFrame, train_size: float = 0.7, val_size: float = 0.15):
    """
    Splits the data into training, validation, and test sets.

    Args:
        df (pd.DataFrame): The input data.
        train_size (float): The proportion of the data to use for training.
        val_size (float): The proportion of the data to use for validation.

    Returns:
        tuple: A tuple containing the training, validation, and test sets.
    """
    train_end = int(len(df) * train_size)
    val_end = train_end + int(len(df) * val_size)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df
