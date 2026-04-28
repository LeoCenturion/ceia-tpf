# src/data_preprocessing/data_preprocessing.py

import logging

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {file_path}.")
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return pd.DataFrame()


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the data for cleaning.

    Args:
        df (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The prepared data.
    """
    logger = logging.getLogger(__name__)
    logger.info("Preparing data for cleaning.")
    # Add any data preparation steps here
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the data by handling missing values and outliers.

    Args:
        df (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The cleaned data.
    """
    logger = logging.getLogger(__name__)
    logger.info("Cleaning data.")

    # Handle missing values
    df = df.ffill()

    # Handle outliers
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df
