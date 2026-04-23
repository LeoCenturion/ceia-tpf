# src/data_preprocessing/scaling.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales the numerical data using StandardScaler.

    Args:
        df (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The scaled data.
    """
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=np.number).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df
