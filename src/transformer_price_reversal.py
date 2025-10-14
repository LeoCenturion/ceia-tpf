import pandas as pd
import numpy as np
import optuna
from functools import partial
from scipy.signal import find_peaks
from sklearn.metrics import classification_report
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

from backtest_utils import fetch_historical_data, sma
# AI write a forecast with chronos as such
# from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# df = TimeSeriesDataFrame("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv")

# predictor = TimeSeriesPredictor(prediction_length=48).fit(
#     df,
#     hyperparameters={
#         "Chronos": {"model_path": "amazon/chronos-bolt-base"},
#     },
# )

# predictions = predictor.predict(df)
# use data
# data = fetch_historical_data(
#         data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
#         start_date="2022-01-01T00:00:00Z"
#     )
# use 70/30 for training and validating AI!
