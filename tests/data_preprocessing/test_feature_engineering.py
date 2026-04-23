# tests/data_preprocessing/test_feature_engineering.py

import unittest

import pandas as pd

from src.app.data_preprocessing.feature_engineering import (
    add_lagged_features,
    add_technical_indicators,
    add_time_based_features,
)


class TestFeatureEngineering(unittest.TestCase):
    """
    Tests for the feature engineering functions.
    """

    def test_add_technical_indicators(self):
        """
        Test adding technical indicators to the data.
        """
        data = {
            "open": list(range(10, 300, 10)),
            "high": list(range(15, 305, 10)),
            "low": list(range(5, 295, 10)),
            "close": list(range(12, 302, 10)),
            "volume": list(range(100, 3000, 100)),
        }
        df = pd.DataFrame(data)

        df_with_indicators = add_technical_indicators(df)

        self.assertIn("momentum_rsi", df_with_indicators.columns)
        self.assertIn("trend_macd", df_with_indicators.columns)
        self.assertIn("volatility_bbh", df_with_indicators.columns)

    def test_add_time_based_features(self):
        """
        Test adding time-based features to the data.
        """
        data = {
            "timestamp": pd.to_datetime(["2023-01-01 12:00:00", "2023-01-02 13:00:00"])
        }
        df = pd.DataFrame(data)

        df_with_time_features = add_time_based_features(df)

        self.assertIn("hour", df_with_time_features.columns)
        self.assertIn("dayofweek", df_with_time_features.columns)

    def test_add_lagged_features(self):
        """
        Test adding lagged features to the data.
        """
        data = {"close": [10, 20, 30, 40, 50]}
        df = pd.DataFrame(data)

        df_with_lagged_features = add_lagged_features(df, "close", [1, 2])

        self.assertIn("close_lag_1", df_with_lagged_features.columns)
        self.assertIn("close_lag_2", df_with_lagged_features.columns)


if __name__ == "__main__":
    unittest.main()
