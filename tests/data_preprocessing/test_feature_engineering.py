# tests/data_preprocessing/test_feature_engineering.py

import unittest
import pandas as pd
from src.data_preprocessing.feature_engineering import add_technical_indicators, add_time_based_features, add_lagged_features

class TestFeatureEngineering(unittest.TestCase):
    """
    Tests for the feature engineering functions.
    """

    def test_add_technical_indicators(self):
        """
        Test adding technical indicators to the data.
        """
        data = {
            'open': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300],
            'high': [15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205, 215, 225, 235, 245, 255, 265, 275, 285, 295, 305],
            'low': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205, 215, 225, 235, 245, 255, 265, 275, 285, 295],
            'close': [12, 22, 32, 42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 232, 242, 252, 262, 272, 282, 292, 302],
            'volume': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]
        }
        df = pd.DataFrame(data)
        
        df_with_indicators = add_technical_indicators(df)
        
        self.assertIn('momentum_rsi', df_with_indicators.columns)
        self.assertIn('trend_macd', df_with_indicators.columns)
        self.assertIn('volatility_bbh', df_with_indicators.columns)

    def test_add_time_based_features(self):
        """
        Test adding time-based features to the data.
        """
        data = {'timestamp': pd.to_datetime(['2023-01-01 12:00:00', '2023-01-02 13:00:00'])}
        df = pd.DataFrame(data)
        
        df_with_time_features = add_time_based_features(df)
        
        self.assertIn('hour', df_with_time_features.columns)
        self.assertIn('dayofweek', df_with_time_features.columns)

    def test_add_lagged_features(self):
        """
        Test adding lagged features to the data.
        """
        data = {'close': [10, 20, 30, 40, 50]}
        df = pd.DataFrame(data)
        
        df_with_lagged_features = add_lagged_features(df, 'close', [1, 2])
        
        self.assertIn('close_lag_1', df_with_lagged_features.columns)
        self.assertIn('close_lag_2', df_with_lagged_features.columns)

if __name__ == '__main__':
    unittest.main()
