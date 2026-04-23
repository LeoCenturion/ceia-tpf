import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.app.chronos_strategy import ChronosPalazzoStrategy


class TestChronosPalazzoStrategy(unittest.TestCase):

    def setUp(self):
        self.config = {'volume_threshold': 50000, 'min_bars_to_fit': 30}

    def _make_strategy(self):
        with patch('src.app.chronos_strategy.PalazzoChronosBinaryClassificationPipeline'):
            return ChronosPalazzoStrategy(self.config)

    def _setup_pipeline_mocks(self, strategy, predicted_class=1):
        n = 35  # > min_bars_to_fit=30
        idx = range(n)
        features = pd.DataFrame({'f': [1.0] * n}, index=idx)
        y = pd.Series([1] * n, index=idx)
        mock_predictor = MagicMock()

        strategy.pipeline.step_2_feature_engineering = MagicMock(return_value=features)
        strategy.pipeline.step_3_labeling_and_weighting = MagicMock(return_value=(y,))
        strategy.pipeline.fit_predictor = MagicMock(return_value=(mock_predictor, ['f']))
        strategy.pipeline.predict_next = MagicMock(return_value=predicted_class)

    def _make_sample_data(self):
        return pd.DataFrame(
            {'close': [100], 'open': [90], 'high': [110], 'low': [80], 'volume': [1000]},
            index=pd.to_datetime(['2023-01-01']),
        )

    def test_initialization_creates_pipeline(self):
        with patch('src.app.chronos_strategy.PalazzoChronosBinaryClassificationPipeline') as mock_cls:
            strategy = ChronosPalazzoStrategy(self.config)
            mock_cls.assert_called_once_with(self.config)
            self.assertIs(strategy.pipeline, mock_cls.return_value)

    def test_get_signal_buy(self):
        strategy = self._make_strategy()
        self._setup_pipeline_mocks(strategy, predicted_class=1)

        with patch.object(strategy, '_process_new_1m_bars', return_value=True):
            signal = strategy.get_signal(self._make_sample_data())

        self.assertEqual(signal, 'BUY')

    def test_get_signal_sell(self):
        strategy = self._make_strategy()
        self._setup_pipeline_mocks(strategy, predicted_class=0)

        with patch.object(strategy, '_process_new_1m_bars', return_value=True):
            signal = strategy.get_signal(self._make_sample_data())

        self.assertEqual(signal, 'SELL')

    def test_get_signal_no_new_bars_returns_hold(self):
        strategy = self._make_strategy()

        with patch.object(strategy, '_process_new_1m_bars', return_value=False):
            signal = strategy.get_signal(self._make_sample_data())

        self.assertEqual(signal, 'HOLD')

    def test_get_signal_prediction_error_returns_hold(self):
        strategy = self._make_strategy()
        self._setup_pipeline_mocks(strategy)
        strategy.pipeline.step_2_feature_engineering.side_effect = Exception("Pipeline failed")

        with patch.object(strategy, '_process_new_1m_bars', return_value=True), \
             patch('logging.error') as mock_log_error:
            signal = strategy.get_signal(self._make_sample_data())

        self.assertEqual(signal, 'HOLD')
        mock_log_error.assert_called_once()


if __name__ == '__main__':
    unittest.main()
