
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.app.chronos_strategy import ChronosPalazzoStrategy

class TestChronosPalazzoStrategy(unittest.TestCase):

    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_pipeline = MagicMock()
        self.config = {
            'model_path': '/fake/path'
        }

    @patch('autogluon.timeseries.TimeSeriesPredictor.load')
    def test_initialization_loads_model(self, mock_load):
        mock_load.return_value = self.mock_model
        strategy = ChronosPalazzoStrategy(self.config)
        mock_load.assert_called_with('/fake/path')
        self.assertEqual(strategy.model, self.mock_model)

    @patch('autogluon.timeseries.TimeSeriesPredictor.load')
    @patch('src.modeling.chronos_modeling.PalazzoChronosBinaryClassificationPipeline')
    def test_get_signal_buy(self, mock_pipeline_class, mock_load):
        self.mock_model.predict.return_value = pd.DataFrame({'mean': [1]})
        mock_load.return_value = self.mock_model
        mock_pipeline_instance = self.mock_pipeline
        mock_pipeline_class.return_value = mock_pipeline_instance
        
        strategy = ChronosPalazzoStrategy(self.config)
        
        # Create a sample dataframe
        data = {'timestamp': pd.to_datetime(['2023-01-01']), 'close': [100], 'open': [90], 'high': [110], 'low': [80], 'volume': [1000]}
        sample_data = pd.DataFrame(data).set_index('timestamp')

        signal = strategy.get_signal(sample_data)
        self.assertEqual(signal, 'BUY')

    @patch('autogluon.timeseries.TimeSeriesPredictor.load')
    @patch('src.modeling.chronos_modeling.PalazzoChronosBinaryClassificationPipeline')
    def test_get_signal_sell(self, mock_pipeline_class, mock_load):
        self.mock_model.predict.return_value = pd.DataFrame({'mean': [-1]})
        mock_load.return_value = self.mock_model
        mock_pipeline_instance = self.mock_pipeline
        mock_pipeline_class.return_value = mock_pipeline_instance

        strategy = ChronosPalazzoStrategy(self.config)
        
        data = {'timestamp': pd.to_datetime(['2023-01-01']), 'close': [100], 'open': [90], 'high': [110], 'low': [80], 'volume': [1000]}
        sample_data = pd.DataFrame(data).set_index('timestamp')

        signal = strategy.get_signal(sample_data)
        self.assertEqual(signal, 'SELL')
        
        self.mock_model.predict.return_value = pd.DataFrame({'mean': [0]})
        signal = strategy.get_signal(sample_data)
        self.assertEqual(signal, 'SELL')

    @patch('autogluon.timeseries.TimeSeriesPredictor.load')
    @patch('src.modeling.chronos_modeling.PalazzoChronosBinaryClassificationPipeline')
    @patch('logging.error')
    def test_get_signal_prediction_error(self, mock_log_error, mock_pipeline_class, mock_load):
        self.mock_model.predict.side_effect = Exception("Prediction failed")
        mock_load.return_value = self.mock_model
        mock_pipeline_instance = self.mock_pipeline
        mock_pipeline_class.return_value = mock_pipeline_instance
        
        strategy = ChronosPalazzoStrategy(self.config)
        
        data = {'timestamp': pd.to_datetime(['2023-01-01']), 'close': [100], 'open': [90], 'high': [110], 'low': [80], 'volume': [1000]}
        sample_data = pd.DataFrame(data).set_index('timestamp')

        signal = strategy.get_signal(sample_data)
        self.assertIsNone(signal)
        mock_log_error.assert_called_once()

if __name__ == '__main__':
    unittest.main()
