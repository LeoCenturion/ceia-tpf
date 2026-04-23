
import logging
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from src.modeling.chronos_pipeline_palazzo import PalazzoChronosBinaryClassificationPipeline

class ChronosPalazzoStrategy:
    def __init__(self, config):
        self.model_path = config['model_path']
        self.model = TimeSeriesPredictor.load(self.model_path)
        self.pipeline = PalazzoChronosBinaryClassificationPipeline(config)

    def get_signal(self, data: pd.DataFrame):
        try:
            # Rename columns
            data = data.rename(columns={
                'open': 'open_price',
                'high': 'High',
                'low': 'Low',
                'close': 'close_price',
                'volume': 'volume'
            })
            
            data['bar_return'] = (data['close_price'] / data['open_price']) - 1
            data['intra_bar_std'] = 0

            # Feature engineering
            features = self.pipeline.step_2_feature_engineering(data)
            
            # Prepare data for prediction
            features['item_id'] = 'btc'
            features['timestamp'] = features.index
            # The last row is the one we want to predict for, so it's removed from the training data
            train_data = TimeSeriesDataFrame.from_data_frame(
                features.iloc[:-1],
                id_column="item_id",
                timestamp_column="timestamp"
            )

            # The known covariates should include the features for the future timestamp
            known_covariates = TimeSeriesDataFrame.from_data_frame(
                features,
                id_column="item_id",
                timestamp_column="timestamp"
            )

            # Predict
            prediction = self.model.predict(train_data, known_covariates=known_covariates)
            
            # Generate signal
            if prediction['mean'].iloc[0] > 0:
                return 'BUY'
            else:
                return 'SELL'
        except Exception as e:
            logging.error(f"Error generating signal: {e}")
            return None

    def get_order_size(self, data: pd.DataFrame, signal: str) -> float:
        return 1.0

