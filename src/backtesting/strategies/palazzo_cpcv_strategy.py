import pandas as pd
from src.backtesting.backtesting import TrialStrategy
from src.modeling.xgboost_pipeline_palazzo import PalazzoXGBoostPipeline

class PalazzoXGBoostCPCVStrategy(TrialStrategy):
    """
    A strategy that uses a pre-trained PalazzoXGBoostPipeline for making predictions
    within the Combinatorial Purged Cross-Validation (CPCV) framework.
    """
    pipeline_config = {}
    model_cls = None
    model_params = {}
    trained_pipeline = None # Add trained_pipeline as a class variable

    def init(self):
        """
        Initializes the strategy. The trained pipeline is passed via _params.
        """
        self.trained_pipeline = self._params.get("trained_pipeline")
        if self.trained_pipeline is None:
            raise ValueError("PalazzoXGBoostCPCVStrategy requires a 'trained_pipeline' in _params.")

        # This strategy works with volume bars, which are event-based. The backtester is time-based.
        # The pipeline's predict method will need a window of time-based data to generate
        # the necessary features and make a prediction for the current bar.
        # We'll define a lookback window based on the number of time bars needed.
        self.window_size = self._params.get("pipeline_config", {}).get("prediction_window_size", 1000)


    def next(self):
        """
        This method is called for each time bar in the test set.
        """
        # Ensure we have enough data in the current window to generate features
        if len(self.data.Close) < self.window_size:
            # Not enough data, do nothing.
            self.signal = 0
            return

        # Extract the window required by the pipeline's predict method
        # The pipeline will handle the conversion to volume bars internally
        data_window = self.data.df.iloc[-self.window_size:]
        
        # The pipeline expects specific column names
        data_window = data_window.rename(columns={"Close": "close", "Volume": "volume"})

        # Generate a prediction using the pre-trained pipeline
        # The pipeline's predict method should return a single signal (0 or 1)
        prediction = self.trained_pipeline.predict(data_window)

        self.signal = prediction
        
        # The Palazzo strategy logic is to buy on signal and sell on the next bar's close.
        # In this simplified CPCV context, we just register the signal.
        # The runner will calculate returns based on holding for one period.
        if prediction == 1:
            self.buy()
        else:
            # We close any open position if the signal is not to buy.
            if self.position:
                self.position.close()
