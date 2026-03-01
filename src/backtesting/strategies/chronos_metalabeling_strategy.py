import pandas as pd
from src.backtesting.backtesting import TrialStrategy
from src.modeling.chronos_metalabeling_pipeline import ChronosMetaLabelingPipeline

class ChronosMetaLabelingCPCVStrategy(TrialStrategy):
    # Define parameters as class variables for the backtesting library
    pipeline_config = {}
    model_cls = None
    model_params = {}
    trained_pipeline = None # Add trained_pipeline as a class variable
    def init(self):
        # The 'trained_pipeline' is expected to be passed in _params by the CPCV runner
        self.trained_pipeline = self._params.get("trained_pipeline")
        if self.trained_pipeline is None:
            raise ValueError("ChronosMetaLabelingCPCVStrategy requires a 'trained_pipeline' in _params.")

        # Use the class variable directly
        self.window_size = self._params.get("pipeline_config", {}).get("chronos_window_size", 128)


    def next(self):
        # The `self.data` in `next` is an expanding window of `test_data` from `TrialStrategy.predict`.
        # We need to get a sufficient window for Chronos feature engineering.

        # Ensure we have enough data in the current window to generate features
        if len(self.data.Close) < self.window_size:
            # Not enough data to generate features, do nothing.
            return

        # Extract the window required by the pipeline's predict method
        data_window = self.data.iloc[-self.window_size:]

        # Generate a prediction using the pre-trained pipeline
        prediction = self.trained_pipeline.predict(data_window)

        if prediction == 1:
            self.buy()
