import os

import mlflow
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

from backtest_utils import TrialStrategy, fetch_historical_data, run_optimizations


class ChronosStrategy(TrialStrategy):
    """
    A backtesting strategy using the Chronos time-series model for price prediction.
    """

    # --- Strategy Parameters (can be optimized) ---
    model_path = "amazon/chronos-t5-small"  # Model to use
    refit_every = 24 * 14  # Refit the model every 14 days (on hourly data)
    trade_threshold = 0.01  # % change required to trigger a trade (1%)
    min_train_bars = (
        refit_every * 2
    )  # Require at least 2 refit periods of data to train

    def init(self):
        """
        Initialize the strategy.
        """
        self.predictor = None
        self.periods_since_refit = np.inf  # Force refit on first valid occasion
        self.predictions_df = pd.DataFrame()
        self._last_prediction = None

        # --- Initial Fine-tuning ---
        full_df = self.data.df
        split_idx = int(len(full_df) * 0.7)
        self.backtest_start_index = split_idx

        initial_train_df = full_df.iloc[:split_idx]

        print(
            f"[{initial_train_df.index[-1]}] Performing initial fine-tuning on {split_idx} data points..."
        )

        # Prepare data for AutoGluon
        history_df = initial_train_df.copy()
        history_df["target"] = history_df["Close"]
        history_df["item_id"] = "series_0"
        history_df["timestamp"] = history_df.index.values

        train_data = TimeSeriesDataFrame.from_data_frame(
            history_df, id_column="item_id", timestamp_column="timestamp"
        )

        # Initialize and fit the predictor
        self.predictor = TimeSeriesPredictor(
            prediction_length=1, verbosity=0, freq="1h", target="target"
        )
        try:
            self.predictor.fit(
                train_data,
                hyperparameters={
                    "Chronos": {
                        "model_path": self.model_path,
                        "fine_tune_batch_size": 2048,
                        "batch_size": 2048,
                    }
                },
            )
            print(f"[{initial_train_df.index[-1]}] Initial fine-tuning complete.")
            self.periods_since_refit = 0  # Reset counter
        except Exception as e:
            print(f"[{initial_train_df.index[-1]}] Initial model fitting failed: {e}")
            self.predictor = None
            self.periods_since_refit = np.inf

    def next(self):
        """
        Called on each bar of data.
        """
        is_backtesting_phase = len(self.data) >= self.backtest_start_index
        if not is_backtesting_phase:
            return
        # --- Model Refitting ---
        if self.periods_since_refit >= self.refit_every:
            print(f"[{self.data.index[-1]}] Refitting Chronos model...")

            # Prepare data for AutoGluon. Use all available columns as features.
            history_df = self.data.df.copy()
            history_df["target"] = history_df["Close"]  # Define the target column
            history_df["item_id"] = "series_0"
            history_df["timestamp"] = history_df.index.values

            train_data = TimeSeriesDataFrame.from_data_frame(
                history_df, id_column="item_id", timestamp_column="timestamp"
            )

            # Initialize and fit the predictor
            self.predictor = TimeSeriesPredictor(
                prediction_length=1, verbosity=0, freq="1h", target="target"
            )
            try:
                self.predictor.fit(
                    train_data,
                    hyperparameters={
                        "Chronos": {
                            "model_path": self.model_path,
                            "fine_tune_batch_size": 2048,
                            "batch_size": 2048,
                        }
                    },
                )
                print(f"[{self.data.index[-1]}] Model refit complete.")
                self.periods_since_refit = 0  # Reset counter
            except Exception as e:
                print(f"[{self.data.index[-1]}] Model fitting failed: {e}")
                self.predictor = None  # Ensure predictor is None if fit fails

        if self._last_prediction is not None:
            prediction_timestamp = self._last_prediction.index.get_level_values(
                "timestamp"
            )[0]
            if prediction_timestamp == self.data.index[-1]:
                log_entry = self._last_prediction.iloc[0].to_dict()
                log_entry["actual_close"] = self.data.Close[-1]
                new_row = pd.DataFrame(log_entry, index=[prediction_timestamp])
                self.predictions_df = pd.concat([self.predictions_df, new_row])

        # --- Prediction and Trading Logic ---
        if self.predictor:
            # Prepare data for prediction
            current_data_df = self.data.df.copy()
            current_data_df["target"] = current_data_df["Close"]
            current_data_df["item_id"] = "series_0"
            current_data_df["timestamp"] = current_data_df.index.values

            # Predict the next closing price
            prediction = self.predictor.predict(current_data_df)
            self._last_prediction = prediction
            predicted_price = prediction["mean"].values[0]
            current_price = self.data.Close[-1]

            # --- Trading Signal Generation ---
            upper_bound = current_price * (1 + self.trade_threshold)
            lower_bound = current_price * (1 - self.trade_threshold)

            if predicted_price > upper_bound:
                if self.position.is_short:
                    self.position.close()
                if not self.position.is_long:
                    self.buy()
            elif predicted_price < lower_bound:
                if self.position.is_long:
                    self.position.close()
                if not self.position.is_short:
                    self.sell()

        self.periods_since_refit += 1

    def save_artifacts(self, trial, stats, bt):
        super().save_artifacts(trial, stats, bt)

        predictions_filename = f"predictions_trial_{trial.number}.csv"
        if not self.predictions_df.empty:
            self.predictions_df.to_csv(predictions_filename)
            if os.path.exists(predictions_filename):
                mlflow.log_artifact(predictions_filename)
                os.remove(predictions_filename)

    @classmethod
    def get_optuna_params(cls, trial):
        """
        Define the hyperparameter space for Optuna.
        """
        return {
            "model_path": trial.suggest_categorical(
                "model_path",
                ["amazon/chronos-t5-tiny"],
            ),
            "refit_every": trial.suggest_int("refit_every", 24 * 7, 24 * 14, step=24),
            "trade_threshold": trial.suggest_float(
                "trade_threshold", 0.005, 0.01, log=True
            ),
        }


# AI add a new strategy that reads a csv that contains the prediction from a previous run. It contains the mean and 10 perentiles 0.1, 0.2, ...etc AI!
def main():
    """Main function to run optimization for the Chronos strategy."""
    strategies = {
        "ChronosStrategy": ChronosStrategy,
    }
    # Note: Chronos backtests are very slow. Consider a low number of trials
    # and a single job (n_jobs=1) if using a GPU.
    run_optimizations(
        strategies=strategies,
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2018-01-01T00:00:00Z",
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="Chronos Strategy Optimization v1.0",
        n_trials_per_strategy=1,
        n_jobs=1,  # Chronos/AutoGluon can be heavy, especially on GPU
    )


if __name__ == "__main__":
    main()
