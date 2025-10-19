import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

from backtest_utils import fetch_historical_data, run_optimizations


class ChronosStrategy(Strategy):
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

    def next(self):
        """
        Called on each bar of data.
        """
        # --- Model Refitting ---
        # Refit the model periodically, but only if we have enough data
        if (
            self.periods_since_refit >= self.refit_every
            and len(self.data.df) >= self.min_train_bars
        ):
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
                    hyperparameters={"Chronos": {"model_path": self.model_path}},
                )
                print(f"[{self.data.index[-1]}] Model refit complete.")
                self.periods_since_refit = 0  # Reset counter
            except Exception as e:
                print(f"[{self.data.index[-1]}] Model fitting failed: {e}")
                self.predictor = None  # Ensure predictor is None if fit fails

        # --- Prediction and Trading Logic ---
        if self.predictor:
            # Prepare data for prediction
            current_data_df = self.data.df.copy()
            current_data_df["target"] = current_data_df["Close"]
            current_data_df["item_id"] = "series_0"
            current_data_df["timestamp"] = current_data_df.index.values

            # Predict the next closing price
            prediction = self.predictor.predict(current_data_df)
            predicted_price = prediction["mean"].values[0]
            current_price = self.data.Close[-1]

            # --- Trading Signal Generation ---
            upper_bound = current_price * (1 + self.trade_threshold)
            lower_bound = current_price * (1 - self.trade_threshold)

            if predicted_price > upper_bound:
                # If we predict a significant price increase, close any short and go long
                if self.position.is_short:
                    self.position.close()
                if not self.position.is_long:
                    self.buy()

            elif predicted_price < lower_bound:
                # If we predict a significant price decrease, close any long and go short
                if self.position.is_long:
                    self.position.close()
                if not self.position.is_short:
                    self.sell()

        self.periods_since_refit += 1

    @classmethod
    def get_optuna_params(cls, trial):
        """
        Define the hyperparameter space for Optuna.
        """
        return {
            "model_path": trial.suggest_categorical(
                "model_path",
                ["amazon/chronos-t5-tiny", "amazon/chronos-t5-small"],
            ),
            "refit_every": trial.suggest_int("refit_every", 24 * 7, 24 * 30, step=24),
            "trade_threshold": trial.suggest_float(
                "trade_threshold", 0.005, 0.05, log=True
            ),
        }


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
        start_date="2023-01-01T00:00:00Z",
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="Chronos Strategy Optimization",
        n_trials_per_strategy=10,
        n_jobs=1,  # Chronos/AutoGluon can be heavy, especially on GPU
    )


if __name__ == "__main__":
    main()
