import pandas as pd
import numpy as np
from backtesting import Strategy
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.backtesting.backtesting import run_optimizations  # pylint: disable=no-name-in-module

# --- Helper functions from src/xgboost_price_reversal_palazzo.py ---


def aggregate_to_volume_bars(df, volume_threshold=50000):
    """
    Aggregates time-series data into volume bars based on a volume threshold.
    """
    bars = []
    current_bar_data = []
    cumulative_volume = 0

    for _, row in df.iterrows():
        current_bar_data.append(row)
        cumulative_volume += row["volume"]
        if cumulative_volume >= volume_threshold:
            bar_df = pd.DataFrame(current_bar_data)

            open_time = bar_df.index[0]
            close_time = bar_df.index[-1]
            open_price = bar_df["close"].iloc[0]
            close_price = bar_df["close"].iloc[-1]

            bar_log_returns = np.log(
                bar_df["close"] / bar_df["close"].shift(1)
            ).dropna()
            intra_bar_std = bar_log_returns.std()

            bars.append(
                {
                    "open_time": open_time,
                    "close_time": close_time,
                    "open_price": open_price,
                    "close_price": close_price,
                    "total_volume": cumulative_volume,
                    "intra_bar_std": intra_bar_std if len(bar_log_returns) > 1 else 0,
                }
            )
            current_bar_data = []
            cumulative_volume = 0

    volume_bars_df = pd.DataFrame(bars)
    if volume_bars_df.empty:
        return volume_bars_df

    volume_bars_df["bar_return"] = (
        volume_bars_df["close_price"] / volume_bars_df["open_price"]
    ) - 1
    return volume_bars_df


def create_labels(df, tau=0.35):
    """
    Creates target labels based on the triple-barrier method variation.
    """
    df["label"] = 0
    df["next_bar_return"] = df["bar_return"].shift(-1)
    cond1 = df["next_bar_return"] >= 0
    cond2 = df["next_bar_return"] >= (df["bar_return"] + df["intra_bar_std"] * tau)
    df.loc[cond1 & cond2, "label"] = 1
    df.dropna(subset=["next_bar_return"], inplace=True)
    df.drop(columns=["next_bar_return"], inplace=True)
    return df


def create_features(df):
    """
    Creates simple features for the model based on Palazzo's work.
    """
    df["feature_return_lag_1"] = df["bar_return"].shift(1)
    df["feature_volatility_lag_1"] = df["intra_bar_std"].shift(1)
    df["feature_rolling_mean_return_5"] = (
        df["bar_return"].shift(1).rolling(window=5).mean()
    )
    df["feature_rolling_std_return_5"] = (
        df["bar_return"].shift(1).rolling(window=5).std()
    )
    df.dropna(inplace=True)
    return df


class XGBoostPriceReversalPalazzoStrategy(
    Strategy
):  # pylint: disable=attribute-defined-outside-init
    # Default Hyperparameters (to be tuned by Optuna)
    volume_threshold = 37798
    tau = 1.2688331479071624
    n_estimators = 281
    learning_rate = 0.0010163133112639452
    max_depth = 7
    subsample = 0.8486986034061127
    colsample_bytree = 0.7033426743185334
    gamma = 2.035537039974458e-05
    min_child_weight = 10

    # Strategy Parameters
    refit_period = 100  # Refit every 100 volume bars
    lookback_length = 500  # Use 500 volume bars for training

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 1. Pre-process data to get volume bars and features.
        # This is computationally intensive and done once per backtest run.
        minute_data = self.data.df.copy()
        minute_data.rename(columns={"Close": "close", "Volume": "volume"}, inplace=True)

        volume_bars = aggregate_to_volume_bars(minute_data, self.volume_threshold)
        if volume_bars.empty:
            self.processed_data = pd.DataFrame()
            return

        labeled_bars = create_labels(volume_bars, self.tau)
        self.processed_data = create_features(labeled_bars)

        if self.processed_data.empty:
            return

        # Set index to map time-based backtest to event-based bars
        self.processed_data.set_index("close_time", inplace=True, drop=False)
        self.volume_bar_indices = {
            timestamp: i for i, timestamp in enumerate(self.processed_data.index)
        }

        # 2. Initialize strategy state
        self.model = None
        self.scaler = StandardScaler()
        self.in_trade = False

    def next(self):
        if not hasattr(self, "processed_data") or self.processed_data.empty:
            return

        # Check if the current time bar corresponds to the completion of a volume bar
        current_timestamp = self.data.index[-1]
        if current_timestamp not in self.volume_bar_indices:
            return

        # A volume bar has completed. Get its index.
        bar_idx = self.volume_bar_indices[current_timestamp]

        # "Sell on the next bar's close" logic
        if self.in_trade:
            self.position.close()
            self.in_trade = False

        # Periodically refit the model
        # print(f"bar_id {bar_idx}; loopback_length {self.lookback_length}; % = {(bar_idx - self.lookback_length) % self.refit_period}")
        if (bar_idx - self.lookback_length) % self.refit_period == 0:
            # print("retraining")
            end_idx = bar_idx
            start_idx = max(0, end_idx - self.lookback_length)
            train_df = self.processed_data.iloc[start_idx:end_idx]

            y_train = train_df["label"]
            features = [col for col in train_df.columns if "feature_" in col]
            X_train_raw = train_df[features]

            if y_train.nunique() < 2:
                return  # Not enough classes to train, skip refitting

            X_train = self.scaler.fit_transform(X_train_raw)

            # Handle class imbalance
            scale_pos_weight = (
                (y_train == 0).sum() / (y_train == 1).sum()
                if (y_train == 1).sum() > 0
                else 1
            )

            self.model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cuda",
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                gamma=self.gamma,
                min_child_weight=self.min_child_weight,
                seed=42,
                scale_pos_weight=scale_pos_weight,
            )
            self.model.fit(X_train, y_train)

        # Make prediction if model is trained
        if self.model:
            current_features_df = self.processed_data.iloc[[bar_idx]]
            features = [col for col in current_features_df.columns if "feature_" in col]
            X_current_raw = current_features_df[features]

            scaled_features = self.scaler.transform(X_current_raw)
            prediction = self.model.predict(scaled_features)[0]
            # print(f'predicted {prediction}')
            if prediction == 1 and not self.position:
                self.buy()
                self.in_trade = True

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            "volume_threshold": trial.suggest_int("volume_threshold", 37798, 37798),
            "tau": trial.suggest_float("tau", 1.2688331479071624, 1.2688331479071624),
            "n_estimators": trial.suggest_int("n_estimators", 281, 281),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.0010163133112639452, 0.0010163133112639452
            ),
            "max_depth": trial.suggest_int("max_depth", 7, 7),
            "subsample": trial.suggest_float(
                "subsample", 0.8486986034061127, 0.8486986034061127
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.7033426743185334, 0.7033426743185334
            ),
            "gamma": trial.suggest_float(
                "gamma", 2.035537039974458e-05, 2.035537039974458e-05
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 10),
            "refit_period": trial.suggest_int("refit_period", 24, 24, step=25),
            "lookback_length": trial.suggest_int(
                "lookback_length", 4500, 4500, step=100
            ),
        }


def main():
    """Main function to run optimization for the Palazzo strategy."""
    strategies = {
        "XGBoostPriceReversalPalazzoStrategy": XGBoostPriceReversalPalazzoStrategy,
    }
    # This strategy requires high-frequency data (e.g., 1-minute) to build volume bars.
    run_optimizations(
        strategies=strategies,
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv",
        start_date="2020-01-01T00:00:00Z",
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="Palazzo Strategy v1",
        timeframe="1m",
        n_trials_per_strategy=1,
        n_jobs=1,  # Use a single job for GPU-based training to avoid conflicts
    )


if __name__ == "__main__":
    main()
