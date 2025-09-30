import pandas as pd
import numpy as np
from backtesting import Strategy
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from backtest_utils import (
    sma,
    rsi_indicator,
    std,
    momentum_indicator,
    run_classification_optimizations,
)


def _create_features(data):
    """Creates a feature matrix from the price data."""
    close = pd.Series(data.Close)

    # Technical Indicators
    rsi = rsi_indicator(close, n=14)
    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    momentum = momentum_indicator(close, window=10)

    # Bollinger Bands
    std20 = std(close, 20)
    upper_band = sma20 + (std20 * 2)
    lower_band = sma20 - (std20 * 2)

    # Price-derived features
    volatility = std(close, 20)
    lagged_return = close.pct_change(1)

    # Combine into a DataFrame for easier manipulation
    df = pd.DataFrame({
        'RSI': rsi,
        'SMA20': sma20,
        'SMA50': sma50,
        'Momentum': momentum,
        'Upper_Band': upper_band,
        'Lower_Band': lower_band,
        'Volatility': volatility,
        'Lagged_Return': lagged_return
    })

    # Create relative features
    df['SMA_Ratio'] = df['SMA20'] / df['SMA50']
    df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['SMA20']

    # Select and clean final features
    final_features = df[['RSI', 'Momentum', 'Volatility', 'Lagged_Return', 'SMA_Ratio', 'BB_Width']]
    return final_features.bfill().ffill().values


class SVCStrategy(Strategy):
    # SVC Hyperparameters
    kernel = 'rbf'
    C = 1.0
    gamma = 'scale'

    # Strategy Parameters
    refit_period = 24 * 7  # Refit weekly
    lookback_length = 24 * 30 * 3  # 3 months of hourly data

    def init(self):
        self.model = None
        self.scaler = StandardScaler()

        # Feature and target creation
        self.features = self.I(_create_features, self.data)
        self.target = (self.data.Close.to_series().pct_change(1).shift(-1) > 0).astype(int).bfill().values

        # For F1 score calculation
        self.y_true = []
        self.y_pred = []

    def next(self):
        # Retrain the model periodically
        if len(self.data) > self.lookback_length and len(self.data) % self.refit_period == 0:
            X_train_raw = self.features[-self.lookback_length:-1]
            y_train = self.target[-self.lookback_length:-1]

            X_train = self.scaler.fit_transform(X_train_raw)
            self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, probability=True)
            self.model.fit(X_train, y_train)

        # Make prediction and trade if the model is trained
        if self.model:
            current_features = self.features[-1].reshape(1, -1)
            scaled_features = self.scaler.transform(current_features)
            prediction = self.model.predict(scaled_features)[0]

            true_label = self.target[len(self.data.Close) - 1]
            self.y_true.append(true_label)
            self.y_pred.append(prediction)

            if prediction == 1 and not self.position.is_long:
                self.buy()
            elif prediction == 0 and self.position:
                self.position.close()

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            "kernel": trial.suggest_categorical("kernel", ['rbf', 'linear', 'poly']),
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "gamma": trial.suggest_categorical("gamma", ['scale', 'auto']),
        }


class RandomForestClassifierStrategy(Strategy):
    # Random Forest Hyperparameters
    n_estimators = 100
    max_depth = 10
    min_samples_split = 10
    max_features = 0.5

    # Strategy Parameters
    refit_period = 24 * 7
    lookback_length = 24 * 30 * 3

    def init(self):
        self.model = None
        self.scaler = StandardScaler()

        self.features = self.I(_create_features, self.data)
        self.target = (self.data.Close.to_series().pct_change(1).shift(-1) > 0).astype(int).bfill().values

        self.y_true = []
        self.y_pred = []

    def next(self):
        if len(self.data) > self.lookback_length and len(self.data) % self.refit_period == 0:
            X_train_raw = self.features[-self.lookback_length:-1]
            y_train = self.target[-self.lookback_length:-1]

            X_train = self.scaler.fit_transform(X_train_raw)

            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)

        if self.model:
            current_features = self.features[-1].reshape(1, -1)
            scaled_features = self.scaler.transform(current_features)
            prediction = self.model.predict(scaled_features)[0]

            true_label = self.target[len(self.data.Close) - 1]
            self.y_true.append(true_label)
            self.y_pred.append(prediction)

            if prediction == 1 and not self.position.is_long:
                self.buy()
            elif prediction == 0 and self.position:
                self.position.close()

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 30, step=5),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        }


def main():
    """Main function to run optimization for ML classification strategies."""
    strategies = {
        "SVCStrategy": SVCStrategy,
        "RandomForestClassifierStrategy": RandomForestClassifierStrategy,
    }
    run_classification_optimizations(
        strategies=strategies,
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2022-01-01T00:00:00Z",
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="ML Classification Strategies",
        n_trials_per_strategy=10,
        n_jobs=4
    )


if __name__ == "__main__":
    main()
