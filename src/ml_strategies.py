import numpy as np
import pandas as pd
from backtesting import Strategy
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from backtest_utils import sma, ewm, rsi_indicator, std, run_optimizations


# Helper functions for indicators to be used with `self.I`
def bbands(close_prices: np.ndarray, n: int = 20, std_dev: int = 2) -> tuple:
    """Calculates Bollinger Bands."""
    series = pd.Series(close_prices)
    middle_band = sma(series, n)
    upper_band = middle_band + std_dev * std(series, n)
    lower_band = middle_band - std_dev * std(series, n)
    return upper_band, lower_band


def macd(close_prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
    """Calculates MACD and its signal line."""
    series = pd.Series(close_prices)
    ema_fast = ewm(series, span=fast_period)
    ema_slow = ewm(series, span=slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = ewm(macd_line, span=signal_period)
    return macd_line, signal_line


def lagged_returns(close_prices: np.ndarray, lag: int = 1) -> np.ndarray:
    """Calculates lagged percentage returns."""
    series = pd.Series(close_prices)
    return series.pct_change(periods=lag).fillna(0).values


class SVMStrategy(Strategy):
    """
    A trading strategy that uses a Support Vector Regressor (SVR) to predict
    the magnitude of the next price movement.
    """
    # --- Strategy Parameters ---
    refit_period = 24 * 7
    lookback_length = 24 * 30 * 3
    threshold = 0.001  # Trade signal threshold (0.1%)

    # --- SVR Hyperparameters ---
    kernel = 'rbf'
    C = 1.0
    gamma = 'scale'
    epsilon = 0.1

    def init(self):
        # --- Feature Engineering ---
        self.rsi = self.I(rsi_indicator, self.data.Close, n=14)
        self.upper_band, self.lower_band = self.I(bbands, self.data.Close, n=20)
        self.macd, self.signal_line = self.I(macd, self.data.Close)
        self.lag_1 = self.I(lagged_returns, self.data.Close, lag=1)
        self.lag_3 = self.I(lagged_returns, self.data.Close, lag=3)
        self.lag_6 = self.I(lagged_returns, self.data.Close, lag=6)

        # --- Model and Scaler Initialization ---
        self.model = None
        self.scaler = None

    def next(self):
        # --- Model Training ---
        if len(self.data.Close) > self.lookback_length and len(self.data.Close) % self.refit_period == 0:
            print(f"Refitting SVR model at bar: {len(self.data.Close)}")

            # 1. Create a DataFrame with features
            df = pd.DataFrame({
                'rsi': self.rsi[-self.lookback_length:],
                'upper_band': self.upper_band[-self.lookback_length:],
                'lower_band': self.lower_band[-self.lookback_length:],
                'macd': self.macd[-self.lookback_length:],
                'signal_line': self.signal_line[-self.lookback_length:],
                'lag_1': self.lag_1[-self.lookback_length:],
                'lag_3': self.lag_3[-self.lookback_length:],
                'lag_6': self.lag_6[-self.lookback_length:],
            })

            # 2. Create and align the target variable (next percentage price change)
            df['target'] = pd.Series(self.data.Close[-self.lookback_length:]).pct_change().shift(-1)

            # 3. Drop rows with any NaN values
            df.dropna(inplace=True)

            # 4. Separate features (X) and target (y)
            X = df.drop('target', axis=1)
            y = df['target']

            if len(X) == 0:
                self.model = None
                return

            # 5. Scale features and train model
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            self.model = SVR(kernel=self.kernel, C=self.C, gamma=self.gamma, epsilon=self.epsilon)
            self.model.fit(X_scaled, y)

        # --- Signal Generation and Trading ---
        if self.model:
            current_features = np.array([
                self.rsi[-1], self.upper_band[-1], self.lower_band[-1],
                self.macd[-1], self.signal_line[-1], self.lag_1[-1],
                self.lag_3[-1], self.lag_6[-1]
            ]).reshape(1, -1)

            if np.isnan(current_features).any():
                return

            current_features_scaled = self.scaler.transform(current_features)
            prediction = self.model.predict(current_features_scaled)[0]

            # Buy if predicted percentage change is above the positive threshold
            if prediction > self.threshold and not self.position.is_long:
                self.buy()
            # Sell if predicted percentage change is below the negative threshold
            elif prediction < -self.threshold and not self.position.is_short:
                self.sell()

    @classmethod
    def get_optuna_params(cls, trial):
        """Define the hyperparameter search space for Optuna."""
        return {
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "gamma": trial.suggest_categorical("gamma", ['scale', 'auto']),
            "epsilon": trial.suggest_float("epsilon", 1e-2, 1e-1, log=True),
            "threshold": trial.suggest_float("threshold", 1e-4, 1e-2, log=True),
            "refit_period": trial.suggest_categorical("refit_period", [24 * 7, 24 * 14]),
            "lookback_length": trial.suggest_categorical("lookback_length", [24 * 90, 24 * 180]),
        }




def main():
    """Main function to run the optimization with default parameters."""
    strategies = {
        "SVMStrategy": SVMStrategy
    }
    run_optimizations(
        strategies=strategies,
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2023-01-01T00:00:00Z",
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="Trading Strategies 2",
        n_trials_per_strategy=1
    )


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from backtesting import Strategy
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from backtest_utils import (
    sma,
    rsi_indicator,
    std,
    momentum_indicator,
    run_optimizations,
    fetch_historical_data,
    adjust_data_to_ubtc,
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


class SVRStrategy(Strategy):
    # SVR Hyperparameters
    kernel = 'rbf'
    C = 1.0
    gamma = 'scale'
    epsilon = 0.1

    # Strategy Parameters
    refit_period = 24 * 7  # Refit weekly
    lookback_length = 24 * 30 * 3  # 3 months of hourly data
    stop_loss = 0.05
    take_profit = 0.10
    threshold = 0.001

    def init(self):
        self.model = None
        self.scaler = StandardScaler()

        # Feature and target creation
        self.features = self.I(_create_features, self.data)
        self.target = self.data.Close.to_series().pct_change(1).shift(-1).bfill().values

    def next(self):
        # Retrain the model periodically
        if len(self.data) > self.lookback_length and len(self.data) % self.refit_period == 0:
            X_train_raw = self.features[-self.lookback_length:-1]
            y_train = self.target[-self.lookback_length:-1]

            X_train = self.scaler.fit_transform(X_train_raw)
            self.model = SVR(kernel=self.kernel, C=self.C, gamma=self.gamma, epsilon=self.epsilon)
            self.model.fit(X_train, y_train)

        # Make prediction and trade if the model is trained
        if self.model:
            current_features = self.features[-1].reshape(1, -1)
            scaled_features = self.scaler.transform(current_features)
            prediction = self.model.predict(scaled_features)[0]

            price = self.data.Close[-1]
            if prediction > self.threshold and not self.position.is_long:
                self.buy(sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit))
            elif prediction < -self.threshold and not self.position.is_short:
                self.sell(sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit))

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            "kernel": trial.suggest_categorical("kernel", ['rbf', 'linear', 'poly']),
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "gamma": trial.suggest_categorical("gamma", ['scale', 'auto']),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 1e-1, log=True),
        }


class RandomForestStrategy(Strategy):
    # Random Forest Hyperparameters
    n_estimators = 100
    max_depth = 10
    min_samples_split = 10
    max_features = 0.5

    # Strategy Parameters
    refit_period = 24 * 7
    lookback_length = 24 * 30 * 3
    stop_loss = 0.05
    take_profit = 0.10
    threshold = 0.001

    def init(self):
        self.model = None
        self.scaler = StandardScaler()

        self.features = self.I(_create_features, self.data)
        self.target = self.data.Close.to_series().pct_change(1).shift(-1).bfill().values

    def next(self):
        if len(self.data) > self.lookback_length and len(self.data) % self.refit_period == 0:
            X_train_raw = self.features[-self.lookback_length:-1]
            y_train = self.target[-self.lookback_length:-1]

            X_train = self.scaler.fit_transform(X_train_raw)

            self.model = RandomForestRegressor(
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

            price = self.data.Close[-1]
            if prediction > self.threshold and not self.position.is_long:
                self.buy(sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit))
            elif prediction < -self.threshold and not self.position.is_short:
                self.sell(sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit))

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 30, step=5),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        }


def main():
    """Main function to run optimization for ML strategies."""
    strategies = {
        "SVRStrategy": SVRStrategy,
        "RandomForestStrategy": RandomForestStrategy,
    }
    run_optimizations(
        strategies=strategies,
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2022-01-01T00:00:00Z",
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="ML Trading Strategies",
        n_trials_per_strategy=10,
        n_jobs=4 # Using fewer jobs as ML models can be memory intensive
    )


if __name__ == "__main__":
    main()
