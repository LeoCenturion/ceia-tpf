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
