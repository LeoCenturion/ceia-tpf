import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
from prophet import Prophet
import statsmodels.api as sm
from backtest_utils import (
    run_optimizations,
    fetch_historical_data,
    adjust_data_to_ubtc,
)


# Custom indicator for preprocessing
def normalized_price_change(series: np.ndarray, window: int = 1000) -> np.ndarray:
    """Calculates (price[i] - price[i-1]) / std(price[-window:])."""
    series = pd.Series(series)
    diff = series.diff()
    std_dev = series.rolling(window).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        result = diff / std_dev
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result.fillna(0).values


class ProphetStrategy(Strategy):
    refit_period = 100  # Refit the model every N bars
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        self.model = None
        self.forecast = None

    def next(self):
        # Refit model periodically
        if len(self.data.Close) > 1 and len(self.data.Close) % self.refit_period == 0:
            prophet_data = pd.DataFrame({"ds": self.data.index, "y": self.data.Close})
            self.model = Prophet()
            self.model.fit(prophet_data, progress=False)

        # Generate forecast if model is fitted
        if self.model:
            future = self.model.make_future_dataframe(
                periods=1, freq="H"
            )  # Assuming hourly data
            self.forecast = self.model.predict(future)

        price = self.data.Close[-1]
        if self.forecast is not None:
            forecast_price = self.forecast["yhat"].iloc[-1]
            if forecast_price > price and not self.position.is_long:
                self.buy(
                    sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit)
                )
            elif forecast_price < price and not self.position.is_short:
                self.sell(
                    sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit)
                )

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            "refit_period": trial.suggest_int("refit_period", 50, 200),
        }


class ARIMAStrategy(Strategy):
    p = 5
    d = 1
    q = 0
    refit_period = 24 * 30
    std_window = 24 * 30
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        self.model_fit = None
        self.processed_data = self.I(
            normalized_price_change, self.data.Close, self.std_window
        )

    def next(self):
        price = self.data.Close[-1]

        # Refit model periodically and if we have enough data
        if len(self.data) % self.refit_period == 0 and len(self.data) > self.std_window:
            print(f"retraining. IDX ${len(self.data)}")
            try:
                model = sm.tsa.ARIMA(
                    self.processed_data, order=(self.p, self.d, self.q)
                )
                self.model_fit = model.fit()
            except Exception:  # Catches convergence errors etc.
                self.model_fit = None

        if self.model_fit:
            try:
                forecast_processed = self.model_fit.forecast(steps=1)[0]
                if forecast_processed > 0 and not self.position.is_long:
                    self.buy(
                        sl=price * (1 - self.stop_loss),
                        tp=price * (1 + self.take_profit),
                    )
                elif forecast_processed < 0 and not self.position.is_short:
                    self.sell(
                        sl=price * (1 + self.stop_loss),
                        tp=price * (1 - self.take_profit),
                    )
            except Exception:
                pass  # Ignore forecast errors

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            "p": trial.suggest_int("p", 1, 10),
            "d": trial.suggest_int("d", 0, 2),
            "q": trial.suggest_int("q", 0, 5),
            "refit_period": trial.suggest_int("refit_period", 50, 200),
            "std_window": trial.suggest_int("std_window", 500, 1500),
        }


class SARIMAStrategy(Strategy):
    p, d, q = 5, 1, 0
    P, D, Q, s = (
        1,
        1,
        0,
        24,
    )  # Seasonal order, s=24 for daily seasonality on hourly data
    refit_period = 100
    std_window = 1000
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        self.model_fit = None
        self.processed_data = self.I(
            normalized_price_change, self.data.Close, self.std_window
        )

    def next(self):
        price = self.data.Close[-1]

        if len(self.data) % self.refit_period == 0 and len(self.data) > self.std_window:
            try:
                model = sm.tsa.SARIMAX(
                    self.processed_data,
                    order=(self.p, self.d, self.q),
                    seasonal_order=(self.P, self.D, self.Q, self.s),
                )
                self.model_fit = model.fit(disp=False)
            except Exception:
                self.model_fit = None

        if self.model_fit:
            try:
                forecast_processed = self.model_fit.forecast(steps=1)[0]
                if forecast_processed > 0 and not self.position.is_long:
                    self.buy(
                        sl=price * (1 - self.stop_loss),
                        tp=price * (1 + self.take_profit),
                    )
                elif forecast_processed < 0 and not self.position.is_short:
                    self.sell(
                        sl=price * (1 + self.stop_loss),
                        tp=price * (1 - self.take_profit),
                    )
            except Exception:
                pass

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            "p": trial.suggest_int("p", 1, 5),
            "d": trial.suggest_int("d", 0, 2),
            "q": trial.suggest_int("q", 0, 5),
            "P": trial.suggest_int("P", 0, 2),
            "D": trial.suggest_int("D", 0, 2),
            "Q": trial.suggest_int("Q", 0, 2),
            "s": trial.suggest_categorical("s", [12, 24, 48]),
            "refit_period": trial.suggest_int("refit_period", 50, 200),
            "std_window": trial.suggest_int("std_window", 500, 1500),
        }


def main():
    """Main function to run the optimization with default parameters."""
    strategies = {
        "ARIMAStrategy": ARIMAStrategy
        # "ProphetStrategy": ProphetStrategy
    }
    run_optimizations(
        strategies=strategies,
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2022-01-01T00:00:00Z",
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="Trading Strategies",
        n_trials_per_strategy=10,
    )


if __name__ == "__main__":
    # Run a single backtest for SARIMAStrategy
    print("Running single backtest for ARIMAStrategy...")
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2022-01-01T00:00:00Z",
    )
    data = adjust_data_to_ubtc(data)
    print(f"data len {len(data)}")
    bt = Backtest(data, ARIMAStrategy, cash=10_000, commission=0.002)
    stats = bt.run()
    print(stats)
    bt.plot()

    # print("\nStarting optimizations defined in main()...")
    # main()
