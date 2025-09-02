import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
from prophet import Prophet
import statsmodels.api as sm
from pykalman import KalmanFilter

try:
    from arch import arch_model
except ImportError:
    arch_model = None
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


def kalman_filter_indicator(series: np.ndarray) -> np.ndarray:
    """Applies a Kalman filter to a time series."""
    # Using a simple KF configuration for smoothing
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    (filtered_state_means, _) = kf.filter(series)
    return filtered_state_means.flatten()


class ProphetStrategy(Strategy):
    """
    Intractable. Prophet doesn't take data when predicting, so if we train every 30 days then all 30 days will have the same prediction.
    If we train every hour then it takes more than a day do backtest with hourly data from 2022 to 2025.
    """
    refit_period = 24 * 30  # Refit the model every N bars
    stop_loss = 0.05
    take_profit = 0.10
    lookback_length = 24 * 30 * 1
    def init(self):
        self.model = None
        self.forecast = None

    def next(self):
        if len(self.data.Close) > 1 and len(self.data.Close) % self.refit_period == 0:
            print(f"Refitting, idx {len(self.data)} len {len(self.data.index[-self.lookback_length:])}")
            prophet_data = pd.DataFrame({"ds": self.data.index[-self.lookback_length:], "y": self.data.Close[-self.lookback_length:]})
            self.model = Prophet()
            self.model.fit(prophet_data)
            print(f"Model refitted")

        # Generate forecast if model is fitted
        if self.model:
            future = self.model.make_future_dataframe(
                periods=1, freq="h", include_history=False
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
            "refit_period": trial.suggest_int("refit_period", 24 * 5, 24 * 30 * 2),
        }


class ARIMAStrategy(Strategy):
    p = 12
    d = 1
    q = 12
    refit_period = 24 * 10
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
            "refit_period": trial.suggest_int("refit_period", 24 * 5, 24 * 30),
            "std_window": trial.suggest_categorical("std_window", [24 * 30, 24 * 30 * 2, 24 * 30 * 3]),
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


class KalmanARIMAStrategy(Strategy):
    p = 12
    d = 1
    q = 12
    refit_period = 24 * 10
    std_window = 24 * 30
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        self.model_fit = None
        processed_data = self.I(
            normalized_price_change, self.data.Close, self.std_window
        )
        self.kalman_filtered_data = self.I(kalman_filter_indicator, processed_data)

    def next(self):
        price = self.data.Close[-1]

        # Refit model periodically and if we have enough data
        if len(self.data) % self.refit_period == 0 and len(self.data) > self.std_window:
            print(f"retraining KalmanARIMA. IDX ${len(self.data)}")
            try:
                model = sm.tsa.ARIMA(
                    self.kalman_filtered_data, order=(self.p, self.d, self.q)
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
            "refit_period": trial.suggest_int("refit_period", 24 * 5, 24 * 30),
            "std_window": trial.suggest_int("std_window", 24 * 30,  24 * 30 * 3),
        }


class ARIMAGARCHStrategy(Strategy):
    # ARIMA params
    p, d, q = 12, 1, 12
    # GARCH params
    g_p, g_q = 1, 1

    refit_period = 24 * 10
    std_window = 24 * 30
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        self.arima_fit = None
        self.garch_fit = None
        self.processed_data = self.I(
            normalized_price_change, self.data.Close, self.std_window
        )

    def next(self):
        price = self.data.Close[-1]

        # Refit models periodically
        if len(self.data) % self.refit_period == 0 and len(self.data) > self.std_window:
            print(f"Retraining ARIMAGARCH. IDX ${len(self.data)}")
            try:
                # 1. Fit ARIMA model
                arima_model = sm.tsa.ARIMA(
                    self.processed_data, order=(self.p, self.d, self.q)
                )
                self.arima_fit = arima_model.fit()

                # 2. Fit GARCH on ARIMA residuals
                residuals = self.arima_fit.resid
                garch_model = arch_model(residuals, p=self.g_p, q=self.g_q)
                self.garch_fit = garch_model.fit(disp="off")

            except Exception:
                self.arima_fit = None
                self.garch_fit = None

        if self.arima_fit:
            try:
                # Forecast mean from ARIMA
                forecast_mean = self.arima_fit.forecast(steps=1)[0]

                if forecast_mean > 0 and not self.position.is_long:
                    self.buy(
                        sl=price * (1 - self.stop_loss),
                        tp=price * (1 + self.take_profit),
                    )
                elif forecast_mean < 0 and not self.position.is_short:
                    self.sell(
                        sl=price * (1 + self.stop_loss),
                        tp=price * (1 - self.take_profit),
                    )
            except Exception:
                pass

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            "p": trial.suggest_int("p", 1, 12),
            "d": trial.suggest_int("d", 0, 2),
            "q": trial.suggest_int("q", 0, 12),
            "g_p": trial.suggest_int("g_p", 1, 5),
            "g_q": trial.suggest_int("g_q", 1, 5),
            "refit_period": trial.suggest_int("refit_period", 24 * 5, 24 * 30),
            "std_window": trial.suggest_categorical(
                "std_window", [24 * 30, 24 * 30 * 2, 24 * 30 * 3]
            ),
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
    data.sort_index(inplace=True)
    print(f"data len {len(data)}")

    chunk_size = 100
    num_chunks_to_test = 5  # Limiting to 5 chunks to keep the test reasonably short
    stats_list = []

    print(f"Splitting data into {num_chunks_to_test} chunks of {chunk_size} hours and averaging backtest stats...")

    for i in range(num_chunks_to_test):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        if end_idx > len(data):
            break

        chunk_data = data.iloc[start_idx:end_idx]
        print(f"\n--- Backtesting on Chunk {i+1}/{num_chunks_to_test} (Index: {start_idx}-{end_idx}) ---")

        bt = Backtest(chunk_data, ProphetStrategy, cash=10_000, commission=0.002)
        try:
            stats = bt.run()
            stats_list.append(stats)
            print(f"Chunk {i+1} Stats:\n{stats[['Return [%]', '# Trades', 'Win Rate [%]']]}")
        except Exception as e:
            print(f"Backtest on chunk {i+1} failed: {e}")

    if stats_list:
        stats_df = pd.DataFrame(stats_list)

        # Select only numeric columns for averaging
        # AI also filter or convert timedeltas here AI!
        numeric_stats_df = stats_df.select_dtypes(include=np.number)
        averaged_stats = numeric_stats_df.mean()

        print("\n\n--- Averaged Backtest Stats ---")
        print(averaged_stats)
    else:
        print("\nNo backtests were successfully completed.")

    # print("\nStarting optimizations defined in main()...")
    # main()
    # window = 5000
    # import matplotlib.pyplot as plt

    # The following loop demonstrates one-step-ahead prediction with Prophet.
    # WARNING: It is very slow because it retrains the model at each step.
    # predictions = []

    # for i in range(24, len(data) - 1):
    #     prophet_data = pd.DataFrame({"ds": data.index[:i], "y": data.Close[:i]})
    #     model = Prophet()
    #     model.fit(prophet_data)
    #     future = model.make_future_dataframe(periods=1, freq="h", include_history=False)
    #     forecast = model.predict(future)

    #     prediction = forecast["yhat"].iloc[0]
    #     predictions.append(prediction)

    # # Plotting the results
    # plot_range = range(24, 24 + len(predictions))
    # actual_values = data.Close.iloc[plot_range]

    # plt.figure(figsize=(15, 7))
    # plt.plot(actual_values.index, actual_values, label="Actual Price (Trend)")
    # plt.plot(
    #     actual_values.index,
    #     predictions,
    #     label="Predicted Price (1-step ahead)",
    #     linestyle="--",
    # )
    # plt.title("Prophet One-Step-Ahead Forecast vs Actual Price")
    # plt.xlabel("Date")
    # plt.ylabel("Price")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

