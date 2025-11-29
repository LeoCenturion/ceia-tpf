import pandas as pd
import numpy as np
from backtesting import Backtest
from prophet import Prophet
import statsmodels.api as sm
from pykalman import KalmanFilter
import logging
import itertools

from src.backtesting.backtesting import TrialStrategy, run_optimizations

logging.getLogger("prophet").setLevel(logging.WARNING)

try:
    from arch import arch_model
except ImportError:
    arch_model = None


# Custom indicator for preprocessing
def price_difference(series: np.ndarray) -> np.ndarray:
    """Calculates (price[i] - price[i-1])/price[i-1]."""
    series: pd.Series = pd.Series(series)
    return series.pct_change().fillna(0).values


def kalman_filter_indicator(series: np.ndarray) -> np.ndarray:
    """Applies a Kalman filter to a time series."""
    # Using a simple KF configuration for smoothing
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    (filtered_state_means, _) = kf.filter(series)
    return filtered_state_means.flatten()


class ProphetStrategy(TrialStrategy):  # pylint: disable=attribute-defined-outside-init
    """
    Intractable. Prophet doesn't take data when predicting, so if we train every 30 days then all 30 days will have the same prediction.
    If we train every hour then it takes more than a day do backtest with hourly data from 2022 to 2025.
    """

    refit_period = 1  # 24 * 30  # Refit the model every N bars
    stop_loss = 0.05
    take_profit = 0.10
    threshold = 0.001

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.forecast = None

    def next(self):
        if len(self.data.Close) > 1 and len(self.data.Close) % self.refit_period == 0:
            # print(f"Refitting, idx {len(self.data)} len {len(self.data.index[-self.lookback_length:])}")
            prophet_data = pd.DataFrame({"ds": self.data.index, "y": self.data.Close})
            self.model = Prophet()
            self.model.fit(prophet_data)
            # print(f"Model refitted")

        # Generate forecast if model is fitted
        if self.model:
            future = self.model.make_future_dataframe(
                periods=1, freq="h", include_history=False
            )  # Assuming hourly data
            self.forecast = self.model.predict(future)

        price = self.data.Close[-1]
        if self.forecast is not None:
            forecast_price = self.forecast["yhat"].iloc[-1]
            if (
                forecast_price > price * (1 + self.threshold)
                and not self.position.is_long
            ):
                self.buy(
                    sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit)
                )
            elif (
                forecast_price < price * (1 - self.threshold)
                and not self.position.is_short
            ):
                self.sell(
                    sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit)
                )

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            "refit_period": trial.suggest_categorical("refit_period", [1]),
            "threshold": trial.suggest_categorical(
                "threshold", [0.1, 0.01, 0.001, 0.0001]
            ),
        }


class ARIMAStrategy(TrialStrategy):  # pylint: disable=attribute-defined-outside-init
    p = 12
    d = 1
    q = 12
    refit_period = 24 * 7  # Refit weekly
    stop_loss = 0.05
    take_profit = 0.10
    threshold = 1e-5
    lookback_length = 24 * 30 * 1
    threshold = 1e-5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_fit = None
        self.processed_data = self.I(price_difference, self.data.Close)

    def next(self):
        price = self.data.Close[-1]

        # Refit model periodically and if we have enough data
        if len(self.data) % self.refit_period == 0:
            try:
                model = sm.tsa.ARIMA(
                    self.processed_data[-self.lookback_length :],
                    order=(self.p, self.d, self.q),
                )
                self.model_fit = model.fit()
            except Exception:  # pylint: disable=broad-exception-caught
                self.model_fit = None
        elif self.model_fit:
            try:
                self.model_fit = self.model_fit.append([self.processed_data[-1]])
            except Exception:  # pylint: disable=broad-exception-caught
                self.model_fit = None

        if self.model_fit:
            try:
                forecast_processed = self.model_fit.forecast(steps=1)[0]
                if forecast_processed > self.threshold and not self.position.is_long:
                    self.buy(
                        sl=price * (1 - self.stop_loss),
                        tp=price * (1 + self.take_profit),
                    )
                elif (
                    forecast_processed < -self.threshold and not self.position.is_short
                ):
                    self.sell(
                        sl=price * (1 + self.stop_loss),
                        tp=price * (1 - self.take_profit),
                    )
            except Exception:  # pylint: disable=broad-exception-caught
                pass  # Ignore forecast errors

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            "p": trial.suggest_int("p", 1, 10),
            "d": trial.suggest_int("d", 0, 2),
            "q": trial.suggest_int("q", 0, 5),
            "refit_period": trial.suggest_categorical(
                "refit_period", [24 * 7, 24 * 30]
            ),
            "threshold": trial.suggest_categorical(
                "threshold", [0.1, 0.01, 0.001, 0.0001]
            ),
        }

    def save_artifacts(self, _trial, _stats, _bt):
        return


class SARIMAStrategy(TrialStrategy):  # pylint: disable=attribute-defined-outside-init
    p, d, q = 5, 1, 0
    P, D, Q, s = (
        1,
        1,
        0,
        24,
    )  # Seasonal order, s=24 for daily seasonality on hourly data
    refit_period = 24 * 7  # Refit weekly
    stop_loss = 0.05
    take_profit = 0.10
    threshold = 1e-5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_fit = None
        self.processed_data = self.I(price_difference, self.data.Close)

    def next(self):
        price = self.data.Close[-1]

        if len(self.data) % self.refit_period == 0:
            try:
                model = sm.tsa.SARIMAX(
                    self.processed_data,
                    order=(self.p, self.d, self.q),
                    seasonal_order=(self.P, self.D, self.Q, self.s),
                )
                self.model_fit = model.fit(disp=False)
            except Exception:  # pylint: disable=broad-exception-caught
                self.model_fit = None
        elif self.model_fit:
            try:
                self.model_fit = self.model_fit.append([self.processed_data[-1]])
            except Exception:  # pylint: disable=broad-exception-caught
                self.model_fit = None

        if self.model_fit:
            try:
                forecast_processed = self.model_fit.forecast(steps=1)[0]
                if forecast_processed > self.threshold and not self.position.is_long:
                    self.buy(
                        sl=price * (1 - self.stop_loss),
                        tp=price * (1 + self.take_profit),
                    )
                elif (
                    forecast_processed < -self.threshold and not self.position.is_short
                ):
                    self.sell(
                        sl=price * (1 + self.stop_loss),
                        tp=price * (1 - self.take_profit),
                    )
            except Exception:  # pylint: disable=broad-exception-caught
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
            "refit_period": trial.suggest_categorical(
                "refit_period", [24 * 7, 24 * 30]
            ),
            "threshold": trial.suggest_categorical(
                "threshold", [0.1, 0.01, 0.001, 0.0001]
            ),
        }


class KalmanARIMAStrategy(
    TrialStrategy
):  # pylint: disable=attribute-defined-outside-init
    p = 12
    d = 0
    q = 12
    refit_period = 24 * 7  # Refit weekly
    stop_loss = 0.05
    take_profit = 0.10
    threshold = 0.001  # Default threshold: 0.1%

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_fit = None
        self.processed_data = self.I(price_difference, self.data.Close)

        # Kalman Filter initialization for iterative updates
        self.kalman_state_mean = 0
        self.kalman_state_covariance = 1
        self.kalman_mean_history = []
        self.kf = KalmanFilter(
            initial_state_mean=self.kalman_state_mean,
            initial_state_covariance=self.kalman_state_covariance,
            n_dim_obs=1,
        )

        self.kalman_filtered_data = self.I(kalman_filter_indicator, self.processed_data)

    def kalman_update(self, series):
        if len(series) <= 1:  # Initialize the Kalman filter
            self.kalman_state_mean, self.kalman_state_covariance = self.kf.filter(
                series
            )
            print(self.kalman_state_mean)
            self.kalman_mean_history.append(self.kalman_state_mean[:, 0].flatten())
        else:
            self.kalman_state_mean, self.kalman_state_covariance = (
                self.kf.filter_update(
                    self.kalman_state_mean, self.kalman_state_covariance, series
                )
            )
            print(self.kalman_state_mean)
            self.kalman_mean_history.append(self.kalman_state_mean[:, 0].flatten())
        return self.kalman_mean_history

    def next(self):
        price = self.data.Close[-1]

        # Refit model periodically and if we have enough data
        if len(self.data) % self.refit_period == 0:
            # print(f"retraining KalmanARIMA. IDX ${len(self.data)}")
            try:
                model = sm.tsa.ARIMA(
                    self.kalman_filtered_data, order=(self.p, self.d, self.q)
                )
                self.model_fit = model.fit()
            except Exception:  # pylint: disable=broad-exception-caught
                self.model_fit = None
        elif self.model_fit:
            try:
                self.model_fit = self.model_fit.append([self.kalman_filtered_data[-1]])
            except Exception:  # pylint: disable=broad-exception-caught
                self.model_fit = None

        if self.model_fit:
            try:
                forecast_processed = self.model_fit.forecast(steps=1)[0]

                # New logic: buy if forecast is > threshold % of current price
                if forecast_processed > self.threshold and not self.position.is_long:
                    # print(f'forecast: {forecast_processed}, trhesh: {self.threshold}, buying')
                    self.buy(
                        sl=price * (1 - self.stop_loss),
                        tp=price * (1 + self.take_profit),
                    )
                # Sell if forecast is < threshold % of current price
                elif forecast_processed < self.threshold and not self.position.is_short:
                    # print(f'forecast: {forecast_processed}, trhesh: {self.threshold}, selling')
                    self.sell(
                        sl=price * (1 + self.stop_loss),
                        tp=price * (1 - self.take_profit),
                    )
            except Exception:  # pylint: disable=broad-exception-caught
                pass  # Ignore forecast errors

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            "p": trial.suggest_int("p", 5, 10),
            "d": trial.suggest_int("d", 1, 3),
            "q": trial.suggest_int("q", 3, 7),
            "threshold": trial.suggest_float("threshold", 1e-4, 1e-2, log=True),
            "refit_period": trial.suggest_categorical(
                "refit_period", [24 * 7, 24 * 30]
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


class ARIMAXGARCHStrategy(
    TrialStrategy
):  # pylint: disable=attribute-defined-outside-init
    # ARIMAX params
    p, d, q = 5, 1, 0
    # GARCH params
    g_p, g_q = 1, 1

    refit_period = 24 * 7
    stop_loss = 0.05
    take_profit = 0.10
    threshold = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arimax_fit = None
        self.garch_fit = None
        self.processed_data = self.I(
            lambda series: price_difference(series) * 10000.0, self.data.Close
        )

    def next(self):
        price = self.data.Close[-1]

        # Refit models periodically
        if len(self.data) % self.refit_period == 0:
            try:
                # 1. Fit GARCH model on processed data (returns)
                garch_model = arch_model(self.processed_data, p=self.g_p, q=self.g_q)
                self.garch_fit = garch_model.fit(disp="off")

                # Get historical conditional volatility as exogenous variable
                cond_vol = self.garch_fit.conditional_volatility

                # 2. Fit ARIMAX with GARCH volatility as exog
                arimax_model = sm.tsa.ARIMA(
                    self.processed_data,
                    exog=cond_vol,
                    order=(self.p, self.d, self.q),
                )
                self.arimax_fit = arimax_model.fit()
            except Exception:  # pylint: disable=broad-exception-caught
                self.arimax_fit = None
                self.garch_fit = None
        elif self.arimax_fit and self.garch_fit:
            try:
                # This is an approximation: GARCH is not refit, so volatility estimates can become stale.
                # Forecast volatility for the current step to use as exog for appending.
                garch_forecast = self.garch_fit.forecast(horizon=1)
                next_vol = np.sqrt(garch_forecast.variance.iloc[-1, 0])

                self.arimax_fit = self.arimax_fit.append(
                    [self.processed_data[-1]], exog=np.array([[next_vol]])
                )
            except Exception:  # pylint: disable=broad-exception-caught
                self.arimax_fit = None

        if self.arimax_fit and self.garch_fit:
            try:
                # Forecast next volatility from GARCH
                garch_forecast = self.garch_fit.forecast(horizon=1)
                next_vol = np.sqrt(garch_forecast.variance.iloc[-1, 0])

                # Forecast from ARIMAX using the GARCH forecast as exog
                forecast = self.arimax_fit.forecast(steps=1, exog=[next_vol])[0]

                if forecast > self.threshold and not self.position.is_long:
                    self.buy(
                        sl=price * (1 - self.stop_loss),
                        tp=price * (1 + self.take_profit),
                    )
                elif forecast < -self.threshold and not self.position.is_short:
                    self.sell(
                        sl=price * (1 + self.stop_loss),
                        tp=price * (1 - self.take_profit),
                    )
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            "p": trial.suggest_int("p", 1, 5),
            "d": trial.suggest_int("d", 0, 1),
            "q": trial.suggest_int("q", 0, 5),
            "g_p": trial.suggest_int("g_p", 1, 3),
            "g_q": trial.suggest_int("g_q", 1, 3),
            "refit_period": trial.suggest_categorical(
                "refit_period", [24 * 7, 24 * 30]
            ),
            "threshold": trial.suggest_categorical("threshold", [0.01, 0.001, 0.0001]),
        }


def find_best_arima_params(
    data: pd.Series, p_range=range(0, 5), d_range=range(0, 3), q_range=range(0, 5)
):
    """
    Performs a grid search to find the best (p, d, q) parameters for an ARIMA model
    based on the Akaike Information Criterion (AIC).
    """
    best_aic = float("inf")
    best_order = None
    best_model = None

    # Generate all different combinations of p, d, and q triplets
    pdq = list(itertools.product(p_range, d_range, q_range))

    for order in pdq:
        try:
            model = sm.tsa.ARIMA(data, order=order)
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = order
                best_model = results
        except Exception:  # pylint: disable=broad-exception-caught
            # Errors can occur for certain parameter combinations
            continue

    print(f"Best ARIMA{best_order} model found with AIC: {best_aic}")
    return best_order, best_model


def kalman_arima_forecast(series: pd.Series, p: int, d: int, q: int) -> float:
    """
    Applies a Kalman filter to the price difference of a series,
    fits an ARIMA model, and returns a one-step forecast.
    """
    # 1. Preprocess the data
    processed_data = price_difference(series.values)
    kalman_filtered_data = kalman_filter_indicator(processed_data)

    # 2. Fit ARIMA model
    try:
        model = sm.tsa.ARIMA(kalman_filtered_data, order=(p, d, q))
        model_fit = model.fit()
        # 3. Generate forecast
        forecast = model_fit.forecast(steps=1)[0]
        return forecast
    except Exception:  # pylint: disable=broad-exception-caught
        return 0.0


def calculate_chunks_for_coverage(
    total_data_size: int, chunk_size: int, coverage_percentage: float = 20.0
) -> int:
    """
    Calculates the number of chunks required to cover a certain percentage of the dataset.

    This calculation provides an estimate of how many random chunks one might need to sample
    to have seen a certain percentage of the total data points, assuming minimal overlap.
    The actual coverage with random sampling might be less due to overlapping chunks.

    :param total_data_size: The total number of data points in the dataset.
    :param chunk_size: The size of each chunk.
    :param coverage_percentage: The desired percentage of the dataset to cover (e.g., 20 for 20%).
    :return: The estimated number of chunks required.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not (0 <= coverage_percentage <= 100):
        raise ValueError("coverage_percentage must be between 0 and 100.")

    data_to_cover = total_data_size * (coverage_percentage / 100.0)
    num_chunks = data_to_cover / chunk_size
    return int(np.ceil(num_chunks))


def run_strategy_backtest(data, strategy, threshold):
    """
    Runs a single backtest for the strategy on the full dataset.
    """
    print("Running single backtest for KalmanARIMAStrategy...")
    bt = Backtest(data, strategy, cash=10_000, commission=0.002)
    stats = bt.run(threshold=threshold)
    print(stats)
    plot_filename = "kalman_arima_backtest.html"
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"Backtest plot saved to {plot_filename}")
    return stats


if __name__ == "__main__":
    price_change_bars = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1h/BTCUSDT_price_change_bars_0_32.csv"
    hour_bars = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv"

    strategies_to_run = {
        "ARIMAStrategy": ARIMAStrategy,
        # "KalmanARIMAStrategy": KalmanARIMAStrategy,
        # "ARIMAXGARCHStrategy": ARIMAXGARCHStrategy,
        # "ProphetStrategy": ProphetStrategy
    }

    run_optimizations(
        strategies=strategies_to_run,
        data_path=hour_bars,
        start_date="2020-01-01T00:00:00Z",
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="Statistical Strategies",
        n_trials_per_strategy=50,
        n_jobs=12,
    )

    # run_strategy_backtest(data.iloc[:3000], KalmanARIMAStrategy, threshold = 0.2/100.0)
