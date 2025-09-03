import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
from prophet import Prophet
import statsmodels.api as sm
from pykalman import KalmanFilter
import logging
import itertools
logging.getLogger("prophet").setLevel(logging.WARNING)

try:
    from arch import arch_model
except ImportError:
    arch_model = None
from backtest_utils import (
    run_optimizations,
    run_optimizations_random_chunks,
    fetch_historical_data,
    adjust_data_to_ubtc,
)

# Custom indicator for preprocessing
def price_difference(series: np.ndarray) -> np.ndarray:
    """Calculates (price[i] - price[i-1])."""
    series = pd.Series(series)
    return series.diff().fillna(0).values


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
    refit_period = 1 #24 * 30  # Refit the model every N bars
    stop_loss = 0.05
    take_profit = 0.10
    def init(self):
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
            "refit_period": trial.suggest_categorical("refit_period", [1])
        }


class ARIMAStrategy(Strategy):
    p = 12
    d = 1
    q = 12
    refit_period = 1
    stop_loss = 0.05
    take_profit = 0.10
    lookback_length = 24 * 30 * 1
    def init(self):
        self.model_fit = None
        self.processed_data = self.I(
            price_difference, self.data.Close
        )

    def next(self):
        price = self.data.Close[-1]

        # Refit model periodically and if we have enough data
        if len(self.data) % self.refit_period == 0:
            try:
                model = sm.tsa.ARIMA(
                    self.processed_data[-self.lookback_length:], order=(self.p, self.d, self.q)
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
            "refit_period": trial.suggest_categorical("refit_period", [1]),
        }


class SARIMAStrategy(Strategy):
    p, d, q = 5, 1, 0
    P, D, Q, s = (
        1,
        1,
        0,
        24,
    )  # Seasonal order, s=24 for daily seasonality on hourly data
    refit_period = 1
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        self.model_fit = None
        self.processed_data = self.I(
            price_difference, self.data.Close
        )

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
            "refit_period": trial.suggest_categorical("refit_period", [1]),
        }


class KalmanARIMAStrategy(Strategy):
    p = 12
    d = 1
    q = 12
    refit_period = 1
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        self.model_fit = None
        processed_data = self.I(
            price_difference, self.data.Close
        )
        self.kalman_filtered_data = self.I(kalman_filter_indicator, processed_data)

    def next(self):
        price = self.data.Close[-1]

        # Refit model periodically and if we have enough data
        if len(self.data) % self.refit_period == 0 :
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
            "q": trial.suggest_int("q", 0, 5)
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

def backtest_random_chunks(
        strategy,
        chunk_size = 100,
        num_chunks_to_test = 30
):
    # Run a single backtest for SARIMAStrategy
    print("Running single backtest for ARIMAStrategy...")
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2022-01-01T00:00:00Z",
    )
    data = adjust_data_to_ubtc(data)
    data.sort_index(inplace=True)
    print(f"data len {len(data)}")

      # Limiting to 5 chunks to keep the test reasonably short
    stats_list = []
    np.random.seed(42)  # For reproducible
    print(f"Splitting data into {num_chunks_to_test} random chunks of {chunk_size} hours and averaging backtest stats...")
    for i in range(num_chunks_to_test):
        max_start_idx = len(data) - chunk_size
        if max_start_idx <= 0:
            print("Data is smaller than chunk size, cannot create random chunks.")
            break

        start_idx = np.random.randint(0, max_start_idx)
        end_idx = start_idx + chunk_size

        chunk_data = data.iloc[start_idx:end_idx]
        print(f"\n--- Backtesting on Chunk {i+1}/{num_chunks_to_test} (Index: {start_idx}-{end_idx}) ---")

        bt = Backtest(chunk_data, strategy, cash=10_000, commission=0.002)
        try:
            stats = bt.run()
            stats_list.append(stats)
            print(f"Chunk {i+1} Stats:\n{stats[['Return [%]', '# Trades', 'Win Rate [%]']]}")
        except Exception as e:
            print(f"Backtest on chunk {i+1} failed: {e}")

    if stats_list:
        stats_df = pd.DataFrame(stats_list)

        # Convert timedelta columns to total seconds before averaging
        for col in stats_df.columns:
            if pd.api.types.is_timedelta64_dtype(stats_df[col]):
                stats_df[col] = stats_df[col].dt.total_seconds()

        # Select only numeric columns for averaging
        numeric_stats_df = stats_df.select_dtypes(include=np.number)
        averaged_stats = numeric_stats_df.mean()

        print("\n\n--- Averaged Backtest Stats ---")
        print(averaged_stats)
    else:
        print("\nNo backtests were successfully completed.")

class ARIMAXGARCHStrategy(Strategy):
    # ARIMAX params
    p, d, q = 5, 1, 0
    # GARCH params
    g_p, g_q = 1, 1

    refit_period = 1
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        self.arimax_fit = None
        self.garch_fit = None
        self.processed_data = self.I(price_difference, self.data.Close)

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
            except Exception:
                self.arimax_fit = None
                self.garch_fit = None

        if self.arimax_fit and self.garch_fit:
            try:
                # Forecast next volatility from GARCH
                garch_forecast = self.garch_fit.forecast(horizon=1)
                next_vol = np.sqrt(garch_forecast.variance.iloc[-1, 0])

                # Forecast from ARIMAX using the GARCH forecast as exog
                forecast = self.arimax_fit.forecast(steps=1, exog=[next_vol])[0]

                if forecast > 0 and not self.position.is_long:
                    self.buy(
                        sl=price * (1 - self.stop_loss),
                        tp=price * (1 + self.take_profit),
                    )
                elif forecast < 0 and not self.position.is_short:
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
            "g_p": trial.suggest_int("g_p", 1, 3),
            "g_q": trial.suggest_int("g_q", 1, 3),
            "refit_period": trial.suggest_categorical("refit_period", [1]),
        }


def find_best_arima_params(data: pd.Series, p_range=range(0, 5), d_range=range(0, 3), q_range=range(0, 5)):
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
        except Exception:
            # Errors can occur for certain parameter combinations
            continue

    print(f"Best ARIMA{best_order} model found with AIC: {best_aic}")
    return best_order, best_model


if __name__ == "__main__":
    data = fetch_historical_data(
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2022-01-01T00:00:00Z"
    )
    data = adjust_data_to_ubtc(data)
    data.sort_index(inplace=True)
    best_order, best_model = find_best_arima_params(data.Close)
    print(f"Best arima order {best_order}")
    # strategies = {
    #     # "ARIMAStrategy": ARIMAStrategy,
    #     # "KalmanARIMAStrategy": KalmanARIMAStrategy,
    #     "ProphetStrategy": ProphetStrategy
    # }
    # run_optimizations_random_chunks(
    #     strategies=strategies,
    #     data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
    #     start_date="2022-01-01T00:00:00Z",
    #     tracking_uri="sqlite:///mlflow.db",
    #     experiment_name="Trading Strategies",
    #     n_trials_per_strategy=1,
    #     n_chunks=2,
    #     chunk_size=200
    # )
