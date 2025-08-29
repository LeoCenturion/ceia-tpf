import ccxt
import pandas as pd
import optuna
import mlflow
import numpy as np
import os # Added for file operations
import re
from backtesting import Backtest, Strategy
from backtesting.lib import crossover


def fetch_historical_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_date: str = None,
    end_date: str = None,
    data_path: str = None,
) -> pd.DataFrame:
    """
    Fetch historical price data for backtesting.

    The `backtesting` library requires column names: 'Open', 'High', 'Low', 'Close'.

    :param symbol: Trading pair
    :param timeframe: Candle timeframe
    :param start_date: Start date for data fetch
    :param end_date: End date for data fetch
    :param data_path: Path to local CSV file. If provided, data is loaded from here.
    :return: DataFrame with OHLCV data
    """
    if data_path:
        df = pd.read_csv(data_path)
        df.rename(columns={"date": "timestamp", "Volume BTC": "volume"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        if start_date:
            start_date = pd.to_datetime(start_date).tz_localize(None)
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

    else:
        exchange = ccxt.binance(
            {"enableRateLimit": True, "options": {"defaultType": "future"}}
        )
        start_timestamp = None
        if start_date:
            start_timestamp = exchange.parse8601(start_date)
        else:
            start_timestamp = None
        end_timestamp = None
        if end_date:
            end_timestamp = exchange.parse8601(end_date)
        else:
            end_timestamp = None

        ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe,
            since=start_timestamp,
            limit=end_timestamp,
        )

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

    # The backtesting library requires uppercase column names
    df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )
    return df


# Indicator functions to be used with `self.I`
def pct_change(series):
    return pd.Series(series).pct_change()


def sma(series, n):
    return pd.Series(series).rolling(n).mean()


def ewm(series, span):
    return pd.Series(series).ewm(span=span, adjust=False).mean()

def std(series, n):
    return pd.Series(series).rolling(n).std()


def rsi_indicator(series, n=14):
    delta = pd.Series(series).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = sma(gain, n)
    avg_loss = sma(loss, n)

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def momentum_indicator(series, window=10):
    return pd.Series(series).diff(window)


def adjust_data_to_ubtc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusts the price data from BTC to micro-BTC (uBTC).
    1 BTC = 1,000,000 uBTC.
    This function divides the OHLC prices by 1,000,000.
    """
    df_copy = df.copy()
    for col in ["Open", "High", "Low", "Close"]:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col] / 1_000_000
    return df_copy


class MaCrossover(Strategy):
    short_window = 50
    long_window = 200
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        price_change = self.I(pct_change, self.data.Close)
        self.ma_short = self.I(sma, price_change, self.short_window)
        self.ma_long = self.I(sma, price_change, self.long_window)

    def next(self):
        price = self.data.Close[-1]
        if crossover(self.ma_short, self.ma_long):
            self.buy(sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit))
        elif crossover(self.ma_long, self.ma_short):
            self.sell(
                sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit)
            )

    @classmethod
    def get_optuna_params(cls, trial):
        """Suggest hyperparameters for Optuna optimization."""
        return {
            'short_window': trial.suggest_int('short_window', 10, 100),
            'long_window': trial.suggest_int('long_window', 50, 250),
        }


class BollingerBands(Strategy):
    bb_window = 20
    bb_std = 2
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        self.ma = self.I(sma, self.data.Close, self.bb_window)
        self.std = self.I(std, self.data.Close, self.bb_window)
        self.upper_band = self.ma + self.bb_std * self.std
        self.lower_band = self.ma - self.bb_std * self.std

    def next(self):
        price = self.data.Close[-1]
        if price < self.lower_band:
            self.buy(sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit))
        elif price > self.upper_band:
            self.sell(
                sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit)
            )

    @classmethod
    def get_optuna_params(cls, trial):
        """Suggest hyperparameters for Optuna optimization."""
        return {
            'bb_window': trial.suggest_int('bb_window', 10, 100),
            'bb_std': trial.suggest_float('bb_std', 1.5, 3.5),
        }


class MACD(Strategy):
    fast_span = 12
    slow_span = 26
    signal_span = 9
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        self.macd = self.I(ewm, self.data.Close, self.fast_span) - self.I(ewm, self.data.Close, self.slow_span)
        self.signal = self.I(ewm, self.macd, self.signal_span)

    def next(self):
        price = self.data.Close[-1]
        if crossover(self.macd, self.signal):
            self.buy(sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit))
        elif crossover(self.signal, self.macd):
            self.sell(
                sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit)
            )

    @classmethod
    def get_optuna_params(cls, trial):
        """Suggest hyperparameters for Optuna optimization."""
        return {
            'fast_span': trial.suggest_int('fast_span', 5, 50),
            'slow_span': trial.suggest_int('slow_span', 20, 100),
            'signal_span': trial.suggest_int('signal_span', 5, 50),
        }

def sanitize_metric_name(name):
    """Sanitize metric name to be MLflow compliant."""
    return re.sub(r'[^a-zA-Z0-9_\-.\s:/]', '', name)

def run_expanding_window_backtest(data, strategy, params, window_size, step_size, name):
    """
    Run backtest with expanding window.
    """
    from collections import defaultdict
    import os
    import pandas as pd

    stats_list = []

    for i in range(window_size, len(data), step_size):
        window_data = data.iloc[i-window_size:i]
        bt = Backtest(window_data, strategy, cash=10000, commission=.002)
        stats = bt.run(**params)
        stats_list.append(stats)

    if not stats_list:
        return []

    # Average the stats
    stats_df = pd.DataFrame(stats_list)
    numeric_stats_df = stats_df.select_dtypes(include='number')
    averaged_stats = numeric_stats_df.mean().to_dict()

    run_name = f"{name}-" + "-".join([f"{k}={v}" for k, v in params.items()])
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(params)

        # Sanitize and log metrics
        for key, value in averaged_stats.items():
            sanitized_key = sanitize_metric_name(key)
            mlflow.log_metric(sanitized_key, value)

    return averaged_stats.get('Return [%]', [])

def optimize_strategy(data, strategy, study_name, n_trials=100):
    """
    Optimize strategy hyperparameters using Optuna.
    For each trial, run an expanding window backtest and log the averaged stats to MLflow.
    """
    def objective(trial):
        params = strategy.get_optuna_params(trial)
        
        # Run expanding window backtest for the given params
        stats_list = []
        last_bt_instance = None # To store the Backtest object for the last successful step
        step_size = 1000 # Increment for expanding window
        min_window_size = step_size # Assuming minimum window size is the step_size

        for i in range(min_window_size, len(data) + 1, step_size):
            window_data = data.iloc[:i]
            
            # Skip if window_data is empty or too small for strategy to initialize (e.g., for indicators)
            # A more robust check might involve actual strategy requirements.
            if len(window_data) < min_window_size: 
                continue 
            
            bt = Backtest(window_data, strategy, cash=10000, commission=.002)
            stats = bt.run(**params)
            
            if stats is not None: # Only append if backtest ran successfully
                stats_list.append(stats)
                last_bt_instance = bt # Keep track of the last successful Backtest instance

        if not stats_list:
            return 0 # Return a neutral value if no backtests were run


        stats_df = pd.DataFrame(stats_list)
        columns_to_drop = ['Duration', 'Max. Drawdown Duration', 'Avg. Drawdown Duration', 'Max. Trade Duration', 'Avg. Trade Duration']
        stats_df = stats_df.drop(columns=columns_to_drop, errors='ignore') # Use errors='ignore' for robustness


        averaged_stats = stats_df.select_dtypes(include=np.number).mean().to_dict()

        # Log to MLflow
        run_name = f"{study_name}-" + "-".join([f"{k}={v}" for k, v in params.items()])
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_params(params)
            for key, value in averaged_stats.items():
                sanitized_key = sanitize_metric_name(key)
                mlflow.log_metric(sanitized_key, value)

            # Log artifacts for the last step if available
            if last_bt_instance:
                plot_filename = "backtest_plot.html"
                trades_filename = "trades.csv"

                # Save plot
                # open_browser=False prevents the plot from opening automatically
                last_bt_instance.plot(filename=plot_filename, open_browser=False)
                mlflow.log_artifact(plot_filename)
                # Save trades
                if not last_bt_instance._strategy.trades:
                    last_bt_instance._strategy.trades.to_csv(trades_filename, index=False)
                    mlflow.log_artifact(trades_filename)

                # Clean up created files
                if os.path.exists(plot_filename):
                    os.remove(plot_filename)
                if os.path.exists(trades_filename):
                    os.remove(trades_filename)

        return averaged_stats.get('Sortino Ratio', 0)

    study = optuna.create_study(study_name=study_name, direction='maximize', storage="sqlite:///optuna-study.db", load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def run_optimizations(strategies, data_path, start_date, tracking_uri, experiment_name, n_trials_per_strategy=10):
    """
    Run optimization for a set of strategies.

    :param strategies: Dictionary of strategy names to strategy classes.
    :param data_path: Path to the historical data CSV file.
    :param start_date: Start date for the data.
    :param tracking_uri: MLflow tracking URI.
    :param experiment_name: MLflow experiment name.
    :param n_trials_per_strategy: Number of Optuna trials for each strategy.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    data = fetch_historical_data(
        data_path=data_path,
        start_date=start_date
    )
    data = adjust_data_to_ubtc(data)

    for name, strategy in strategies.items():
        # This outer run is for grouping the optimization trials
        with mlflow.start_run(run_name=f"Optimize_{name}"):
            print(f"Optimizing {name}...")
            optimize_strategy(data, strategy, n_trials=n_trials_per_strategy, study_name=name)
            print(f"Optimization for {name} complete.")

def main():
    """Main function to run the optimization with default parameters."""
    strategies = {
        "MaCrossover": MaCrossover,
        "BollingerBands": BollingerBands,
        "MACD": MACD
    }
    run_optimizations(
        strategies=strategies,
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2022-01-01T00:00:00Z",
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="Trading Strategies",
        n_trials_per_strategy=10
    )


if __name__ == "__main__":
    main()
