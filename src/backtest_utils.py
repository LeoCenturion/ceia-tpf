import ccxt
import pandas as pd
import optuna
import mlflow
import numpy as np
import os
import re
from backtesting import Backtest


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
    df.sort_index(inplace=True)
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


def sanitize_metric_name(name):
    """Sanitize metric name to be MLflow compliant."""
    return re.sub(r'[^a-zA-Z0-9_\-.\s:/]', '', name)


def optimize_strategy_random_chunks(
    data, strategy, study_name, n_trials=100, n_chunks=20, chunk_size=300, n_jobs=8
):
    """
    Optimize strategy hyperparameters using Optuna by backtesting on random data chunks.
    For each trial, it averages the stats over several random chunks and logs artifacts for each chunk.
    """

    def objective(trial):
        params = strategy.get_optuna_params(trial)
        stats_list = []
        np.random.seed(42)  # For reproducibility of chunks across trials

        run_name = f"{study_name}-" + "-".join([f"{k}={v}" for k, v in params.items()])
        with mlflow.start_run(run_name=run_name, nested=True) as run:
            mlflow.log_params(params)

            for i in range(n_chunks):
                max_start_idx = len(data) - chunk_size
                if max_start_idx <= 0:
                    print("Data is smaller than chunk size, cannot create random chunks.")
                    break

                start_idx = np.random.randint(0, max_start_idx)
                end_idx = start_idx + chunk_size
                chunk_data = data.iloc[start_idx:end_idx]

                bt = Backtest(chunk_data, strategy, cash=10000, commission=0.01)
                stats = bt.run(**params)
                if stats is not None:
                    stats_list.append(stats)
                    # Log artifacts for each chunk, organizing them in subdirectories
                    plot_filename = f"backtest_plot_chunk_{i}.html"
                    trades_filename = f"trades_chunk_{i}.csv"

                    bt.plot(filename=plot_filename, open_browser=False)
                    if os.path.exists(plot_filename):
                        mlflow.log_artifact(plot_filename, artifact_path=f"chunk_{i}")

                    trades_df = stats['_trades']
                    if not trades_df.empty:
                        trades_df.to_csv(trades_filename, index=False)
                        mlflow.log_artifact(trades_filename, artifact_path=f"chunk_{i}")

                        # Clean up local files
                        if os.path.exists(plot_filename):
                            os.remove(plot_filename)
                        if os.path.exists(trades_filename):
                            os.remove(trades_filename)

            if not stats_list:
                return 0

            # Average the stats
            stats_df = pd.DataFrame(stats_list)
            for col in stats_df.columns:
                if pd.api.types.is_timedelta64_dtype(stats_df[col]):
                    stats_df[col] = stats_df[col].dt.total_seconds()

            numeric_stats_df = stats_df.select_dtypes(include=np.number)
            averaged_stats = numeric_stats_df.mean().to_dict()

            # Log averaged stats to MLflow
            for key, value in averaged_stats.items():
                sanitized_key = sanitize_metric_name(key)
                mlflow.log_metric(sanitized_key, value)

            return averaged_stats.get("Sharpe Ratio", 0)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage="sqlite:///optuna-study.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs)
    return study.best_params


def optimize_strategy(data, strategy, study_name, n_trials=100, n_jobs=8):
    """
    Optimize strategy hyperparameters using Optuna.
    For each trial, run an expanding window backtest and log the averaged stats to MLflow.
    """
    def objective(trial):
        params = strategy.get_optuna_params(trial)

        bt = Backtest(data, strategy, cash=10000, commission=0.001)
        stats = bt.run(**params)

        run_name = f"{study_name}-" + "-".join([f"{k}={v}" for k, v in params.items()])
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_params(params)
            for key, value in stats.items():
                sanitized_key = sanitize_metric_name(key)
                mlflow.log_metric(sanitized_key, value)

            # Log artifacts for the last step if available
            if bt:
                plot_filename = "backtest_plot.html"
                trades_filename = "trades.csv"

                # Save plot
                # open_browser=False prevents the plot from opening automatically
                bt.plot(filename=plot_filename, open_browser=False)
                if os.path.exists(plot_filename):
                    mlflow.log_artifact(plot_filename)
                # Save trades
                trades_df = stats['_trades']
                if not trades_df.empty:
                    trades_df.to_csv(trades_filename, index=False)
                    mlflow.log_artifact(trades_filename)

                # Clean up created files
                if os.path.exists(plot_filename):
                    os.remove(plot_filename)
                if os.path.exists(trades_filename):
                    os.remove(trades_filename)

        return stats.get('Sharpe Ratio', 0)

    study = optuna.create_study(study_name=study_name, direction='maximize', storage="sqlite:///optuna-study.db", load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs)
    return study.best_params


def run_optimizations(strategies, data_path, start_date, tracking_uri, experiment_name, n_trials_per_strategy=10, n_jobs=1):
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
            optimize_strategy(data, strategy, n_trials=n_trials_per_strategy, study_name=name, n_jobs=n_jobs)
            print(f"Optimization for {name} complete.")

def run_optimizations_random_chunks(strategies, data_path, start_date, tracking_uri, experiment_name, n_trials_per_strategy=10, n_chunks=15, chunk_size = 200, n_jobs=1):
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
    data.sort_index(inplace=True)
    for name, strategy in strategies.items():
        # This outer run is for grouping the optimization trials
        with mlflow.start_run(run_name=f"Optimize_{name}"):
            print(f"Optimizing {name}...")
            optimize_strategy_random_chunks(data, strategy, n_trials=n_trials_per_strategy, study_name=name,n_chunks=n_chunks,chunk_size=chunk_size, n_jobs=n_jobs)
            print(f"Optimization for {name} complete.")
