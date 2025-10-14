import ccxt
import pandas as pd
import optuna
import mlflow
import numpy as np
import os
import re
from backtesting import Backtest
from sklearn.metrics import f1_score


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
        print(f'read csv {df}')
        df.rename(columns={"date": "timestamp", "Volume BTC": "volume"}, inplace=True)
        # s_timestamp = str(int(df['unix'].iloc[0]))
        # unit = 'us' if len(s_timestamp) > 13 else 'ms'
        # df["timestamp"] = pd.to_datetime(df["unix"], unit=unit)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        print(f'after rename {df}')
        df.set_index("timestamp", inplace=True)

        if start_date:
            start_date = pd.to_datetime(start_date)
            if start_date.tzinfo is not None:
                start_date = start_date.tz_convert(None)
            df = df[df.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            if end_date.tzinfo is not None:
                end_date = end_date.tz_convert(None)
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



def optimize_strategy(data, strategy, study_name, n_trials=100, n_jobs=8):
    """
    Optimize strategy hyperparameters using Optuna.
    For each trial, run an expanding window backtest and log the averaged stats to MLflow.
    """
    parent_run = mlflow.active_run()
    parent_run_id = parent_run.info.run_id if parent_run else None

    def objective(trial):
        params = strategy.get_optuna_params(trial)

        bt = Backtest(data, strategy, cash=10000, commission=0.001)
        stats = bt.run(**params)

        run_name = f"{study_name}-" + "-".join([f"{k}={v}" for k, v in params.items()])
        tags = {"mlflow.parentRunId": parent_run_id} if parent_run_id else {}
        with mlflow.start_run(run_name=run_name, tags=tags, nested=True):
            mlflow.log_params(params)

            for key, value in stats.items():
                print(f' key value {key} {value} {type(value)}')
                if isinstance(value, pd.Timestamp):
                    value = value.timestamp()
                if isinstance(value, pd.Timedelta):
                    value = value.total_seconds()
                if not np.issubdtype(type(value), np.number):
                    print('skipping')
                    continue
                sanitized_key = sanitize_metric_name(key)
                mlflow.log_metric(sanitized_key, value)
            # Log artifacts for the last step if available
            if bt:
                plot_filename = f"backtest_plot_trial_{trial.number}.html"
                trades_filename = f"trades_trial_{trial.number}.csv"

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

        sharpe_ratio = stats.get('Sharpe Ratio', 0)
        if sharpe_ratio is None or np.isnan(sharpe_ratio):
            return 0.0
        return sharpe_ratio

    study = optuna.create_study(study_name=study_name, direction='maximize', storage="sqlite:///optuna-study.db", load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs)
    try:
        return study.best_params
    except ValueError:
        print(f"Study {study_name} finished, but no best trial was found.")
        return None


def optimize_classification_strategy(data, strategy, study_name, n_trials=100, n_jobs=8):
    """
    Optimize a classification-based strategy using Optuna.
    This function is similar to `optimize_strategy` but is tailored for classification
    tasks by optimizing for 'F1 Score' by default. The strategy should compute
    and return 'F1 Score' in its stats.
    """
    parent_run = mlflow.active_run()
    parent_run_id = parent_run.info.run_id if parent_run else None

    def objective(trial):
        params = strategy.get_optuna_params(trial)

        bt = Backtest(data, strategy, cash=10000, commission=0.001)
        stats = bt.run(**params)

        run_name = f"{study_name}-" + "-".join([f"{k}={v}" for k, v in params.items()])
        tags = {"mlflow.parentRunId": parent_run_id} if parent_run else {}
        with mlflow.start_run(run_name=run_name, tags=tags, nested=True):
            mlflow.log_params(params)

            for key, value in stats.items():
                print(f' key value {key} {value} {type(value)}')
                if isinstance(value, pd.Timestamp):
                    value = value.timestamp()
                if isinstance(value, pd.Timedelta):
                    value = value.total_seconds()
                if not np.issubdtype(type(value), np.number):
                    print('skipping')
                    continue
                sanitized_key = sanitize_metric_name(key)
                mlflow.log_metric(sanitized_key, value)
            # Log artifacts for the last step if available
            if bt:
                plot_filename = f"backtest_plot_trial_{trial.number}.html"
                trades_filename = f"trades_trial_{trial.number}.csv"

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

        # For classification, we might optimize for a metric like F1 score
        # The strategy needs to compute and return this.
        strategy_instance = bt._strategy
        if hasattr(strategy_instance, 'y_true') and hasattr(strategy_instance, 'y_pred'):
            if len(strategy_instance.y_true) > 0 and len(strategy_instance.y_pred) > 0:
                f1 = f1_score(strategy_instance.y_true, strategy_instance.y_pred, zero_division=0)
                stats['F1 Score'] = f1
                mlflow.log_metric("F1 Score", f1)

        f1_val = stats.get('F1 Score', 0)
        if f1_val is None or np.isnan(f1_val):
            return 0.0
        return f1_val

    study = optuna.create_study(study_name=study_name, direction='maximize', storage="sqlite:///optuna-study.db", load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs)
    try:
        return study.best_params
    except ValueError:
        print(f"Study {study_name} finished, but no best trial was found.")
        return None


def run_optimizations(strategies, data_path, start_date, tracking_uri, experiment_name, n_trials_per_strategy=10, n_jobs=1,timeframe='1h'):
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
        start_date=start_date,
        timeframe=timeframe
    )
    data = adjust_data_to_ubtc(data)
    # Get the actual start and end dates from the data
    actual_start_date = data.index.min().strftime('%Y-%m-%d %H:%M:%S')
    actual_end_date = data.index.max().strftime('%Y-%m-%d %H:%M:%S')

    for name, strategy in strategies.items():
        # This outer run is for grouping the optimization trials
        with mlflow.start_run(run_name=f"Optimize_{name}"):
            mlflow.log_param("start_date", actual_start_date)
            mlflow.log_param("end_date", actual_end_date)
            print(f"Optimizing {name}...")
            optimize_strategy(data, strategy, n_trials=n_trials_per_strategy, study_name=f'{experiment_name}-{name}', n_jobs=n_jobs)
            print(f"Optimization for {name} complete.")


def run_classification_optimizations(strategies, data_path, start_date, tracking_uri, experiment_name, n_trials_per_strategy=10, n_jobs=1):
    """
    Run optimization for a set of classification-based strategies.

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

    # Get the actual start and end dates from the data
    actual_start_date = data.index.min().strftime('%Y-%m-%d %H:%M:%S')
    actual_end_date = data.index.max().strftime('%Y-%m-%d %H:%M:%S')

    for name, strategy in strategies.items():
        # This outer run is for grouping the optimization trials
        with mlflow.start_run(run_name=f"Optimize_{name}"):
            mlflow.log_param("start_date", actual_start_date)
            mlflow.log_param("end_date", actual_end_date)
            print(f"Optimizing {name}...")
            optimize_classification_strategy(data, strategy, n_trials=n_trials_per_strategy, study_name=f'{experiment_name}-{name}', n_jobs=n_jobs)
            print(f"Optimization for {name} complete.")

