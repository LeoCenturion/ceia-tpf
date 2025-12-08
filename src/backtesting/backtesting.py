"""Backtesting framework with Optuna optimization and MLflow logging."""
import os
import re

import mlflow
import numpy as np
import optuna
import pandas as pd
from backtesting import Backtest, Strategy
from sklearn.metrics import f1_score
from abc import  abstractmethod
from src.data_analysis import (
    adjust_data_to_ubtc,
    fetch_historical_data,
)


class TrialStrategy(Strategy):
    """Base strategy class for Optuna trials with artifact saving."""

    # trades_df = pd.DataFrame()

    def save_artifacts(self, trial, stats, bt):
        """Save backtest plot and trades to MLflow."""
        plot_filename = f"backtest_plot_trial_{trial.number}.html"
        trades_filename = f"trades_trial_{trial.number}.csv"

        bt.plot(filename=plot_filename, open_browser=False)
        if os.path.exists(plot_filename):
            mlflow.log_artifact(plot_filename)
            # Save trades
        trades_df = stats["_trades"]
        if not trades_df.empty:
            trades_df.to_csv(trades_filename, index=False)
            mlflow.log_artifact(trades_filename)

            # Clean up created files
            if os.path.exists(plot_filename):
                os.remove(plot_filename)
            if os.path.exists(trades_filename):
                os.remove(trades_filename)

    @classmethod
    def get_optuna_params(cls, _trial):
        """Define the hyperparameter space for Optuna."""
        return {}

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def next(self):
        pass

def sanitize_metric_name(name):
    """Sanitize metric name to be MLflow compliant."""
    return re.sub(r"[^a-zA-Z0-9_\-.\s:/]", "", name)


def optimize_strategy(  # pylint: disable=too-many-arguments
    data, strategy_class: TrialStrategy, study_name, n_trials=100, n_jobs=8
):
    """
    Optimize strategy hyperparameters using Optuna.
    For each trial, run an expanding window backtest and log the averaged stats to MLflow.
    """
    parent_run = mlflow.active_run()
    parent_run_id = parent_run.info.run_id if parent_run else None

    def objective(trial):
        params = strategy_class.get_optuna_params(trial)

        bt: Backtest = Backtest(data, strategy_class, cash=10000, commission=0.001)
        stats = bt.run(**params)

        run_name = f"{study_name}-" + "-".join([f"{k}={v}" for k, v in params.items()])
        tags = {"mlflow.parentRunId": parent_run_id} if parent_run_id else {}
        with mlflow.start_run(run_name=run_name, tags=tags, nested=True):
            mlflow.log_params(params)

            for key, value in stats.items():
                print(f" key value {key} {value} {type(value)}")
                if isinstance(value, pd.Timestamp):
                    value = value.timestamp()
                if isinstance(value, pd.Timedelta):
                    value = value.total_seconds()
                if not np.issubdtype(type(value), np.number):
                    print("skipping")
                    continue
                sanitized_key = sanitize_metric_name(key)
                mlflow.log_metric(sanitized_key, value)
            # Log artifacts for the last step if available
            if bt:
                stats._strategy.save_artifacts(  # pylint: disable=protected-access
                    trial, stats, bt
                )

        sharpe_ratio = stats.get("Sharpe Ratio", 0)
        if sharpe_ratio is None or np.isnan(sharpe_ratio):
            return 0.0
        return sharpe_ratio

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage="sqlite:///optuna-study.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs)
    try:
        return study.best_params
    except ValueError:
        print(f"Study {study_name} finished, but no best trial was found.")
        return None


def optimize_classification_strategy(  # pylint: disable=too-many-arguments
    data, strategy, study_name, n_trials=100, n_jobs=8
):
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
                print(f" key value {key} {value} {type(value)}")
                if isinstance(value, pd.Timestamp):
                    value = value.timestamp()
                if isinstance(value, pd.Timedelta):
                    value = value.total_seconds()
                if not np.issubdtype(type(value), np.number):
                    print("skipping")
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
                trades_df = stats["_trades"]
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
        if hasattr(strategy_instance, "y_true") and hasattr(
            strategy_instance, "y_pred"
        ):
            if len(strategy_instance.y_true) > 0 and len(strategy_instance.y_pred) > 0:
                f1 = f1_score(
                    strategy_instance.y_true, strategy_instance.y_pred, zero_division=0
                )
                stats["F1 Score"] = f1
                mlflow.log_metric("F1 Score", f1)

        f1_val = stats.get("F1 Score", 0)
        if f1_val is None or np.isnan(f1_val):
            return 0.0
        return f1_val

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage="sqlite:///optuna-study.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs)
    try:
        return study.best_params
    except ValueError:
        print(f"Study {study_name} finished, but no best trial was found.")
        return None


def run_optimizations(  # pylint: disable=too-many-arguments
    strategies,
    data_path,
    start_date,
    tracking_uri,
    experiment_name,
    n_trials_per_strategy=10,
    n_jobs=1,
    timeframe="1h",
):
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
        data_path=data_path, start_date=start_date, timeframe=timeframe
    )
    data = adjust_data_to_ubtc(data)
    # Get the actual start and end dates from the data
    actual_start_date = data.index.min().strftime("%Y-%m-%d %H:%M:%S")
    actual_end_date = data.index.max().strftime("%Y-%m-%d %H:%M:%S")

    for name, strategy in strategies.items():
        # This outer run is for grouping the optimization trials
        with mlflow.start_run(run_name=f"Optimize_{name}"):
            mlflow.log_param("start_date", actual_start_date)
            mlflow.log_param("end_date", actual_end_date)
            print(f"Optimizing {name}...")
            optimize_strategy(
                data,
                strategy,
                n_trials=n_trials_per_strategy,
                study_name=f"{experiment_name}-{name}",
                n_jobs=n_jobs,
            )
            print(f"Optimization for {name} complete.")


def run_classification_optimizations(  # pylint: disable=too-many-arguments
    strategies,
    data_path,
    start_date,
    tracking_uri,
    experiment_name,
    n_trials_per_strategy=10,
    n_jobs=1,
):
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

    data = fetch_historical_data(data_path=data_path, start_date=start_date)
    data = adjust_data_to_ubtc(data)

    # Get the actual start and end dates from the data
    actual_start_date = data.index.min().strftime("%Y-%m-%d %H:%M:%S")
    actual_end_date = data.index.max().strftime("%Y-%m-%d %H:%M:%S")

    for name, strategy in strategies.items():
        # This outer run is for grouping the optimization trials
        with mlflow.start_run(run_name=f"Optimize_{name}"):
            mlflow.log_param("start_date", actual_start_date)
            mlflow.log_param("end_date", actual_end_date)
            print(f"Optimizing {name}...")
            optimize_classification_strategy(
                data,
                strategy,
                n_trials=n_trials_per_strategy,
                study_name=f"{experiment_name}-{name}",
                n_jobs=n_jobs,
            )
            print(f"Optimization for {name} complete.")


# AI add a function to perform combinatorially symetric cross validation AI! 
