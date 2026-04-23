import argparse
import copy
import logging
from typing import Any, Dict

import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score

from src.backtesting.cpcv import (
    construct_backtest_paths,
    generate_combinatorial_splits,
    purge_and_embargo_split,
    time_based_partition,
)
from src.data_analysis.data_analysis import fetch_historical_data
from src.modeling.autogluon_adapter import AutoGluonAdapter
from src.modeling.mlflow_utils import MLflowLogger
from src.modeling.pipeline import AbstractMLPipeline
from src.modeling.xgboost_pipeline_palazzo import PalazzoXGBoostPipeline
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def run_cpcv_for_pipeline(
    pipeline: PalazzoXGBoostPipeline,
    raw_data: pd.DataFrame,
    model_cls: Any,
    model_params: Dict[str, Any],
    experiment_name: str,
) -> float:
    """
    Runs Combinatorially Purged Cross-Validation for a given ML pipeline.
    """
    # pylint: disable=too-many-locals
    mlflow_logger = MLflowLogger(experiment_name=experiment_name)
    run_name = f"CPCV_{pipeline.__class__.__name__}"
    mean_f1 = 0.0
    with mlflow_logger.start_run(run_name=run_name):
        # Steps 1-3: Data processing from pipeline
        bars = pipeline.step_1_data_structuring(raw_data)  # type: ignore
        if bars is None:
            logger.error("Data structuring returned None. Aborting CPCV.")
            return 0.0
        features = pipeline.step_2_feature_engineering(bars)  # type: ignore
        if features is None:
            logger.error("Feature engineering returned None. Aborting CPCV.")
            return 0.0
        (
            y,
            sample_weights,
            t1_from_labeling,
        ) = pipeline.step_3_labeling_and_weighting(bars)
        if y is None or sample_weights is None or t1_from_labeling is None:
            logger.error("Labeling and weighting returned None. Aborting CPCV.")
            return 0.0

        common_index = features.index.intersection(y.index)
        X, y, sample_weights, t1 = (
            features.loc[common_index],
            y.loc[common_index],
            sample_weights.loc[common_index],
            t1_from_labeling.loc[common_index],
        )
        logger.info(f"Data aligned. X shape: {X.shape}, y shape: {y.shape}")

        # CPCV parameters from pipeline config
        n_groups = pipeline.config.get("n_groups", 10)
        k_test_groups = pipeline.config.get("k_test_groups", 2)
        pct_embargo = pipeline.config.get("pct_embargo", 0.01)

        # Log parameters to MLflow
        mlflow_logger.log_params(
            {
                "pipeline": pipeline.__class__.__name__,
                "n_groups": n_groups,
                "k_test_groups": k_test_groups,
                "pct_embargo": pct_embargo,
            }
        )
        mlflow_logger.log_params(pipeline.config, prefix="pipeline_config")
        mlflow_logger.log_params(model_params, prefix="model_params")

        # Step 1: Data Partitioning (Time-based)
        path_indices = time_based_partition(X.index, n_groups)

        # Step 2: Combinatorial Splitting
        splits = generate_combinatorial_splits(n_groups, k_test_groups)
        logger.info(f"Total combinations to test: {len(splits)}")

        # Step 3 & 4: Purging, Embargoing, Training, and Forecasting
        split_predictions = []
        for fold, (train_group_idxs, test_group_idxs) in enumerate(splits):
            logger.info(
                f"--- Fold {fold + 1}/{len(splits)} --- Test groups: {test_group_idxs}"
            )

            train_indices, test_indices = purge_and_embargo_split(
                X, t1, path_indices, train_group_idxs, test_group_idxs, pct_embargo
            )

            if test_indices.size == 0 or train_indices.size == 0:
                logger.warning(
                    f"Skipping fold {fold + 1} due to empty train or test set after purging."
                )
                continue

            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            sw_train = sample_weights.iloc[train_indices]

            model = model_cls(**model_params)

            # The model is expected to handle dataframe inputs directly.
            # For XGBoost, this is enabled by `device='cuda'`.
            # For AutoGluon, the adapter handles it.
            model.fit(X_train, y_train, sample_weight=sw_train)
            preds = model.predict(X_test)

            split_predictions.append(
                {
                    "test_path_idxs": test_group_idxs,
                    "preds": preds,
                    "y_test": y_test,
                }
            )

        # Step 5: Backtest Path Construction
        logger.info("--- Constructing and Evaluating Backtest Paths ---")

        path_results = construct_backtest_paths(
            split_predictions, n_groups, k_test_groups
        )

        path_scores = []
        for i, result in enumerate(path_results):
            path_run_name = f"path_{i + 1}"
            with mlflow.start_run(run_name=path_run_name, nested=True):
                y_true = result["y_true"]
                y_pred = result["y_pred"]

                # Calculate F1 score for logging and aggregation
                score = f1_score(
                    y_true, y_pred, average="weighted", zero_division="warn"
                )
                path_scores.append(score)
                logger.info(
                    f"Path {i + 1}/{len(path_results)} F1 Score (weighted): {score:.4f}"
                )

                # Generate detailed classification report
                report: Dict[str, Any] = classification_report(
                    y_true, y_pred, output_dict=True, zero_division="warn"
                )

                # Flatten the report for MLflow logging
                if isinstance(report, dict):
                    flat_report = {}
                    for class_label, metrics in report.items():
                        clean_class_label = class_label.replace(" ", "_")
                        if isinstance(metrics, dict):
                            for metric_name, value in metrics.items():
                                clean_metric_name = metric_name.replace("-", "_")
                                flat_report[
                                    f"{clean_class_label}_{clean_metric_name}"
                                ] = float(value)
                        else:
                            flat_report[clean_class_label] = float(metrics)

                    mlflow.log_metrics(flat_report)
                mlflow.log_metric("f1_weighted", float(score))

        logger.info("--- CPCV Path Results ---")
        mean_f1 = np.mean(path_scores) if path_scores else 0.0
        std_f1 = np.std(path_scores) if path_scores else 0.0
        if path_scores:
            logger.info(
                f"Individual Path F1 Scores: {[f'{s:.4f}' for s in path_scores]}"
            )
            logger.info(f"Mean Path F1 Score: {mean_f1:.4f}")
            logger.info(f"Std Dev of Path F1 Scores: {std_f1:.4f}")
            mlflow_logger.log_metrics({"f1_mean": mean_f1, "f1_std": std_f1})
        else:
            logger.warning("No complete backtest paths were evaluated.")

    logger.info("CPCV process finished.")
    return float(mean_f1)


@setup_logging
def main():
    """
    Main function to run the CPCV backtest for the Palazzo pipeline with XGBoost or AutoGluon.
    """
    parser = argparse.ArgumentParser(
        description="Run CPCV backtest for the Palazzo pipeline."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["xgboost", "autogluon"],
        help="The model to run the backtest with.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable Optuna hyperparameter optimization for AutoGluon.",
    )
    args = parser.parse_args()

    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        data_path=data_path,
        # start_date="2024-01-01T00:00:00Z",  # Use a smaller subset for quicker testing
    )

    logger.info(
        f"Running CPCV for Palazzo strategy with {len(data)} datapoints using {args.model}"
    )

    # Configuration for the pipeline
    pipeline_config = {
        "volume_threshold": 50000,
        "tau": 0.7,
        "n_groups": 10,
        "k_test_groups": 2,
        "pct_embargo": 0.01,
    }
    pipeline = PalazzoXGBoostPipeline(pipeline_config)

    if args.model == "xgboost":
        model_cls = xgb.XGBClassifier
        model_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": "cuda",
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "min_child_weight": 5,
        }
        experiment_name = "Palazzo_CPCV_Backtest"
        run_cpcv_for_pipeline(
            pipeline=pipeline,
            raw_data=data,
            model_cls=model_cls,
            model_params=model_params,
            experiment_name=experiment_name,
        )

    elif args.model == "autogluon":
        if args.optimize:

            def autogluon_objective(trial: optuna.Trial) -> float:
                try:
                    base_model_params = {
                        "label": "label",
                        "eval_metric": "f1_weighted",
                        "hyperparameters": {
                            "FT_TRANSFORMER": {},
                            "GBM": {},
                            "NN_TORCH": {},
                            "FASTAI": {},
                        },
                        "time_limit": 600,
                        "verbosity": 1,
                    }

                    presets = trial.suggest_categorical(
                        "presets", ["medium_quality", "high_quality", "best_quality"]
                    )

                    model_params = copy.deepcopy(base_model_params)
                    model_params["presets"] = presets
                    model_params["path"] = (
                        f"AutogluonModels/palazzo_cpcv_autogluon_trial_{trial.number}"
                    )

                    return run_cpcv_for_pipeline(
                        pipeline=pipeline,
                        raw_data=data,
                        model_cls=AutoGluonAdapter,
                        model_params=model_params,
                        experiment_name="Palazzo_CPCV_AutoGluon_Optuna",
                    )
                except Exception as e:
                    logger.warning(f"Trial {trial.number} failed with error: {e}")
                    return 0.0

            study = optuna.create_study(
                direction="maximize",
                study_name="palazzo_autogluon_cpcv_optimization",
                storage="sqlite:///optuna-study.db",
                load_if_exists=True,
            )
            study.optimize(autogluon_objective, n_trials=10)
            logger.info(f"Best trial: {study.best_trial.value}")
            logger.info(f"Best params: {study.best_trial.params}")
        else:
            logger.info("Running single AutoGluon CPCV without Optuna optimization.")
            model_cls = AutoGluonAdapter
            model_params = {
                "label": "label",
                "eval_metric": "f1_weighted",
                "problem_type": "binary",
                "presets": "medium_quality",
                "time_limit": 600,  # Longer time for a single run
                "verbosity": 1,
                "path": "AutogluonModels/palazzo_cpcv_autogluon_run_single",  # Unique path for single run
            }
            experiment_name = "Palazzo_CPCV_AutoGluon_Single_Run"
            run_cpcv_for_pipeline(
                pipeline=pipeline,
                raw_data=data,
                model_cls=model_cls,
                model_params=model_params,
                experiment_name=experiment_name,
            )


if __name__ == "__main__":
    main()
