import logging
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

from src.backtesting.cpcv import (
    construct_backtest_paths,
    generate_combinatorial_splits,
    purge_and_embargo_split,
    time_based_partition,
)
from src.constants import CLOSE_COL, VOLUME_COL
from src.data_analysis.data_analysis import fetch_historical_data
from src.modeling.autogluon_adapter import AutoGluonAdapter
from src.modeling.chronos_feature_pipeline import ChronosFeaturePipeline
from src.modeling.mlflow_utils import MLflowLogger
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class ChronosFeaturePipelineCPCV(ChronosFeaturePipeline):
    """
    Extends ChronosFeaturePipeline with a dedicated method for running CPCV.
    """

    def run_cpcv(self, raw_data, model_cls, model_params, experiment_name):
        """
        Runs Combinatorially Purged Cross-Validation for the ChronosFeaturePipeline.
        """
        # pylint: disable=too-many-locals
        mlflow_logger = MLflowLogger(experiment_name=experiment_name)
        run_name = f"CPCV_{self.__class__.__name__}"
        mlflow_logger.start_run(run_name=run_name)
        logger.info("Starting CPCV process for Chronos pipeline...")

        try:
            # Steps 1-3: Data processing from pipeline
            bars = self.step_1_data_structuring(raw_data)
            features = self.step_2_feature_engineering(bars)
            y, sample_weights, t1_from_labeling = self.step_3_labeling_and_weighting(
                bars
            )

            common_index = features.index.intersection(y.index)
            X, y, sample_weights, t1 = (
                features.loc[common_index],
                y.loc[common_index],
                sample_weights.loc[common_index],
                t1_from_labeling.loc[common_index],
            )
            logger.info(f"Data aligned. X shape: {X.shape}, y shape: {y.shape}")

            # CPCV parameters from pipeline config
            n_groups = self.config.get("n_groups", 10)
            k_test_groups = self.config.get("k_test_groups", 2)
            pct_embargo = self.config.get("pct_embargo", 0.01)

            # Log parameters to MLflow
            mlflow_logger.log_params(
                {
                    "pipeline": self.__class__.__name__,
                    "n_groups": n_groups,
                    "k_test_groups": k_test_groups,
                    "pct_embargo": pct_embargo,
                }
            )
            mlflow_logger.log_params(self.config, prefix="pipeline_config")
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
                path_run_name = f"path_{i+1}"
                with mlflow.start_run(run_name=path_run_name, nested=True):
                    y_true = result["y_true"]
                    y_pred = result["y_pred"]

                    score = f1_score(y_true, y_pred, average="weighted", zero_division=0)
                    path_scores.append(score)
                    logger.info(
                        f"Path {i+1}/{len(path_results)} F1 Score (weighted): {score:.4f}"
                    )

                    report = classification_report(
                        y_true, y_pred, output_dict=True, zero_division=0
                    )

                    flat_report = {}
                    for class_label, metrics in report.items():
                        clean_class_label = class_label.replace(" ", "_")
                        if isinstance(metrics, dict):
                            for metric_name, value in metrics.items():
                                clean_metric_name = metric_name.replace("-", "_")
                                flat_report[
                                    f"{clean_class_label}_{clean_metric_name}"
                                ] = value
                        else:
                            flat_report[clean_class_label] = metrics

                    mlflow.log_metrics(flat_report)
                    mlflow.log_metric("f1_weighted", score)

            logger.info("--- CPCV Path Results ---")
            if path_scores:
                mean_f1 = np.mean(path_scores)
                std_f1 = np.std(path_scores)
                logger.info(
                    f"Individual Path F1 Scores: {[f'{s:.4f}' for s in path_scores]}"
                )
                logger.info(f"Mean Path F1 Score: {mean_f1:.4f}")
                logger.info(f"Std Dev of Path F1 Scores: {std_f1:.4f}")
                mlflow_logger.log_metrics({"f1_mean": mean_f1, "f1_std": std_f1})
            else:
                logger.warning("No complete backtest paths were evaluated.")

        finally:
            mlflow_logger.end_run()

        logger.info("CPCV process finished.")
        return path_scores


@setup_logging
def main():
    """
    Main function to run the CPCV backtest for the ChronosFeaturePipeline.
    """
    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    raw_data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        data_path=data_path
    )
    # raw_data.rename(columns={VOLUME_COL: "volume", CLOSE_COL: "close"}, inplace=True)

    logger.info(
        f"Running CPCV for Chronos strategy with {len(raw_data)} datapoints"
    )

    # Configuration for the ChronosFeaturePipeline
    pipeline_config = {
        "volume_threshold": 50000,
        "tau": 0.7,
        "n_groups": 6,
        "k_test_groups": 2,
        "pct_embargo": 0.01,
        "chronos_model_name": "amazon/chronos-t5-tiny",
        "chronos_window_size": 128,
        "chronos_stride": 1,  # Increased stride for faster CPCV run
    }

    # AutoGluon model parameters
    model_params = {
        "label": "label",
        "eval_metric": "f1_weighted",
        "problem_type": "binary",
        "presets": "medium_quality",
        "time_limit": 300,  # Reduced time limit for faster CPCV fold
        "verbosity": 1,
        "path": "AutogluonModels/chronos_cpcv_run",
    }

    pipeline = ChronosFeaturePipelineCPCV(pipeline_config)
    pipeline.run_cpcv(
        raw_data=raw_data,
        model_cls=AutoGluonAdapter,
        model_params=model_params,
        experiment_name="Chronos_Features_CPCV",
    )


if __name__ == "__main__":
    main()
