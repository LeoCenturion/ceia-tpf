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
from src.modeling.chronos_metalabeling_pipeline import ChronosMetaLabelingPipeline
from src.modeling.mlflow_utils import MLflowLogger
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def run_cpcv_for_metalabeling_pipeline(
    pipeline_cls,
    pipeline_config,
    raw_data,
    model_cls,
    model_params,
    experiment_name,
):
    """
    Runs Combinatorially Purged Cross-Validation for the ChronosMetaLabelingPipeline.
    """
    # pylint: disable=too-many-locals,too-many-statements
    mlflow_logger = MLflowLogger(experiment_name=experiment_name)
    run_name = f"CPCV_{pipeline_cls.__name__}"
    mlflow_logger.start_run(run_name=run_name)
    logger.info("Starting CPCV process for Chronos MetaLabeling pipeline...")

    try:
        # CPCV parameters from pipeline config
        n_groups = pipeline_config.get("n_groups", 10)
        k_test_groups = pipeline_config.get("k_test_groups", 2)
        pct_embargo = pipeline_config.get("pct_embargo", 0.01)

        # Log parameters to MLflow
        mlflow_logger.log_params(
            {
                "pipeline": pipeline_cls.__name__,
                "n_groups": n_groups,
                "k_test_groups": k_test_groups,
                "pct_embargo": pct_embargo,
            }
        )
        mlflow_logger.log_params(pipeline_config, prefix="pipeline_config")
        mlflow_logger.log_params(model_params, prefix="model_params")

        # --- Initial Pipeline Run on full raw_data for CPCV partitioning ---
        # This step is to get consistent `t1` and `X` (features) for the entire dataset
        # that will be used for `time_based_partition` and `purge_and_embargo_split`.
        # The actual feature engineering and labeling for model training will happen
        # inside each fold's `fit` method.
        logger.info("Performing initial pipeline run on full raw data for CPCV partitioning...")
        initial_pipeline = pipeline_cls(pipeline_config)
        
        # --- Step 1: Data Structuring ---
        logger.info(f"Raw data shape before structuring: {raw_data.shape}")
        initial_bars = initial_pipeline.step_1_data_structuring(raw_data)
        logger.info(f"Shape after step_1_data_structuring (bars): {initial_bars.shape}")
        if initial_bars.empty:
            raise ValueError("Initial data structuring resulted in no bars. Cannot proceed with CPCV.")
        
        # --- Step 2: Feature Engineering ---
        initial_features = initial_pipeline.step_2_feature_engineering(initial_bars)
        logger.info(f"Shape after step_2_feature_engineering (features): {initial_features.shape}")
        if initial_features.empty:
            raise ValueError("Initial feature engineering resulted in no features. Cannot proceed with CPCV.")

        # --- Step 3: Labeling and Weighting ---
        initial_labels, initial_sample_weights, initial_t1 = initial_pipeline.step_3_labeling_and_weighting(initial_bars)
        logger.info(f"Shape after step_3_labeling_and_weighting (labels, sw, t1): {initial_labels.shape}, {initial_sample_weights.shape}, {initial_t1.shape}")

        # Align all initial data to the common index for CPCV partitioning
        common_initial_index = initial_features.index.intersection(initial_labels.index).intersection(initial_t1.index)
        X_cpcv = initial_features.loc[common_initial_index]
        y_cpcv = initial_labels.loc[common_initial_index]
        t1_cpcv = initial_t1.loc[common_initial_index]
        sample_weights_cpcv = initial_sample_weights.loc[common_initial_index]

        logger.info(f"Shape after final alignment for CPCV: X_cpcv {X_cpcv.shape}, y_cpcv {y_cpcv.shape}")

        if X_cpcv.empty or y_cpcv.empty:
            raise ValueError("Aligned initial features or labels are empty after full pipeline run. Cannot proceed with CPCV.")

        logger.info(f"Initial pipeline run for CPCV partitioning complete. X_cpcv shape: {X_cpcv.shape}, y_cpcv shape: {y_cpcv.shape}")

        # Step 1: Data Partitioning (Time-based on X_cpcv index)
        path_indices = time_based_partition(X_cpcv.index, n_groups)

        # Step 2: Combinatorial Splitting
        splits = generate_combinatorial_splits(n_groups, k_test_groups)
        logger.info(f"Total combinations to test: {len(splits)}")

        # Step 3 & 4: Purging, Embargoing, Training, and Forecasting
        split_predictions = []
        for fold, (train_group_idxs, test_group_idxs) in enumerate(splits):
            logger.info(
                f"--- Fold {fold + 1}/{len(splits)} --- Test groups: {test_group_idxs}"
            )
            logger.info(f"Data size before purging: {len(X_cpcv)}")
            t_idxs = np.concatenate(
                [path_indices[i] for i in train_group_idxs if path_indices[i].size > 0]
            )
            logger.info(f'Train before purge (based on train_group_idxs): {len(X_cpcv.iloc[t_idxs])}')
            # Purge and embargo based on the initially processed data
            train_indices_pos, test_indices_pos = purge_and_embargo_split(
                X_cpcv, t1_cpcv, path_indices, train_group_idxs, test_group_idxs, pct_embargo
            )
            logger.debug(f'Train after purge: {len(X_cpcv.iloc[train_indices_pos])}')
            # Convert positional indices back to original raw_data index for slicing
            train_indices = X_cpcv.index[train_indices_pos]
            test_indices = X_cpcv.index[test_indices_pos]

            if test_indices.empty or train_indices.empty:
                logger.warning(
                    f"Skipping fold {fold + 1} due to empty train or test set after purging."
                )
                continue

            # Get raw data subsets for this fold
            train_data_raw = raw_data.loc[train_indices]
            test_data_raw = raw_data.loc[test_indices]

            # Create and fit a new pipeline for this fold on raw train data
            fold_pipeline = pipeline_cls(config=pipeline_config)
            try:
                # The fit method of ChronosMetaLabelingPipeline handles internal
                # data structuring, feature engineering, and labeling.
                # It returns X_test, y_test, t1_test from its *internal* train/test split.
                # Here, we only need it to train, the test data for CPCV comes from test_data_raw.
                fold_pipeline.fit(train_data_raw, model_cls, model_params)
            except ValueError as e:
                 logger.warning(f"Skipping fold {fold + 1} due to error during pipeline fitting: {e}")
                 continue
            except Exception as e: # Catch other potential errors during fit
                 logger.error(f"Unexpected error during fold {fold + 1} pipeline fitting: {e}")
                 continue

            # --- Generate predictions for the CPCV test set using the fitted fold_pipeline ---
            # Manually run test_data_raw through the pipeline's steps to get X_test and y_test
            test_bars_fold = fold_pipeline.step_1_data_structuring(test_data_raw)
            if test_bars_fold.empty:
                logger.warning(f"Skipping fold {fold+1} because no bars were generated for test data raw.")
                continue

            X_test_fold = fold_pipeline.step_2_feature_engineering(test_bars_fold)
            if X_test_fold.empty:
                logger.warning(f"Skipping fold {fold+1} because no features were generated for test bars fold.")
                continue

            y_test_fold, _, _ = fold_pipeline.step_3_labeling_and_weighting(test_bars_fold)
            
            # Align X_test_fold and y_test_fold
            common_test_fold_index = X_test_fold.index.intersection(y_test_fold.index)
            X_test_fold_aligned = X_test_fold.loc[common_test_fold_index]
            y_test_fold_aligned = y_test_fold.loc[common_test_fold_index]

            if X_test_fold_aligned.empty or y_test_fold_aligned.empty:
                logger.warning(f"Skipping fold {fold+1} because aligned test features or labels are empty.")
                continue
                
            # Get predictions from the fitted pipeline
            primary_preds = fold_pipeline.primary_model_.predict(X_test_fold_aligned)
            primary_probs = fold_pipeline.primary_model_.predict_proba(X_test_fold_aligned)[:, 1]

            X_meta_test = X_test_fold_aligned.copy()
            X_meta_test["primary_prob"] = primary_probs
            meta_preds = fold_pipeline.meta_model_.predict(X_meta_test)

            final_preds = (primary_preds == 1) & (meta_preds == 1)
            final_preds = final_preds.astype(int)
            
            preds_series = pd.Series(final_preds, index=X_test_fold_aligned.index)

            split_predictions.append(
                {"test_path_idxs": test_group_idxs, "preds": preds_series, "y_test": y_test_fold_aligned}
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

                # Calculate F1 for class 1 for logging and aggregation
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
    Main function to run the CPCV backtest for the ChronosMetaLabelingPipeline.
    """
    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    raw_data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        data_path=data_path
        # start_date="2022-01-01T00:00:00Z"
    )


    logger.info(
        f"Running CPCV for Chronos MetaLabeling strategy with {len(raw_data)} datapoints"
    )

    # Configuration for the ChronosMetaLabelingPipeline
    pipeline_config = {
        "volume_threshold": 5000,  # Significantly reduced to generate more bars
        "tau": 0.7,
        "n_groups": 6,
        "k_test_groups": 2,
        "pct_embargo": 0.01,
        "n_splits": 3, # For inner OOF generation
        "chronos_model_name": "amazon/chronos-t5-tiny",
        "chronos_window_size": 32, # Significantly reduced to generate more embeddings
        "chronos_stride": 1,  # Full overlap for more embeddings
    }

    # Primary Model (AutoGluon) parameters
    primary_model_params = {
        "label": "label",
        "eval_metric": "f1_weighted",
        "presets": "medium_quality",
        "time_limit": 60,
        "path": "AutogluonModels/tmp_cpcv_chronos_primary",
    }

    # Meta Model (AutoGluon) parameters
    meta_model_config = {
        "label": "label",
        "eval_metric": "f1",
        "presets": "best_quality",
        "time_limit": 30,
        "path": "AutogluonModels/tmp_cpcv_chronos_meta",
    }

    model_params = {
        "primary_model_params": primary_model_params,
        "meta_model_config": meta_model_config,
    }

    path_scores = run_cpcv_for_metalabeling_pipeline(
        pipeline_cls=ChronosMetaLabelingPipeline,
        pipeline_config=pipeline_config,
        raw_data=raw_data,
        model_cls=AutoGluonAdapter,
        model_params=model_params,
        experiment_name="Chronos_MetaLabeling_CPCV_Backtest",
    )

    if not path_scores:
        logger.error(
            "CPCV execution resulted in no valid paths. Please check data and configuration."
        )


if __name__ == "__main__":
    main()
