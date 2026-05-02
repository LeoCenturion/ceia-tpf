import logging
import os
import tempfile
from typing import Any, Dict, List, Type, Union

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from src.backtesting.backtesting import TrialStrategy
from src.backtesting.cpcv import (
    construct_backtest_paths,
    generate_combinatorial_splits,
    purge_and_embargo_split,
    time_based_partition,
)
from src.backtesting.ratios import (
    calmar_ratio,
    deflated_sharpe_ratio,
    path_to_returns,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from src.backtesting.strategies.statistical_strategies import SmaCross
from src.data_analysis.data_analysis import adjust_data_to_ubtc, fetch_historical_data
from src.modeling.mlflow_utils import MLflowLogger

logger = logging.getLogger(__name__)


def _save_paths_artifact(paths: list) -> None:
    """
    Serialize all path predictions to a CSV and log it as an MLflow artifact
    under the ``predictions/`` directory of the currently active run.
    """
    rows = []
    for i, path in enumerate(paths):
        for yt, yp in zip(path["y_true"], path["y_pred"]):
            rows.append({"path": i + 1, "y_true": float(yt), "y_pred": float(yp)})

    df = pd.DataFrame(rows)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="cpcv_predictions_"
    ) as f:
        df.to_csv(f, index=False)
        tmp_path = f.name

    try:
        mlflow.log_artifact(tmp_path, artifact_path="predictions")
        logger.info(
            f"Saved predictions artifact: {len(df)} rows across {len(paths)} paths."
        )
    finally:
        os.remove(tmp_path)


def _evaluate_paths_returns(
    path_results: list,
    mlflow_logger: MLflowLogger,
    periods_per_year: int = 365 * 24,
) -> List[float]:
    """
    Evaluate CPCV paths using return-based metrics.

    Computes annualized Sharpe ratio, Calmar ratio, and Probabilistic Sharpe
    Ratio per path (logged to nested MLflow runs), then logs the Deflated
    Sharpe Ratio as the multiple-testing-corrected summary to the parent run.

    Expects path_results entries with ``y_true`` as Close prices and
    ``y_pred`` as position signals (−1, 0, 1).
    """
    all_returns: List[pd.Series] = []
    path_sharpes: List[float] = []
    path_psrs: List[float] = []

    for i, path in enumerate(path_results):
        with mlflow.start_run(run_name=f"path_{i + 1}", nested=True):
            rets = path_to_returns(path, commission=0.001)
            all_returns.append(rets)

            sr = sharpe_ratio(rets, periods_per_year)
            cr = calmar_ratio(rets, periods_per_year)
            psr = probabilistic_sharpe_ratio(rets)

            safe_sr = float(sr) if np.isfinite(sr) else 0.0
            path_sharpes.append(safe_sr)
            if np.isfinite(psr):
                path_psrs.append(float(psr))

            metrics: Dict[str, float] = {"sharpe_ratio": safe_sr}
            if np.isfinite(cr):
                metrics["calmar_ratio"] = float(cr)
            if np.isfinite(psr):
                metrics["probabilistic_sharpe_ratio"] = float(psr)
            mlflow.log_metrics(metrics)

            cr_str = f"{cr:.4f}" if np.isfinite(cr) else "nan"
            psr_str = f"{psr:.4f}" if np.isfinite(psr) else "nan"
            logger.info(
                f"Path {i + 1}/{len(path_results)} | "
                f"Sharpe: {safe_sr:.4f} | Calmar: {cr_str} | PSR: {psr_str}"
            )

    if path_sharpes:
        dsr = deflated_sharpe_ratio(all_returns)
        summary: Dict[str, float] = {
            "sharpe_mean": float(np.nanmean(path_sharpes)),
            "sharpe_std": float(np.nanstd(path_sharpes)),
        }
        if np.isfinite(dsr):
            summary["deflated_sharpe_ratio"] = float(dsr)
        if path_psrs:
            summary["psr_mean"] = float(np.mean(path_psrs))
        mlflow_logger.log_metrics(summary)
        dsr_str = f"{dsr:.4f}" if np.isfinite(dsr) else "nan"
        psr_mean_str = f"{np.mean(path_psrs):.4f}" if path_psrs else "nan"
        logger.info(
            f"Sharpe across paths: {[f'{s:.4f}' for s in path_sharpes]} | "
            f"Mean: {np.nanmean(path_sharpes):.4f} | DSR: {dsr_str} | PSR mean: {psr_mean_str}"
        )
    else:
        logger.warning("No complete backtest paths were evaluated.")

    return path_sharpes


def _evaluate_paths_f1(
    path_results: list,
    mlflow_logger: MLflowLogger,
) -> List[float]:
    """Evaluate backtest paths with weighted F1 and log nested MLflow runs."""
    _save_paths_artifact(path_results)
    path_scores: List[float] = []
    for i, result in enumerate(path_results):
        with mlflow.start_run(run_name=f"path_{i + 1}", nested=True):
            y_true = result["y_true"]
            y_pred = result["y_pred"]

            score = f1_score(y_true, y_pred, average="weighted", zero_division="warn")
            path_scores.append(float(score))
            logger.info(
                f"Path {i + 1}/{len(path_results)} F1 Score (weighted): {score:.4f}"
            )

            report: Union[Dict[str, Any], str] = classification_report(
                y_true, y_pred, output_dict=True, zero_division="warn"
            )
            if isinstance(report, dict):
                flat_report: Dict[str, float] = {}
                for class_label, metrics in report.items():
                    clean_label = class_label.replace(" ", "_")
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            flat_report[
                                f"{clean_label}_{metric_name.replace('-', '_')}"
                            ] = float(value)
                    else:
                        flat_report[clean_label] = float(metrics)
                mlflow.log_metrics(flat_report)
            mlflow.log_metric("f1_weighted", float(score))

    if path_scores:
        logger.info(f"Individual Path F1 Scores: {[f'{s:.4f}' for s in path_scores]}")
        logger.info(f"Mean Path F1 Score: {np.mean(path_scores):.4f}")
        logger.info(f"Std Dev of Path F1 Scores: {np.std(path_scores):.4f}")
        mlflow_logger.log_metrics(
            {
                "f1_mean": float(np.mean(path_scores)),
                "f1_std": float(np.std(path_scores)),
            }
        )
    else:
        logger.warning("No complete backtest paths were evaluated.")

    return path_scores


def run_cpcv_for_strategy(
    data: pd.DataFrame,
    t1: pd.Series,
    strategy_class: Type[TrialStrategy],
    strategy_params: dict,
    n_groups: int,
    k_test_groups: int,
    embargo_pct: float,
    experiment_name: str,
):
    """
    Runs CPCV for a given TrialStrategy and logs Sharpe ratios to MLflow.
    """
    mlflow_logger = MLflowLogger(experiment_name=experiment_name)
    mlflow_logger.start_run(run_name=f"CPCV_{strategy_class.__name__}")

    try:
        mlflow_logger.log_params(
            {
                "strategy": strategy_class.__name__,
                "n_groups": n_groups,
                "k_test_groups": k_test_groups,
                "embargo_pct": embargo_pct,
            }
        )
        mlflow_logger.log_params(strategy_params)

        index = pd.to_datetime(data.index)
        years = (index[-1] - index[0]).total_seconds() / (365.25 * 24 * 3600)
        periods_per_year = int(len(data) / years)

        path_indices = time_based_partition(index, n_groups)
        logger.info(f"Data partitioned into {n_groups} groups.")

        splits = generate_combinatorial_splits(n_groups, k_test_groups)
        logger.info(f"Generated {len(splits)} combinatorial splits.")

        split_predictions = []

        for i, (train_split, test_split) in enumerate(splits):
            logger.info(
                f"Processing split {i + 1}/{len(splits)}: Train={train_split}, Test={test_split}"
            )
            train_indices, test_indices = purge_and_embargo_split(
                data, t1, path_indices, train_split, test_split, embargo_pct
            )

            if test_indices.size == 0 or train_indices.size == 0:
                logger.warning(
                    f"Skipping split {i + 1} due to empty train or test set after purging."
                )
                continue

            test_data = data.iloc[test_indices]
            y_test = data["Close"].iloc[test_indices]

            strategy_instance = strategy_class(
                broker=None, data=test_data, params=strategy_params
            )
            predictions = strategy_instance.predict(test_data)

            split_predictions.append(
                {
                    "test_path_idxs": test_split,
                    "preds": predictions,
                    "y_test": y_test,
                }
            )

        paths = construct_backtest_paths(split_predictions, n_groups, k_test_groups)
        logging.info(f"Constructed {len(paths)} backtest paths.")

        _save_paths_artifact(paths)
        _evaluate_paths_returns(paths, mlflow_logger, periods_per_year)

    finally:
        mlflow_logger.end_run()

    return paths


def run_cpcv_for_ml_pipeline(
    pipeline: Any,
    raw_data: pd.DataFrame,
    model_cls: Any,
    model_params: Dict[str, Any],
    experiment_name: str,
) -> float:
    """
    Generic CPCV runner for ML pipelines with a step_1/step_2/step_3 interface.

    Features are precomputed once on the full dataset. Per-fold logic fits a
    fresh model on the train slice and predicts on the test slice. Suitable for
    pipelines like PalazzoXGBoostPipeline or ChronosFeaturePipeline.

    The pipeline must expose:
      - step_1_data_structuring(raw_data)
      - step_2_feature_engineering(bars)
      - step_3_labeling_and_weighting(bars) -> (y, sample_weights, t1)
      - config (dict) with n_groups, k_test_groups, pct_embargo
    """
    mlflow_logger = MLflowLogger(experiment_name=experiment_name)
    run_name = f"CPCV_{pipeline.__class__.__name__}"
    mean_f1 = 0.0
    with mlflow_logger.start_run(run_name=run_name):
        bars = pipeline.step_1_data_structuring(raw_data)
        if bars is None or (hasattr(bars, "empty") and bars.empty):
            logger.error("Data structuring failed. Aborting CPCV.")
            return 0.0
        features = pipeline.step_2_feature_engineering(bars)
        if features is None or (hasattr(features, "empty") and features.empty):
            logger.error("Feature engineering failed. Aborting CPCV.")
            return 0.0
        y, sample_weights, t1 = pipeline.step_3_labeling_and_weighting(bars)
        if y is None or sample_weights is None or t1 is None:
            logger.error("Labeling and weighting failed. Aborting CPCV.")
            return 0.0

        common_index = features.index.intersection(y.index)
        X = features.loc[common_index]
        y = y.loc[common_index]
        sample_weights = sample_weights.loc[common_index]
        t1 = t1.loc[common_index]
        logger.info(f"Data aligned. X shape: {X.shape}, y shape: {y.shape}")

        n_groups = pipeline.config.get("n_groups", 10)
        k_test_groups = pipeline.config.get("k_test_groups", 2)
        pct_embargo = pipeline.config.get("pct_embargo", 0.01)

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

        path_indices = time_based_partition(X.index, n_groups)
        splits = generate_combinatorial_splits(n_groups, k_test_groups)
        logger.info(f"Total combinations to test: {len(splits)}")

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

        logger.info("--- Constructing and Evaluating Backtest Paths ---")
        path_results = construct_backtest_paths(
            split_predictions, n_groups, k_test_groups
        )
        path_scores = _evaluate_paths_f1(path_results, mlflow_logger)
        mean_f1 = float(np.mean(path_scores)) if path_scores else 0.0

    return mean_f1


def run_cpcv_for_metalabeling_pipeline(
    pipeline_cls: Any,
    pipeline_config: Dict[str, Any],
    raw_data: pd.DataFrame,
    model_cls: Any,
    model_params: Dict[str, Any],
    experiment_name: str,
) -> List[float]:
    """
    CPCV runner for pipelines that need per-fold refitting on raw data with two-stage prediction.

    Runs step_1/step_2/step_3 on the full dataset once to establish CPCV partitions,
    then fits a fresh pipeline per fold on raw training data.

    The fitted pipeline must expose `primary_model_` and `meta_model_` attributes
    for the two-stage (primary + meta) prediction step.
    """
    # pylint: disable=too-many-locals,too-many-statements
    mlflow_logger = MLflowLogger(experiment_name=experiment_name)
    run_name = f"CPCV_{pipeline_cls.__name__}"
    mlflow_logger.start_run(run_name=run_name)
    logger.info("Starting CPCV process for metalabeling pipeline...")
    path_scores: List[float] = []
    try:
        n_groups = pipeline_config.get("n_groups", 10)
        k_test_groups = pipeline_config.get("k_test_groups", 2)
        pct_embargo = pipeline_config.get("pct_embargo", 0.01)

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

        # Run on full data once to get consistent t1 and features for CPCV partitioning
        logger.info(
            "Performing initial pipeline run on full raw data for CPCV partitioning..."
        )
        initial_pipeline = pipeline_cls(pipeline_config)

        logger.info(f"Raw data shape before structuring: {raw_data.shape}")
        initial_bars = initial_pipeline.step_1_data_structuring(raw_data)
        logger.info(f"Shape after step_1_data_structuring (bars): {initial_bars.shape}")
        if initial_bars.empty:
            raise ValueError(
                "Initial data structuring resulted in no bars. Cannot proceed with CPCV."
            )

        initial_features = initial_pipeline.step_2_feature_engineering(initial_bars)
        logger.info(
            f"Shape after step_2_feature_engineering (features): {initial_features.shape}"
        )
        if initial_features.empty:
            raise ValueError(
                "Initial feature engineering resulted in no features. Cannot proceed with CPCV."
            )

        (
            initial_labels,
            initial_sample_weights,
            initial_t1,
        ) = initial_pipeline.step_3_labeling_and_weighting(initial_bars)
        logger.info(
            f"Shape after step_3_labeling_and_weighting (labels, sw, t1): "
            f"{initial_labels.shape}, {initial_sample_weights.shape}, {initial_t1.shape}"
        )

        common_initial_index = initial_features.index.intersection(
            initial_labels.index
        ).intersection(initial_t1.index)
        X_cpcv = initial_features.loc[common_initial_index]
        y_cpcv = initial_labels.loc[common_initial_index]
        t1_cpcv = initial_t1.loc[common_initial_index]

        logger.info(
            f"Shape after final alignment for CPCV: X_cpcv {X_cpcv.shape}, y_cpcv {y_cpcv.shape}"
        )
        if X_cpcv.empty or y_cpcv.empty:
            raise ValueError(
                "Aligned initial features or labels are empty. Cannot proceed with CPCV."
            )

        logger.info(
            f"Initial pipeline run complete. X_cpcv shape: {X_cpcv.shape}, y_cpcv shape: {y_cpcv.shape}"
        )

        path_indices = time_based_partition(X_cpcv.index, n_groups)
        splits = generate_combinatorial_splits(n_groups, k_test_groups)
        logger.info(f"Total combinations to test: {len(splits)}")

        split_predictions = []
        for fold, (train_group_idxs, test_group_idxs) in enumerate(splits):
            logger.info(
                f"--- Fold {fold + 1}/{len(splits)} --- Test groups: {test_group_idxs}"
            )
            logger.info(f"Data size before purging: {len(X_cpcv)}")

            t_idxs = np.concatenate(
                [path_indices[i] for i in train_group_idxs if path_indices[i].size > 0]
            )
            logger.info(
                f"Train before purge (based on train_group_idxs): {len(X_cpcv.iloc[t_idxs])}"
            )

            train_indices_pos, test_indices_pos = purge_and_embargo_split(
                X_cpcv,
                t1_cpcv,
                path_indices,
                train_group_idxs,
                test_group_idxs,
                pct_embargo,
            )
            logger.debug(f"Train after purge: {len(X_cpcv.iloc[train_indices_pos])}")

            train_indices = X_cpcv.index[train_indices_pos]
            test_indices = X_cpcv.index[test_indices_pos]

            if test_indices.empty or train_indices.empty:
                logger.warning(
                    f"Skipping fold {fold + 1} due to empty train or test set after purging."
                )
                continue

            train_data_raw = raw_data.loc[train_indices]
            test_data_raw = raw_data.loc[test_indices]

            fold_pipeline = pipeline_cls(config=pipeline_config)
            try:
                fold_pipeline.fit(train_data_raw, model_cls, model_params)
            except ValueError as e:
                logger.warning(
                    f"Skipping fold {fold + 1} due to error during pipeline fitting: {e}"
                )
                continue
            except Exception as e:  # pylint: disable=broad-except
                logger.error(
                    f"Unexpected error during fold {fold + 1} pipeline fitting: {e}"
                )
                continue

            test_bars_fold = fold_pipeline.step_1_data_structuring(test_data_raw)
            if test_bars_fold.empty:
                logger.warning(
                    f"Skipping fold {fold + 1}: no bars generated for test data."
                )
                continue

            X_test_fold = fold_pipeline.step_2_feature_engineering(test_bars_fold)
            if X_test_fold.empty:
                logger.warning(
                    f"Skipping fold {fold + 1}: no features generated for test bars."
                )
                continue

            y_test_fold, _, _ = fold_pipeline.step_3_labeling_and_weighting(
                test_bars_fold
            )

            common_test_fold_index = X_test_fold.index.intersection(y_test_fold.index)
            X_test_fold = X_test_fold.loc[common_test_fold_index]
            y_test_fold = y_test_fold.loc[common_test_fold_index]

            if X_test_fold.empty or y_test_fold.empty:
                logger.warning(
                    f"Skipping fold {fold + 1}: aligned test features or labels are empty."
                )
                continue

            primary_preds = fold_pipeline.primary_model_.predict(X_test_fold)
            primary_probs = fold_pipeline.primary_model_.predict_proba(X_test_fold)[
                :, 1
            ]

            X_meta_test = X_test_fold.copy()
            X_meta_test["primary_prob"] = primary_probs
            meta_preds = fold_pipeline.meta_model_.predict(X_meta_test)

            final_preds = ((primary_preds == 1) & (meta_preds == 1)).astype(int)
            preds_series = pd.Series(final_preds, index=X_test_fold.index)

            split_predictions.append(
                {
                    "test_path_idxs": test_group_idxs,
                    "preds": preds_series,
                    "y_test": y_test_fold,
                }
            )

        logger.info("--- Constructing and Evaluating Backtest Paths ---")
        path_results = construct_backtest_paths(
            split_predictions, n_groups, k_test_groups
        )
        path_scores = _evaluate_paths_f1(path_results, mlflow_logger)

    finally:
        mlflow_logger.end_run()

    logger.info("CPCV process finished.")
    return path_scores


if __name__ == "__main__":
    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv"
    data = fetch_historical_data(
        data_path=data_path, start_date="2022-01-01T00:00:00Z", timeframe="1h"
    )
    data = adjust_data_to_ubtc(data)

    t1 = pd.Series(data.index[1:], index=data.index[:-1])
    data = data.iloc[:-1]

    run_cpcv_for_strategy(
        data=data,
        t1=t1,
        strategy_class=SmaCross,
        strategy_params={"n1": 10, "n2": 25},
        n_groups=10,
        k_test_groups=2,
        embargo_pct=0.01,
        experiment_name="CPCV_Backtest",
    )
