import logging
import os
import tempfile

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    classification_report,
)

from src.modeling.mlflow_utils import MLflowLogger


def _log_oos_predictions(oos_df: pd.DataFrame) -> None:
    """Serialize OOS predictions DataFrame to CSV and log under predictions/ artifact path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="oos_predictions_"
    ) as f:
        oos_df.to_csv(f, index=True)
        tmp = f.name
    try:
        mlflow.log_artifact(tmp, artifact_path="predictions")
    finally:
        os.remove(tmp)


class _ReplayTrial:
    """Replays Optuna best_params so get_optuna_params can reconstruct the pipeline/model split."""
    number = -1  # sentinel so callers that use trial.number get a safe value

    def __init__(self, params: dict):
        self._params = params
    def suggest_int(self, name, *a, **kw): return self._params[name]
    def suggest_float(self, name, *a, **kw): return self._params[name]
    def suggest_categorical(self, name, *a, **kw): return self._params[name]


def run_pipeline(
    pipeline,
    model_cls,
    raw_data,
    model_params,
    experiment_name,
    data_path=None,
    test_size=0.3,
    nested=False,
    run_name=None,
    parent_run_id=None,
    tracking_uri=None,
):
    """
    Generic function to run an ML pipeline, including:
    1. MLflow tracking setup
    2. Pipeline execution (Purged CV or Time-Series Split)
    3. Final evaluation based on problem_type
    4. Logging of parameters, metrics, and artifacts
    """

    # 1. Setup MLflow
    logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=tracking_uri)
    logger.start_run(run_name=run_name, nested=nested, parent_run_id=parent_run_id)

    try:
        # Log Data Info
        if data_path:
            logger.log_data_info(raw_data, data_path)

        # 2. Log Configuration via Pipeline Hook
        pipeline.log_config(logger)
        if model_params:
            logger.log_params(model_params, prefix="model_params")
        if model_cls:
            logger.log_params({"model_class": model_cls.__name__})

        # 3. Run Pipeline
        logging.debug(
            f"Running pipeline: {pipeline.__class__.__name__} ({pipeline.problem_type})"
        )

        # This is a bit of a hack to handle the meta-labeling pipeline's different signature
        if "MetaLabeling" in pipeline.__class__.__name__:
            (
                trained_primary_model,
                trained_meta_model,
                metrics,
                X_test,
                y_test,
                t1_test,
                primary_test_pred,
                final_decision,
            ) = pipeline.run(raw_data, model_cls, model_params)
            # For meta-labeling, we log results differently
            pipeline.log_results(
                logger,
                trained_primary_model,
                trained_meta_model,
                metrics,
                X_test,
                y_test,
                t1_test,
                primary_test_pred,
                final_decision,
            )
            return trained_primary_model, trained_meta_model, metrics
        else:
            model = model_cls(**model_params) if model_cls else None
            trained_model, scores, X, y, sw, t1, pca, oos_df = pipeline.run_cv(raw_data, model)

            if oos_df is not None and not oos_df.empty:
                _log_oos_predictions(oos_df)

            # Log CV Metrics
            avg_score = np.mean(scores)
            metric_name = {
                "classification": "avg_cv_f1",
                "trading": "avg_cv_sharpe",
            }.get(pipeline.problem_type, "avg_cv_score")
            logging.debug(f"\nAverage CV Score: {avg_score:.4f}")
            logger.log_metrics({metric_name: avg_score})

            # 4. Final Evaluation
            if pipeline.problem_type == "classification" and model is not None:
                logging.debug("\nFinal Classification Report (Sample Split):")
                split_idx = int(len(X) * (1 - test_size))
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

                model_report = clone(model)
                model_report.fit(
                    X_train,
                    y_train,
                    sample_weight=sw.iloc[:split_idx].values
                    if sw is not None
                    else None,
                )
                y_pred = model_report.predict(X_test)

                report_dict = classification_report(y_test, y_pred, output_dict=True)
                logging.debug(classification_report(y_test, y_pred))

                logger.log_metrics(
                    {
                        "test_accuracy": report_dict["accuracy"],
                        "test_macro_f1": report_dict["macro avg"]["f1-score"],
                        "test_weighted_f1": report_dict["weighted avg"]["f1-score"],
                    }
                )
                logger.log_artifact_dict(
                    report_dict, "classification_report.json"
                )  # Added this line
                pipeline.log_results(logger, model_report, X_test, y_test)
            else:
                pipeline.log_results(logger, trained_model, X, y)

            return trained_model, scores, X, y

    except Exception as e:
        logging.debug(f"Pipeline execution failed: {e}")
        raise
    finally:
        logger.end_run()


def run_optuna_optimization(
    pipeline_cls,
    model_cls,
    raw_data,
    pipeline_config: dict,
    experiment_name: str,
    n_trials: int = 30,
    n_jobs: int = 1,
    run_name_prefix: str = None,
    optuna_storage: str = "sqlite:///optuna-study.db",
    tracking_uri: str = "sqlite:///mlflow.db",
    best_metric_name: str = "best_avg_cv_f1",
    data_path: str = None,
) -> tuple[dict, dict, float]:
    """
    Generic Optuna optimization loop shared across all pipeline types.

    Each trial merges pipeline_config with pipeline_cls.get_optuna_params(trial), then
    runs run_pipeline nested under a parent MLflow run.  After the study, _ReplayTrial
    reconstructs the pipeline/model param split from best_params and executes a final
    canonical run logged to {experiment_name}_best.

    Returns (best_pipeline_overrides, best_model_params, best_value).
    """
    prefix = run_name_prefix or pipeline_cls.__name__
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment_name)
    if exp is not None and exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{prefix}_optuna") as parent_run:
        parent_run_id = parent_run.info.run_id
        mlflow.log_param("pipeline_class", pipeline_cls.__name__)
        mlflow.log_param("model_class", model_cls.__name__ if model_cls else "None")
        mlflow.log_param("n_trials", n_trials)
        for k, v in pipeline_config.items():
            try:
                mlflow.log_param(f"pipeline.{k}", v)
            except Exception:
                pass

        def objective(trial):
            pipeline_trial_params = pipeline_cls.get_optuna_params(trial)
            model_params = model_cls.get_optuna_params(trial) if model_cls else {}
            merged_config = {**pipeline_config, **pipeline_trial_params}
            try:
                _, scores, _, _ = run_pipeline(
                    pipeline=pipeline_cls(merged_config),
                    model_cls=model_cls,
                    raw_data=raw_data,
                    model_params=model_params,
                    experiment_name=experiment_name,
                    data_path=data_path,
                    nested=True,
                    run_name=f"trial_{trial.number}",
                    parent_run_id=parent_run_id,
                    tracking_uri=tracking_uri,
                )
                return float(np.nanmean(scores)) if scores else 0.0
            except Exception as exc:
                logging.warning("Trial %d failed: %s", trial.number, exc)
                return 0.0

        study_name = f"{experiment_name}_{pipeline_cls.__name__}"
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=optuna_storage,
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

        try:
            best_params = study.best_params
            best_value = study.best_value
        except ValueError:
            logging.warning("No successful trials for %s.", pipeline_cls.__name__)
            return {}, {}, 0.0

        # _ReplayTrial re-runs get_optuna_params with fixed best values so the
        # conditional branching in each method follows the same path as the best trial.
        replay = _ReplayTrial(best_params)
        best_pipeline_overrides = pipeline_cls.get_optuna_params(replay)
        best_model_params = model_cls.get_optuna_params(replay) if model_cls else {}

        mlflow.log_params({f"best.{k}": v for k, v in best_params.items()})
        mlflow.log_metric(best_metric_name, best_value)

    logging.debug(
        "%s + %s — best %s: %.4f",
        pipeline_cls.__name__, model_cls.__name__ if model_cls else "None", best_metric_name, best_value,
    )

    run_pipeline(
        pipeline=pipeline_cls({**pipeline_config, **best_pipeline_overrides}),
        model_cls=model_cls,
        raw_data=raw_data,
        model_params=best_model_params,
        experiment_name=f"{experiment_name}_best",
        data_path=data_path,
        tracking_uri=tracking_uri,
    )

    return best_pipeline_overrides, best_model_params, best_value
