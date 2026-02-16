import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report
from src.modeling.mlflow_utils import MLflowLogger


def run_pipeline(
    pipeline,
    model_cls,
    raw_data,
    model_params,
    experiment_name,
    data_path=None,
    test_size=0.3,
):
    """
    Generic function to run an ML pipeline, including:
    1. MLflow tracking setup
    2. Pipeline execution (Purged CV)
    3. Final Train/Test split evaluation
    4. Logging of parameters, metrics, and artifacts

    Args:
        pipeline: Instance of the pipeline (subclass of AbstractMLPipeline)
        model_cls: Class of the model (e.g., xgboost.XGBClassifier)
        raw_data: Input DataFrame
        model_params: Model hyperparameters dictionary
        experiment_name: MLflow experiment name
        data_path: Optional path to data file for logging
        test_size: Size of the test split for final evaluation (default 0.3)
    """

    # 1. Setup MLflow
    logger = MLflowLogger(experiment_name=experiment_name)
    logger.start_run()

    try:
        # Log Data Info
        if data_path:
            logger.log_data_info(raw_data, data_path)
        else:
            # Fallback if path not provided
            logger.log_params(
                {
                    "data_rows": raw_data.shape[0],
                    "data_cols": raw_data.shape[1],
                    "data_start_date": raw_data.index.min().isoformat()
                    if isinstance(raw_data.index, pd.DatetimeIndex)
                    else "N/A",
                    "data_end_date": raw_data.index.max().isoformat()
                    if isinstance(raw_data.index, pd.DatetimeIndex)
                    else "N/A",
                }
            )

        # 2. Instantiate Model
        model = model_cls(**model_params)

        # Log Configuration via Pipeline Hook
        # Pipeline is already instantiated, so we just call the hook
        pipeline.log_config(logger)
        logger.log_params(model_params, prefix="model_params")
        logger.log_params({"model_class": model_cls.__name__})

        # 3. Run Pipeline (Purged CV)
        print(
            f"Running pipeline: {pipeline.__class__.__name__} with {model_cls.__name__}"
        )
        trained_model, scores, X, y, sw, t1, pca = pipeline.run_cv(raw_data, model)

        # Log CV Metrics
        avg_cv_f1 = np.mean(scores)
        std_cv_f1 = np.std(scores)
        print(f"\nAverage Purged CV F1 Score: {avg_cv_f1:.4f} (+/- {std_cv_f1:.4f})")

        logger.log_metrics({"avg_cv_f1": avg_cv_f1, "std_cv_f1": std_cv_f1})
        for i, score in enumerate(scores):
            logger.log_metrics({f"fold_{i + 1}_f1": score})

        # 4. Final Train/Test Split Evaluation
        print("\nFinal Classification Report (Sample Split):")

        # Simple time-series split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # We need sample weights for the split if available, aligned with indices
        # sw is a Series with same index as X/y
        sw_train = sw.iloc[:split_idx] if sw is not None else None

        # Train a fresh model on the train split
        # We use clone to ensure it's a fresh instance with same params
        model_report = clone(model)

        # Fit
        if sw_train is not None:
            model_report.fit(X_train, y_train, sample_weight=sw_train.values)
        else:
            model_report.fit(X_train, y_train)

        # Predict
        y_pred = model_report.predict(X_test)

        # Report
        report_str = classification_report(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        print(report_str)

        # Log Test Metrics
        logger.log_metrics(
            {
                "test_accuracy": report_dict["accuracy"],
                "test_macro_f1": report_dict["macro avg"]["f1-score"],
                "test_weighted_f1": report_dict["weighted avg"]["f1-score"],
            }
        )

        # Log class-specific metrics
        for label, metrics in report_dict.items():
            if label not in ["accuracy", "macro avg", "weighted avg"]:
                logger.log_metrics(
                    {
                        f"class_{label}_precision": metrics["precision"],
                        f"class_{label}_recall": metrics["recall"],
                        f"class_{label}_f1": metrics["f1-score"],
                    }
                )

        # Allow pipeline to log custom results (e.g. AutoGluon leaderboard)
        pipeline.log_results(logger, model_report, X_test, y_test)

        return trained_model, scores, X, y

    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        raise
    finally:
        logger.end_run()
