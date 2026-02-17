import numpy as np
import pandas as pd
import json # Added this import
from sklearn.base import clone
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
from src.modeling.mlflow_utils import MLflowLogger

def run_pipeline(
    pipeline,
    model_cls,
    raw_data,
    model_params,
    experiment_name,
    data_path=None,
    test_size=0.3
):
    """
    Generic function to run an ML pipeline, including:
    1. MLflow tracking setup
    2. Pipeline execution (Purged CV or Time-Series Split)
    3. Final evaluation based on problem_type
    4. Logging of parameters, metrics, and artifacts
    """
    
    # 1. Setup MLflow
    logger = MLflowLogger(experiment_name=experiment_name)
    logger.start_run()
    
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
        print(f"Running pipeline: {pipeline.__class__.__name__} ({pipeline.problem_type})")
        
        # This is a bit of a hack to handle the meta-labeling pipeline's different signature
        if "MetaLabeling" in pipeline.__class__.__name__:
            trained_primary_model, trained_meta_model, metrics, X_test, y_test, t1_test, primary_test_pred, final_decision = pipeline.run(raw_data, model_cls, model_params)
            # For meta-labeling, we log results differently
            pipeline.log_results(logger, trained_primary_model, trained_meta_model, metrics, X_test, y_test, t1_test, primary_test_pred, final_decision)
            return trained_primary_model, trained_meta_model, metrics
        else:
            model = model_cls(**model_params) if model_cls else None
            trained_model, scores, X, y, sw, t1, pca = pipeline.run_cv(raw_data, model)

            # Log CV Metrics
            avg_score = np.mean(scores)
            metric_name = "avg_cv_f1" if pipeline.problem_type == "classification" else "avg_cv_mase"
            print(f"\nAverage CV Score: {avg_score:.4f}")
            logger.log_metrics({metric_name: avg_score})

            # 4. Final Evaluation
            if pipeline.problem_type == "classification" and model is not None:
                print("\nFinal Classification Report (Sample Split):")
                split_idx = int(len(X) * (1 - test_size))
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                model_report = clone(model)
                model_report.fit(X_train, y_train, sample_weight=sw.iloc[:split_idx].values if sw is not None else None)
                y_pred = model_report.predict(X_test)
                
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                print(classification_report(y_test, y_pred))
                
                logger.log_metrics({
                    "test_accuracy": report_dict['accuracy'],
                    "test_macro_f1": report_dict['macro avg']['f1-score'],
                    "test_weighted_f1": report_dict['weighted avg']['f1-score']
                })
                logger.log_artifact_dict(report_dict, "classification_report.json") # Added this line
                pipeline.log_results(logger, model_report, X_test, y_test)
            else:
                pipeline.log_results(logger, trained_model, X, y)

            return trained_model, scores, X, y

    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        raise
    finally:
        logger.end_run()
