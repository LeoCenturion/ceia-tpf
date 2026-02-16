import mlflow
import os
import pandas as pd
from datetime import datetime

class MLflowLogger:
    def __init__(self, experiment_name, tracking_uri=None):
        """
        Initialize MLflow logger.
        """
        if tracking_uri is None:
            # Default to a local sqlite database in the project root
            tracking_uri = "sqlite:///mlflow.db"
            
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = None

    def start_run(self, run_name=None):
        """Start a new MLflow run."""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run = mlflow.start_run(run_name=run_name)
        return self.run

    def log_params(self, params, prefix=None):
        """
        Log a dictionary of parameters. 
        Flattens nested dictionaries if necessary or logs them as strings.
        """
        for k, v in params.items():
            key_name = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                # Recursively log nested dicts or convert to string if too deep
                # For simplicity in this utility, we'll converting dicts to string representations 
                # for complex hyperparams to avoid cluttering MLflow params with too many nested keys
                # unless simple.
                self.log_params(v, prefix=key_name) 
            else:
                mlflow.log_param(key_name, v)

    def log_metrics(self, metrics, step=None):
        """Log a dictionary of metrics."""
        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=step)

    def log_data_info(self, data: pd.DataFrame, data_path: str):
        """Log metadata about the input dataset."""
        filename = os.path.basename(data_path)
        start_date = "N/A"
        end_date = "N/A"
        
        if isinstance(data.index, pd.DatetimeIndex):
            start_date = data.index.min().isoformat()
            end_date = data.index.max().isoformat()
        
        info = {
            "data_filename": filename,
            "data_start_date": start_date,
            "data_end_date": end_date,
            "data_rows": data.shape[0],
            "data_cols": data.shape[1]
        }
        self.log_params(info)

    def log_artifact(self, local_path):
        """Log a local file as an artifact."""
        mlflow.log_artifact(local_path)

    def end_run(self):
        """End the current run."""
        mlflow.end_run()
