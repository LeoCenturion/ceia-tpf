import os
import mlflow
import optuna
import numpy as np
import argparse
from functools import partial
from src.data_analysis.data_analysis import fetch_historical_data
from src.modeling.xgboost_pipeline_palazzo import PalazzoXGBoostPipeline
from src.modeling.autogluon_adapter import AutoGluonAdapter
from src.modeling.pipeline_runner import run_pipeline
from src.constants import VOLUME_COL, CLOSE_COL

class PalazzoAutoGluonPipeline(PalazzoXGBoostPipeline):
    """
    Pipeline that reuses PalazzoXGBoostPipeline's feature engineering
    but uses AutoGluon for prediction.
    """
    
    def log_results(self, logger, model, X_test=None, y_test=None):
        """
        Log AutoGluon specific artifacts (Leaderboard).
        """
        if hasattr(model, 'leaderboard') and X_test is not None and y_test is not None:
            print("\n--- AutoGluon Leaderboard ---")
            leaderboard_data = X_test.copy()
            leaderboard_data['label'] = y_test
            leaderboard = model.leaderboard(leaderboard_data, silent=True)
            print(leaderboard)
            
            # Log Best Model Score
            if leaderboard is not None and not leaderboard.empty:
                best_model_score = leaderboard.iloc[0]['score_test']
                best_model_name = leaderboard.iloc[0]['model']
                logger.log_metrics({"test_f1_best_model": best_model_score})
                logger.log_params({"best_model_name": best_model_name})
                
                # Optionally save leaderboard as CSV artifact
                lb_path = "autogluon_leaderboard.csv"
                leaderboard.to_csv(lb_path)
                logger.log_artifact(lb_path)
                # Cleanup local file
                if os.path.exists(lb_path):
                    os.remove(lb_path)

def objective(trial, pipeline_config, raw_data):
    """Optuna objective function for AutoGluon pipeline."""
    # Hyperparameters to tune
    presets = trial.suggest_categorical("presets", ["medium_quality", "high_quality", "best_quality"])
    time_limit = trial.suggest_int("time_limit", 300, 600, step=300)
    
    hyperparameters = {
        'FT_TRANSFORMER': {},
        'GBM': {},
        'NN_TORCH': {},
        'FASTAI': {}
    }
        
    model_params = {
        'label': 'label',
        'eval_metric': 'f1_weighted',
        'presets': presets,
        'hyperparameters': hyperparameters,
        'time_limit': time_limit,
        'verbosity': 0,
        'path': f'AutogluonModels/palazzo_optuna/trial_{trial.number}'
    }
    
    pipeline = PalazzoAutoGluonPipeline(pipeline_config)
    
    try:
        model = AutoGluonAdapter(**model_params)
        _, scores, _, _, _, _, _ = pipeline.run_cv(raw_data, model)
        
        avg_f1 = np.mean(scores)
        return avg_f1
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0

def run_optuna_study(config, raw_data, data_path, n_trials=10):
    """Sets up and runs an Optuna study for the pipeline."""
    study_name = "autogluon_palazzo_pipeline_optimization"
    storage_name = "sqlite:///optuna-study.db"
    
    # MLflow setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Ensure MLflow logs to the local DB
    mlflow.set_experiment(study_name)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )
    
    objective_with_data = partial(
        objective, 
        pipeline_config=config, 
        raw_data=raw_data
    )

    def mlflow_callback(study, trial):
        with mlflow.start_run(run_name=f"autogluon_trial_{trial.number}"):
            mlflow.log_params(trial.params)
            mlflow.log_metric("avg_f1_score", trial.value)

    study.optimize(objective_with_data, n_trials=n_trials, callbacks=[mlflow_callback])

    print("\n--- Optuna Study Best Results ---")
    try:
        best_trial = study.best_trial
        print(f"Best trial value (F1 Score): {best_trial.value:.4f}")
        print("Best parameters found:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
    except ValueError:
        print("No successful trials were completed.")

def run_single_pipeline():
    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    raw_data = fetch_historical_data(
        symbol="BTC/USDT", timeframe="1m",
        data_path=data_path
    )
    raw_data.rename(columns={VOLUME_COL: "volume", CLOSE_COL: "close"}, inplace=True)
    
    config = {
        "volume_threshold": 50000, 
        "tau": 0.7,
        "n_splits": 3, 
        "pct_embargo": 0.01,
        "use_pca": True, 
        "pca_components": 0.95
    }
    
    hyperparameters = {
        'FT_TRANSFORMER': {},
        'GBM': {},
        'NN_TORCH': {},
        'FASTAI': {}
    }
    
    # We pass the hyperparameters dict as part of model_params
    # The adapter expects them in __init__
    model_params = {
        'label': 'label',
        'eval_metric': 'f1_weighted',
        'presets': 'medium_quality',
        'hyperparameters': hyperparameters,
        'time_limit': 600,
        'verbosity': 2,
        'path': 'AutogluonModels/pipeline_run'
    }
    
    pipeline = PalazzoAutoGluonPipeline(config)
    
    run_pipeline(
        pipeline=pipeline,
        model_cls=AutoGluonAdapter,
        raw_data=raw_data,
        model_params=model_params,
        experiment_name="AutoGluon_Palazzo_Pipeline",
        data_path=data_path
    )

def main():
    parser = argparse.ArgumentParser(description="Run AutoGluon Palazzo Pipeline or Optuna study.")
    parser.add_argument('--optimize', action='store_true', help='Run Optuna hyperparameter optimization study.')
    args = parser.parse_args()

    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    raw_data = fetch_historical_data(
        symbol="BTC/USDT", timeframe="1m",
        data_path=data_path
    )
    raw_data.rename(columns={VOLUME_COL: "volume", CLOSE_COL: "close"}, inplace=True)
    
    config = {
        "volume_threshold": 50000, 
        "tau": 0.7,
        "n_splits": 3, 
        "pct_embargo": 0.01,
        "use_pca": True, 
        "pca_components": 0.95
    }

    if args.optimize:
        run_optuna_study(config, raw_data, data_path, n_trials=10)
    else:
        run_single_pipeline()

if __name__ == "__main__":
    main()
