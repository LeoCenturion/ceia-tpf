import os
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

def main():
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

if __name__ == "__main__":
    main()
