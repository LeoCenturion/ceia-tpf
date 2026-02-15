import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# Make the script runnable from anywhere
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_analysis.data_analysis import fetch_historical_data
from src.modeling.xgboost_pipeline_palazzo import PalazzoXGBoostPipeline
from src.modeling.autogluon_adapter import AutoGluonAdapter
from src.constants import VOLUME_COL, CLOSE_COL

class PalazzoAutoGluonPipeline(PalazzoXGBoostPipeline):
    """
    Pipeline that reuses PalazzoXGBoostPipeline's feature engineering
    but uses AutoGluon for prediction.
    """
    pass  # Logic is identical, just need a different model in run_cv

def main():
    # Load data
    raw_data = fetch_historical_data(
        symbol="BTC/USDT", timeframe="1m",
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    )
    raw_data.rename(columns={VOLUME_COL: "volume", CLOSE_COL: "close"}, inplace=True)
    
    # Configuration
    config = {
        "volume_threshold": 50000, 
        "tau": 0.7,
        "n_splits": 3, 
        "pct_embargo": 0.01,
        "use_pca": True, 
        "pca_components": 0.95
    }
    
    # Initialize Pipeline
    pipeline = PalazzoAutoGluonPipeline(config)
    
    # Initialize AutoGluon Adapter
    hyperparameters = {
        'FT_TRANSFORMER': {},
        'GBM': {},
        'NN_TORCH': {},
        'FASTAI': {}
    }
    
    adapter = AutoGluonAdapter(
        label='label',
        eval_metric='f1_weighted',
        presets='medium_quality',
        hyperparameters=hyperparameters,
        time_limit=600, # Increased to allow time for multiple models
        verbosity=2,
        path='AutogluonModels/standalone_pipeline'
    )
    
    print("Running Pipeline with AutoGluon...")
    trained_model, scores, X, y, sw, t1, pca = pipeline.run_cv(raw_data, adapter)
    
    print(f"\nAverage Purged CV F1 Score: {np.mean(scores):.4f}")
    
    print("\nFinal Classification Report (Sample Split):")
    # Simple final split for report
    split = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Fit a fresh model for the report to mimic the original script's behavior
    # (The pipeline returns a model fitted on ALL data, but for this specific report 
    # we want to see Test performance on unseen data)
    model_report = AutoGluonAdapter(
        label='label',
        eval_metric='f1_weighted',
        presets='medium_quality',
        hyperparameters=hyperparameters,
        time_limit=600,
        verbosity=2,
        path='AutogluonModels/standalone_report'
    )
    
    model_report.fit(X_train, y_train, sample_weight=None) # sw is complex to split aligned, skipping for simple report
    
    print("\n--- AutoGluon Leaderboard ---")
    # Pass the test data to evaluate performance on unseen data in the leaderboard
    # We need to construct a test dataframe with labels for the leaderboard to calculate scores
    leaderboard_data = X_test.copy()
    leaderboard_data['label'] = y_test
    print(model_report.leaderboard(leaderboard_data))
    
    y_pred = model_report.predict(X_test)
    
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
