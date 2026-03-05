import pandas as pd
import xgboost as xgb
from src.data_analysis.data_analysis import fetch_historical_data, adjust_data_to_ubtc
from src.backtesting.cpcv_runner import run_cpcv_for_strategy
from src.modeling.xgboost_pipeline_palazzo import PalazzoXGBoostPipeline
from src.utils.logging_config import setup_logging
import logging

logger = logging.getLogger(__name__)

@setup_logging
def main():
    """
    Main function to run the CPCV backtest for the PalazzoXGBoostCPCVStrategy.
    """
    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        data_path=data_path,
        start_date="2024-01-01T00:00:00Z" # Use a smaller subset for quicker testing
    )

    logger.info(f"Running CPCV for Palazzo XGBoost strategy with {len(data)} datapoints")

    # For CPCV, t1 indicates the end of an event. For time bars used as input,
    # we can assume the event ends at the next bar's timestamp. The pipeline
    # will handle the event-based (volume bar) processing internally.
    t1 = pd.Series(data.index[1:], index=data.index[:-1])
    data = data.iloc[:-1]
    t1 = t1.reindex(data.index)


    # Configuration for the pipeline
    pipeline_config = {
        "volume_threshold": 50000,
        "tau": 0.7,
        "prediction_window_size": 2000 # Number of 1m bars to use for prediction
    }

    # XGBoost model parameters
    model_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "min_child_weight": 5,
    }
    
    strategy_params={
        'pipeline_config': pipeline_config,
        'model_cls': xgb.XGBClassifier,
        'model_params': model_params
    }
   

if __name__ == '__main__':
    main()
