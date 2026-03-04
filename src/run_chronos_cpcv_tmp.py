import pandas as pd
from src.data_analysis.data_analysis import fetch_historical_data, adjust_data_to_ubtc
from src.backtesting.cpcv_runner import run_cpcv_for_strategy
from src.backtesting.strategies.chronos_metalabeling_strategy import ChronosMetaLabelingCPCVStrategy
from src.modeling.autogluon_adapter import AutoGluonAdapter
from src.utils.logging_config import setup_logging
import os
import logging

@setup_logging
def main():
    """
    Main function to run the CPCV backtest for the ChronosMetaLabelingCPCVStrategy.
    """
    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv"
    data = fetch_historical_data(
        symbol="BTC/USDA",
        start_date="2024-01-01T00:00:00Z",
        timeframe="1h",
        data_path=data_path,
    )
    # Using a smaller subset for quicker testing
    data = adjust_data_to_ubtc(data)
    logging.info(f"Running cpcv for Chronos metalabeling strategy with {len(data)} datapoints")
    # For CPCV, we need t1, which indicates the end of an event.
    # For a simple time-bar based strategy, we can assume the event ends at the next bar.
    t1 = pd.Series(data.index[1:], index=data.index[:-1])
    data = data.iloc[:-1]
    t1 = t1.reindex(data.index)


    # Configuration for the pipeline
    pipeline_config = {
        "volume_threshold": 50000,
        "tau": 0.7,
        "n_splits": 3,
        "pct_embargo": 0.01,
        "use_pca": False,
        "chronos_model_name": "amazon/chronos-t5-tiny",
        "chronos_window_size": 128,
    }

    # AutoGluon parameters
    primary_model_params = {
        'label': 'label',
        'eval_metric': 'f1_weighted',
        'presets': 'medium_quality',
        'time_limit': 60,
        'path': 'AutogluonModels/tmp_cpcv_chronos_primary',
        'verbosity': 0
    }

    meta_model_config = {
        'label': 'label',
        'eval_metric': 'f1',
        'presets': 'medium_quality',
        'time_limit': 30,
        'path': 'AutogluonModels/tmp_cpcv_chronos_meta',
        'verbosity': 0
    }
    
    model_params = {
        "primary_model_params": primary_model_params,
        "meta_model_config": meta_model_config,
    }
    
    strategy_params={
        'pipeline_config': pipeline_config,
        'model_cls': AutoGluonAdapter,
        'model_params': model_params
    }

    run_cpcv_for_strategy(
        data=data,
        t1=t1,
        strategy_class=ChronosMetaLabelingCPCVStrategy,
        strategy_params=strategy_params,
        n_groups=3,
        k_test_groups=1,
        embargo_pct=0.01,
        experiment_name="tmp_CPCV_Chronos_Test"
    )

if __name__ == '__main__':
    main()
