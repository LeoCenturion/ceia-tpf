import logging

from src.backtesting.cpcv_runner import run_cpcv_for_metalabeling_pipeline
from src.data_analysis.data_analysis import fetch_historical_data
from src.modeling.autogluon_adapter import AutoGluonAdapter
from src.modeling.chronos_metalabeling_pipeline import ChronosMetaLabelingPipeline
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


@setup_logging
def main():
    """
    Main function to run the CPCV backtest for the ChronosMetaLabelingPipeline.
    """
    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    raw_data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        data_path=data_path,
    )

    logger.info(
        f"Running CPCV for Chronos MetaLabeling strategy with {len(raw_data)} datapoints"
    )

    pipeline_config = {
        "volume_threshold": 5000,
        "tau": 0.7,
        "n_groups": 6,
        "k_test_groups": 2,
        "pct_embargo": 0.01,
        "n_splits": 3,
        "chronos_model_name": "amazon/chronos-t5-tiny",
        "chronos_window_size": 32,
        "chronos_stride": 1,
    }

    primary_model_params = {
        "label": "label",
        "eval_metric": "f1_weighted",
        "presets": "medium_quality",
        "time_limit": 60,
        "path": "AutogluonModels/tmp_cpcv_chronos_primary",
        "verbosity": 0,
    }

    meta_model_config = {
        "label": "label",
        "eval_metric": "f1",
        "presets": "best_quality",
        "time_limit": 30,
        "path": "AutogluonModels/tmp_cpcv_chronos_meta",
        "verbosity": 0,
    }

    model_params = {
        "primary_model_params": primary_model_params,
        "meta_model_config": meta_model_config,
    }

    path_scores = run_cpcv_for_metalabeling_pipeline(
        pipeline_cls=ChronosMetaLabelingPipeline,
        pipeline_config=pipeline_config,
        raw_data=raw_data,
        model_cls=AutoGluonAdapter,
        model_params=model_params,
        experiment_name="Chronos_MetaLabeling_CPCV_Backtest",
    )

    if not path_scores:
        logger.error(
            "CPCV execution resulted in no valid paths. Please check data and configuration."
        )


if __name__ == "__main__":
    main()
