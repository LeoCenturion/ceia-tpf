import argparse
import logging
from typing import Any, Dict

import optuna
import pandas as pd

from src.backtesting.cpcv_runner import run_cpcv_for_ml_pipeline
from src.data_analysis.data_analysis import fetch_historical_data
from src.modeling.autogluon_adapter import AutoGluonAdapter
from src.modeling.transformers.chronos_feature_pipeline import ChronosFeaturePipeline
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class ChronosFeaturePipelineCPCV(ChronosFeaturePipeline):
    """Extends ChronosFeaturePipeline with a run_cpcv() convenience method."""

    def run_cpcv(
        self,
        raw_data: pd.DataFrame,
        model_cls: Any,
        model_params: Dict[str, Any],
        experiment_name: str,
    ) -> float:
        return run_cpcv_for_ml_pipeline(
            pipeline=self,
            raw_data=raw_data,
            model_cls=model_cls,
            model_params=model_params,
            experiment_name=experiment_name,
        )


@setup_logging
def main():
    """
    Main function to run the CPCV backtest for the ChronosFeaturePipeline with optional Optuna optimization.
    """
    parser = argparse.ArgumentParser(
        description="Run CPCV backtest for the ChronosFeaturePipeline with optional Optuna optimization."
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable Optuna hyperparameter optimization.",
    )
    args = parser.parse_args()

    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    raw_data = fetch_historical_data(
        symbol="BTC/USDT", timeframe="1m", data_path=data_path
    )

    if args.optimize:

        def objective(trial: optuna.Trial) -> float:
            try:
                chronos_model_name = trial.suggest_categorical(
                    "chronos_model_name",
                    ["amazon/chronos-t5-tiny", "amazon/chronos-t5-small"],
                )
                chronos_window_size = trial.suggest_int(
                    "chronos_window_size", 64, 256, step=64
                )
                autogluon_preset = trial.suggest_categorical(
                    "presets", ["medium_quality", "high_quality", "best_quality"]
                )

                pipeline_config = {
                    "volume_threshold": 50000,
                    "tau": 0.7,
                    "n_groups": 6,
                    "k_test_groups": 2,
                    "pct_embargo": 0.01,
                    "chronos_model_name": chronos_model_name,
                    "chronos_window_size": chronos_window_size,
                    "chronos_stride": 64,
                }

                model_params = {
                    "label": "label",
                    "eval_metric": "f1_weighted",
                    "problem_type": "binary",
                    "presets": autogluon_preset,
                    "time_limit": 600,
                    "verbosity": 1,
                    "path": f"AutogluonModels/chronos_cpcv_run_trial_{trial.number}",
                }

                pipeline = ChronosFeaturePipelineCPCV(pipeline_config)
                return pipeline.run_cpcv(
                    raw_data=raw_data,
                    model_cls=AutoGluonAdapter,
                    model_params=model_params,
                    experiment_name="Chronos_Features_CPCV_Optuna",
                )

            except Exception as e:
                logger.error(f"Trial {trial.number} failed with error: {e}")
                return 0.0

        logger.info("Starting Optuna study for Chronos CPCV...")
        study = optuna.create_study(
            direction="maximize",
            study_name="chronos_features_cpcv_optimization",
            storage="sqlite:///optuna-study.db",
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=20)

        logger.info(f"Best trial: {study.best_trial.value}")
        logger.info(f"Best params: {study.best_trial.params}")

    else:
        logger.info(
            f"Running single CPCV for Chronos strategy with {len(raw_data)} datapoints"
        )

        pipeline_config = {
            "volume_threshold": 50000,
            "tau": 0.7,
            "n_groups": 6,
            "k_test_groups": 2,
            "pct_embargo": 0.01,
            "chronos_model_name": "amazon/chronos-t5-tiny",
            "chronos_window_size": 128,
            "chronos_stride": 64,
        }

        model_params = {
            "label": "label",
            "eval_metric": "f1_weighted",
            "problem_type": "binary",
            "presets": "medium_quality",
            "time_limit": 300,
            "verbosity": 1,
            "path": "AutogluonModels/chronos_cpcv_run",
        }

        pipeline = ChronosFeaturePipelineCPCV(pipeline_config)
        pipeline.run_cpcv(
            raw_data=raw_data,
            model_cls=AutoGluonAdapter,
            model_params=model_params,
            experiment_name="Chronos_Features_CPCV",
        )


if __name__ == "__main__":
    main()
