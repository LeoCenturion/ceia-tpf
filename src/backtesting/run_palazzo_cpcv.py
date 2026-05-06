import argparse
import copy
import logging

import optuna
import xgboost as xgb

from src.backtesting.cpcv_runner import run_cpcv_for_ml_pipeline
from src.data_analysis.data_analysis import fetch_historical_data
from src.modeling.autogluon_adapter import AutoGluonAdapter
from src.modeling.machine_learning.xgboost_pipeline_palazzo import PalazzoXGBoostPipeline
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


@setup_logging
def main():
    """
    Main function to run the CPCV backtest for the Palazzo pipeline with XGBoost or AutoGluon.
    """
    parser = argparse.ArgumentParser(
        description="Run CPCV backtest for the Palazzo pipeline."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["xgboost", "autogluon"],
        help="The model to run the backtest with.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable Optuna hyperparameter optimization for AutoGluon.",
    )
    args = parser.parse_args()

    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        data_path=data_path,
    )

    logger.info(
        f"Running CPCV for Palazzo strategy with {len(data)} datapoints using {args.model}"
    )

    pipeline_config = {
        "volume_threshold": 50000,
        "tau": 0.7,
        "n_groups": 10,
        "k_test_groups": 2,
        "pct_embargo": 0.01,
    }
    pipeline = PalazzoXGBoostPipeline(pipeline_config)

    if args.model == "xgboost":
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
        run_cpcv_for_ml_pipeline(
            pipeline=pipeline,
            raw_data=data,
            model_cls=xgb.XGBClassifier,
            model_params=model_params,
            experiment_name="Palazzo_CPCV_Backtest",
        )

    elif args.model == "autogluon":
        if args.optimize:

            def autogluon_objective(trial: optuna.Trial) -> float:
                try:
                    base_model_params = {
                        "label": "label",
                        "eval_metric": "f1_weighted",
                        "hyperparameters": {
                            "FT_TRANSFORMER": {},
                            "GBM": {},
                            "NN_TORCH": {},
                            "FASTAI": {},
                        },
                        "time_limit": 600,
                        "verbosity": 1,
                    }

                    presets = trial.suggest_categorical(
                        "presets", ["medium_quality", "high_quality", "best_quality"]
                    )

                    model_params = copy.deepcopy(base_model_params)
                    model_params["presets"] = presets
                    model_params["path"] = (
                        f"AutogluonModels/palazzo_cpcv_autogluon_trial_{trial.number}"
                    )

                    return run_cpcv_for_ml_pipeline(
                        pipeline=pipeline,
                        raw_data=data,
                        model_cls=AutoGluonAdapter,
                        model_params=model_params,
                        experiment_name="Palazzo_CPCV_AutoGluon_Optuna",
                    )
                except Exception as e:
                    logger.warning(f"Trial {trial.number} failed with error: {e}")
                    return 0.0

            study = optuna.create_study(
                direction="maximize",
                study_name="palazzo_autogluon_cpcv_optimization",
                storage="sqlite:///optuna-study.db",
                load_if_exists=True,
            )
            study.optimize(autogluon_objective, n_trials=10)
            logger.info(f"Best trial: {study.best_trial.value}")
            logger.info(f"Best params: {study.best_trial.params}")
        else:
            logger.info("Running single AutoGluon CPCV without Optuna optimization.")
            model_params = {
                "label": "label",
                "eval_metric": "f1_weighted",
                "problem_type": "binary",
                "presets": "medium_quality",
                "time_limit": 600,
                "verbosity": 1,
                "path": "AutogluonModels/palazzo_cpcv_autogluon_run_single",
            }
            run_cpcv_for_ml_pipeline(
                pipeline=pipeline,
                raw_data=data,
                model_cls=AutoGluonAdapter,
                model_params=model_params,
                experiment_name="Palazzo_CPCV_AutoGluon_Single_Run",
            )


if __name__ == "__main__":
    main()
