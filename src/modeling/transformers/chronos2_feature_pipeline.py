import argparse
import logging
from functools import partial

import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
from chronos import Chronos2Pipeline

from src.data_analysis.data_analysis import fetch_historical_data, timer
from src.modeling.autogluon_adapter import AutoGluonAdapter
from src.modeling.pipeline_runner import run_pipeline
from src.modeling.xgboost_pipeline_palazzo import PalazzoXGBoostPipeline

logger = logging.getLogger(__name__)

_DEFAULT_CHRONOS2_MODEL = "autogluon/chronos-2-small"


class Chronos2FeaturePipeline(PalazzoXGBoostPipeline):
    """
    Extends PalazzoXGBoostPipeline with Chronos-2 patch embeddings added to
    the tabular feature set.  The downstream model is XGBoost (or any model
    passed to run_cv), same as the parent pipeline.
    """

    def __init__(self, config):
        super().__init__(config)
        self.problem_type = "classification"
        self.chronos2_pipeline = None

    @timer
    def step_2_feature_engineering(self, bars):
        logger.debug("Step 2: Generating tabular + Chronos-2 embedding features...")

        tabular_features = super().step_2_feature_engineering(bars)
        logger.debug(f"Tabular features shape: {tabular_features.shape}")

        common_index = tabular_features.index.intersection(bars.index)
        bars_aligned = bars.loc[common_index]

        if self.chronos2_pipeline is None:
            model_name = self.config.get("chronos_model_name", _DEFAULT_CHRONOS2_MODEL)
            self.chronos2_pipeline = Chronos2Pipeline.from_pretrained(
                model_name,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.bfloat16,
            )

        window_size = self.config.get("chronos_window_size", 128)
        stride = self.config.get("chronos_stride", 1)

        chronos2_embeddings = []
        embedding_dim = None

        for i in range(0, len(bars_aligned) - window_size + 1, stride):
            window = bars_aligned.iloc[i : i + window_size]
            # Chronos-2 embed requires shape (n_series, n_variates, history_length)
            ts = torch.tensor(window["close_price"].values.astype(np.float32)).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                embeddings_list, _ = self.chronos2_pipeline.embed(ts)

            # embeddings_list[0]: (n_variates=1, num_patches+2, d_model) → mean-pool patches → (d_model,)
            embedding = embeddings_list[0].mean(dim=1).squeeze(0).float().cpu().numpy()
            if embedding_dim is None:
                embedding_dim = embedding.shape[-1]
            chronos2_embeddings.append(embedding)

        if chronos2_embeddings:
            chronos2_df = pd.DataFrame(
                np.array(chronos2_embeddings),
                index=bars_aligned.index[window_size - 1 :: stride],
                columns=[f"chronos2_embed_{j}" for j in range(embedding_dim)],
            )
        else:
            chronos2_df = pd.DataFrame(index=pd.Index([]))

        logger.debug(f"Chronos-2 embedding features shape: {chronos2_df.shape}")

        final_index = tabular_features.index.intersection(chronos2_df.index)
        combined = pd.concat(
            [tabular_features.loc[final_index], chronos2_df.loc[final_index]], axis=1
        )
        features = combined.dropna()
        logger.debug(f"Combined features shape: {features.shape}")
        return features


def objective(trial, raw_data):
    """Optuna objective function for Chronos-2 Feature pipeline."""
    chronos_window_size = trial.suggest_int("chronos_window_size", 32, 256, step=64)
    chronos_model_name = trial.suggest_categorical(
        "chronos_model_name",
        [
            "autogluon/chronos-2-tiny",
            "autogluon/chronos-2-mini",
            "autogluon/chronos-2-small",
        ],
    )

    pipeline_config = {
        "volume_threshold": 50000,
        "tau": 0.7,
        "n_splits": 3,
        "pct_embargo": 0.01,
        "use_pca": False,
        "chronos_model_name": chronos_model_name,
        "chronos_window_size": chronos_window_size,
        "chronos_stride": 1,
    }

    presets = trial.suggest_categorical("presets", ["medium_quality", "high_quality"])
    time_limit = trial.suggest_int("time_limit", 300, 600, step=300)

    model_params = {
        "label": "label",
        "eval_metric": "f1_weighted",
        "presets": presets,
        "time_limit": time_limit,
        "verbosity": 0,
        "path": f"AutogluonModels/chronos2_optuna/trial_{trial.number}",
    }

    pipeline = Chronos2FeaturePipeline(pipeline_config)

    try:
        model = AutoGluonAdapter(**model_params)
        _, scores, _, _, _, _, _ = pipeline.run_cv(raw_data, model)
        avg_f1 = np.mean(scores)
        return avg_f1
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return 0.0


def run_optuna_study(raw_data, data_path, n_trials=10):
    study_name = "chronos2_feature_pipeline_optimization"
    storage_name = "sqlite:///optuna-study.db"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(study_name)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )

    objective_with_data = partial(objective, raw_data=raw_data)

    def mlflow_callback(study, trial):
        with mlflow.start_run(run_name=f"chronos2_trial_{trial.number}"):
            mlflow.log_params(trial.params)
            mlflow.log_metric("avg_f1_score", trial.value)

    study.optimize(objective_with_data, n_trials=n_trials, callbacks=[mlflow_callback])

    print("--- Optuna Study Best Results ---")
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
        symbol="BTC/USDT",
        timeframe="1m",
        data_path=data_path,
    )
    logger.debug(f"Initial raw_data size: {len(raw_data)}")

    pipeline_config = {
        "volume_threshold": 50000,
        "tau": 0.7,
        "n_splits": 3,
        "pct_embargo": 0.01,
        "use_pca": False,
        "chronos_model_name": _DEFAULT_CHRONOS2_MODEL,
        "chronos_window_size": 64,
        "chronos_stride": 1,
    }

    model_params = {
        "label": "label",
        "eval_metric": "f1_weighted",
        "presets": "medium_quality",
        "time_limit": 600,
        "verbosity": 1,
        "path": "AutogluonModels/chronos2_feature_pipeline_run",
    }

    pipeline = Chronos2FeaturePipeline(pipeline_config)
    run_pipeline(
        pipeline=pipeline,
        model_cls=AutoGluonAdapter,
        raw_data=raw_data,
        model_params=model_params,
        experiment_name="Chronos2_Feature_AutoGluon_Pipeline",
        data_path=data_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run Chronos-2 Feature Pipeline or Optuna study."
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Optuna hyperparameter optimization study.",
    )
    args = parser.parse_args()

    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    raw_data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        data_path=data_path,
    )

    if args.optimize:
        run_optuna_study(raw_data, data_path, n_trials=10)
    else:
        run_single_pipeline()


if __name__ == "__main__":
    main()
