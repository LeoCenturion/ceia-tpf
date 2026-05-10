import argparse
import logging

import numpy as np
import pandas as pd
import torch
from chronos import ChronosBoltPipeline

from src.data_analysis.data_analysis import fetch_historical_data, timer
from src.modeling.autogluon_adapter import AutoGluonAdapter
from src.modeling.pipeline_runner import run_optuna_optimization, run_pipeline
from src.modeling.machine_learning.xgboost_pipeline_palazzo import PalazzoXGBoostPipeline

logger = logging.getLogger(__name__)

_DEFAULT_BOLT_MODEL = "autogluon/chronos-bolt-tiny"


class ChronosBoltFeaturePipeline(PalazzoXGBoostPipeline):
    """
    Extends PalazzoXGBoostPipeline with Chronos-Bolt patch embeddings added to
    the tabular feature set.  The downstream model is XGBoost (or any model
    passed to run_cv), same as the parent pipeline.
    """

    def __init__(self, config):
        super().__init__(config)
        self.problem_type = "classification"
        self.bolt_pipeline = None

    @timer
    def step_2_feature_engineering(self, bars):
        logger.debug("Step 2: Generating tabular + Chronos-Bolt embedding features...")

        tabular_features = super().step_2_feature_engineering(bars)
        logger.debug(f"Tabular features shape: {tabular_features.shape}")

        common_index = tabular_features.index.intersection(bars.index)
        bars_aligned = bars.loc[common_index]

        if self.bolt_pipeline is None:
            model_name = self.config.get("chronos_model_name", _DEFAULT_BOLT_MODEL)
            self.bolt_pipeline = ChronosBoltPipeline.from_pretrained(
                model_name,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float32,
            )

        window_size = self.config.get("chronos_window_size", 128)
        stride = self.config.get("chronos_stride", 1)

        bolt_embeddings = []
        embedding_dim = None

        for i in range(0, len(bars_aligned) - window_size + 1, stride):
            window = bars_aligned.iloc[i : i + window_size]
            ts = torch.tensor(window["close_price"].values.astype(np.float32))

            with torch.no_grad():
                embeddings, _ = self.bolt_pipeline.embed(ts)

            # mean-pool over patch dimension → (d_model,)
            embedding = embeddings.mean(dim=1).squeeze(0).float().cpu().numpy()
            if embedding_dim is None:
                embedding_dim = embedding.shape[-1]
            bolt_embeddings.append(embedding)

        if bolt_embeddings:
            bolt_df = pd.DataFrame(
                np.array(bolt_embeddings),
                index=bars_aligned.index[window_size - 1 :: stride],
                columns=[f"bolt_embed_{j}" for j in range(embedding_dim)],
            )
        else:
            bolt_df = pd.DataFrame(index=pd.Index([]))

        logger.debug(f"Bolt embedding features shape: {bolt_df.shape}")

        final_index = tabular_features.index.intersection(bolt_df.index)
        combined = pd.concat(
            [tabular_features.loc[final_index], bolt_df.loc[final_index]], axis=1
        )
        features = combined.dropna()
        logger.debug(f"Combined features shape: {features.shape}")
        return features

    @classmethod
    def get_optuna_params(cls, trial) -> dict:
        return {
            "chronos_model_name": trial.suggest_categorical(
                "chronos_model_name",
                [
                    "amazon/chronos-bolt-tiny",
                    "amazon/chronos-bolt-mini",
                    "amazon/chronos-bolt-small",
                ],
            ),
            "chronos_window_size": trial.suggest_int("chronos_window_size", 32, 256, step=64),
        }


# def main():
#     data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
#     raw_data = fetch_historical_data(
#         symbol="BTC/USDT", timeframe="1m", data_path=data_path
#     )

#     config = {
#         "volume_threshold": 50000,
#         "tau": 0.7,
#         "n_splits": 3,
#         "pct_embargo": 0.01,
#         "use_pca": False,
#         "chronos_model_name": _DEFAULT_BOLT_MODEL,
#         "chronos_window_size": 32,
#         "chronos_stride": 1,
#     }

#     model_params = {
#         "label": "label",
#         "eval_metric": "f1_weighted",
#         "presets": "medium_quality",  # Commented out for 'best_quality' preset
#         # "presets": "best_quality",  # Using 'best_quality' as the highest known preset, 'extreme' is not a recognized preset.
#         "time_limit": 600,
#         "verbosity": 1,
#         "path": "AutogluonModels/chronos_bolt_feature_pipeline_run",
#     }

#     pipeline = ChronosBoltFeaturePipeline(config)
#     run_pipeline(
#         pipeline=pipeline,
#         model_cls=AutoGluonAdapter,  # The final model to train on Chronos features
#         raw_data=raw_data,
#         model_params=model_params,
#         experiment_name="Chronos_Bolt_Feature_AutoGluon_Pipeline",
#         data_path=data_path,
#     )


def run_single_pipeline():
    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    raw_data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        data_path=data_path,
    )
    logger.debug(f"Initial raw_data size: {len(raw_data)}")
    # raw_data.rename(columns={VOLUME_COL: "volume", CLOSE_COL: "close"}, inplace=True)

    # Configuration for the ChronosFeaturePipeline
    pipeline_config = {
        "volume_threshold": 50000,
        "tau": 0.7,
        "n_splits": 3,
        "pct_embargo": 0.01,
        "use_pca": False,  # PCA might be redundant with Chronos embeddings, can be experimented with
        "chronos_model_name": "amazon/chronos-bolt-small",
        "chronos_window_size": 64,
        "chronos_stride": 1,
    }

    model_params = {
        "label": "label",
        "eval_metric": "f1_weighted",
        "presets": "best_quality",  # Commented out for 'best_quality' preset
        # "presets": "best_quality",  # Using 'best_quality' as the highest known preset, 'extreme' is not a recognized preset.
        "time_limit": 600,
        "verbosity": 1,
        "path": "AutogluonModels/chronos_bolt_feature_pipeline_run",
    }

    pipeline = ChronosBoltFeaturePipeline(pipeline_config)
    run_pipeline(
        pipeline=pipeline,
        model_cls=AutoGluonAdapter,  # The final model to train on Chronos features
        raw_data=raw_data,
        model_params=model_params,
        experiment_name="Chronos_Bolt_Feature_AutoGluon_Pipeline",
        data_path=data_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run Chronos Bolt Feature Pipeline or Optuna study."
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
        config = {
            "volume_threshold": 50000,
            "tau": 0.7,
            "n_splits": 3,
            "pct_embargo": 0.01,
            "use_pca": False,
            "chronos_stride": 1,
        }
        run_optuna_optimization(
            pipeline_cls=ChronosBoltFeaturePipeline,
            model_cls=AutoGluonAdapter,
            raw_data=raw_data,
            pipeline_config=config,
            experiment_name="Chronos_Bolt_Feature_AutoGluon_Optimization",
            n_trials=10,
            run_name_prefix="chronos_bolt_feature",
            data_path=data_path,
        )
    else:
        run_single_pipeline()


if __name__ == "__main__":
    main()
