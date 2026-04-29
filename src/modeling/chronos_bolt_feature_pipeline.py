import logging

import numpy as np
import pandas as pd
import torch
from chronos import ChronosBoltPipeline

from src.data_analysis.data_analysis import fetch_historical_data, timer
from src.modeling.pipeline_runner import run_pipeline
from src.modeling.xgboost_pipeline_palazzo import PalazzoXGBoostPipeline

logger = logging.getLogger(__name__)

_DEFAULT_BOLT_MODEL = "autogluon/chronos-bolt-small"


class ChronosBoltFeaturePipeline(PalazzoXGBoostPipeline):
    """
    Extends PalazzoXGBoostPipeline with Chronos-Bolt patch embeddings added to
    the tabular feature set.  The downstream model is XGBoost (or any model
    passed to run_cv), same as the parent pipeline.
    """

    def __init__(self, config):
        super().__init__(config)
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


def main():
    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    raw_data = fetch_historical_data(
        symbol="BTC/USDT", timeframe="1m", data_path=data_path
    )

    config = {
        "volume_threshold": 50000,
        "tau": 0.7,
        "n_splits": 3,
        "pct_embargo": 0.01,
        "use_pca": False,
        "chronos_model_name": _DEFAULT_BOLT_MODEL,
        "chronos_window_size": 32,
        "chronos_stride": 1,
    }

    model_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
    }

    pipeline = ChronosBoltFeaturePipeline(config)
    run_pipeline(
        pipeline=pipeline,
        model_cls=None,
        raw_data=raw_data,
        model_params=model_params,
        experiment_name="ChronosBolt_Feature_XGBoost_Pipeline",
        data_path=data_path,
    )


if __name__ == "__main__":
    main()
