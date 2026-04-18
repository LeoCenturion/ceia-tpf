import logging
import os
import sys
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from chronos import ChronosPipeline

from src.constants import CLOSE_COL, VOLUME_COL
from src.data_analysis.data_analysis import fetch_historical_data
from src.modeling.autogluon_adapter import AutoGluonAdapter
from src.modeling.pipeline_runner import run_pipeline
from src.modeling.xgboost_pipeline_palazzo import PalazzoXGBoostPipeline
from src.modeling.pipeline import timer
from src.modeling import PurgedKFold

logger = logging.getLogger(__name__)


class FinetunedChronosFeaturePipeline(PalazzoXGBoostPipeline):
    """
    This pipeline fine-tunes a Chronos model within each CV fold to generate
    domain-specific embeddings, which are then combined with other features.
    It correctly uses the `cross_validation_feature_engineering` hook.
    """

    def __init__(self, config):
        super().__init__(config)
        self.bars_for_cv = None  # Cache for bars
        self.chronos_model_for_embedding = None
        self.chronos_tokenizer_for_embedding = None

    @timer
    def step_1_data_structuring(self, raw_tick_data):
        """
        Generate volume bars and cache them for use in the CV loop.
        """
        bars = super().step_1_data_structuring(raw_tick_data)
        self.bars_for_cv = bars.copy()  # Cache the bars
        return bars

    def _generate_embeddings(self, bars):
        """Helper function to generate embeddings for a given set of bars."""
        if self.chronos_model_for_embedding is None:
            raise RuntimeError("Chronos model for embedding is not set.")

        window_size = self.config.get("chronos_window_size", 128)
        stride = self.config.get("chronos_stride", 1)

        chronos_embeddings = []
        indices = []
        
        for i in tqdm(range(0, len(bars) - window_size + 1, stride), desc="Generating Embeddings"):
            window = bars.iloc[i : i + window_size]
            indices.append(window.index[-1])
            time_series_data = window["close_price"].values.astype(np.float32)
            time_series_tensor = torch.from_numpy(time_series_data)

            input_ids, attention_mask, _ = self.chronos_tokenizer_for_embedding._input_transform(
                time_series_tensor.unsqueeze(0)
            )
            if torch.cuda.is_available():
                input_ids, attention_mask = input_ids.to("cuda"), attention_mask.to("cuda")

            with torch.no_grad():
                encoder_outputs = self.chronos_model_for_embedding.encode(
                    input_ids, attention_mask=attention_mask
                )
            
            embedding = encoder_outputs.mean(dim=1).squeeze().float().cpu().numpy()
            chronos_embeddings.append(embedding)

        if not chronos_embeddings:
            return pd.DataFrame()

        return pd.DataFrame(
            np.array(chronos_embeddings),
            index=pd.Index(indices, name='close_time'),
            columns=[f"chronos_embed_{j}" for j in range(embedding.shape[0])],
        )

    def cross_validation_feature_engineering(self, X_train_raw, X_test_raw, y_train_for_finetuning):
        logger.info("Performing in-loop feature engineering via fine-tuning...")

        # --- 1. Get original bars data for this fold ---
        bars_train = self.bars_for_cv.loc[X_train_raw.index]
        bars_test = self.bars_for_cv.loc[X_test_raw.index]

        # --- 2. Fine-tune Chronos on the training data for this fold ---
        fold_model_path = f"AutogluonModels/FinetunedChronos_Temp_Fold"
        if os.path.exists(fold_model_path):
            shutil.rmtree(fold_model_path)

        # Use the close price as the target for self-supervised fine-tuning
        train_df_finetune = pd.DataFrame({
            "timestamp": bars_train.index,
            "target": bars_train['close_price'].values,
            "item_id": "fold_train",
        })
        ts_train_finetune = TimeSeriesDataFrame.from_data_frame(
            train_df_finetune, id_column="item_id", timestamp_column="timestamp"
        )

        predictor = TimeSeriesPredictor(
            prediction_length=1, path=fold_model_path, target="target", verbosity=0
        )
        predictor.fit(
            ts_train_finetune,
            hyperparameters={
                "Chronos": {
                    "model_path": self.config.get("chronos_model_name", "amazon/chronos-t5-tiny"),
                    "fine_tune": True,
                    "fine_tune_batch_size": self.config.get("fine_tune_batch_size", 16),
                }
            },
            time_limit=self.config.get("finetune_time_limit", 300),
        )

        # --- 3. Load the fine-tuned model for embedding extraction ---
        try:
            finetuned_model_path = os.path.join(predictor.path, "models", "Chronos")
            finetuned_pipeline = ChronosPipeline.from_pretrained(
                finetuned_model_path,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16,
            )
            self.chronos_model_for_embedding = finetuned_pipeline.model
            self.chronos_tokenizer_for_embedding = finetuned_pipeline.tokenizer
        except Exception as e:
            logger.error(f"Failed to load fine-tuned Chronos model: {e}")
            return pd.DataFrame(), pd.DataFrame()

        # --- 4. Generate embeddings and combine with tabular features ---
        logger.info("Extracting embeddings using the fine-tuned model...")
        chronos_features_train = self._generate_embeddings(bars_train)
        chronos_features_test = self._generate_embeddings(bars_test)
        
        # Combine with the pre-computed tabular features passed into this method
        X_train_final = pd.concat([X_train_raw, chronos_features_train], axis=1).dropna()
        X_test_final = pd.concat([X_test_raw, chronos_features_test], axis=1).dropna()
        
        return X_train_final, X_test_final

def run_single_pipeline():
    """Defines and runs a single pipeline for demonstration or debugging."""
    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    raw_data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        start_date="2022-01-01T00:00:00Z",
        data_path=data_path,
    )

    pipeline_config = {
        "volume_threshold": 50000,
        "tau": 0.7,
        "n_splits": 3,
        "pct_embargo": 0.01,
        "chronos_model_name": "amazon/chronos-t5-tiny",
        "chronos_window_size": 128,
        "finetune_time_limit": 30,
        "fine_tune_batch_size": 16,
    }

    model_params = {
        "label": "label",
        "eval_metric": "f1_weighted",
        "presets": "medium_quality",
        "time_limit": 180,
        "path": "AutogluonModels/finetuned_chronos_run",
    }

    pipeline = FinetunedChronosFeaturePipeline(pipeline_config)

    run_pipeline(
        pipeline=pipeline,
        model_cls=AutoGluonAdapter,
        raw_data=raw_data,
        model_params=model_params,
        experiment_name="Finetuned_Chronos_Feature_Pipeline",
        data_path=data_path,
    )

def main():
    """Main entry point for the script."""
    run_single_pipeline()

if __name__ == "__main__":
    main()
