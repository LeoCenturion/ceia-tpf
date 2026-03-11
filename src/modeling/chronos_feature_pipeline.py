import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# Transformers and AutoGluon for Chronos integration
import torch

from chronos import ChronosPipeline

from src.constants import CLOSE_COL, VOLUME_COL
from src.data_analysis.data_analysis import fetch_historical_data, timer
from src.modeling.autogluon_adapter import AutoGluonAdapter
from src.modeling.pipeline_runner import run_pipeline
from src.modeling.xgboost_pipeline_palazzo import PalazzoXGBoostPipeline

logger = logging.getLogger(__name__)


class ChronosFeaturePipeline(PalazzoXGBoostPipeline):
    """
    Pipeline that uses Chronos for feature engineering, feeding embeddings into XGBoost.
    """

    def __init__(self, config):
        super().__init__(config)
        self.problem_type = "classification"
        self.chronos_model = None
        self.chronos_tokenizer = None

    @timer
    def step_2_feature_engineering(self, bars):
        logger.debug("Step 2: Generating Chronos features and combining with tabular...")
        logger.debug(f"Shape of bars entering feature engineering: {bars.shape}")

        # 1. Generate standard tabular features using parent logic
        tabular_features = super().step_2_feature_engineering(
            bars
        )  # This already calls dropna()
        logger.debug(f"Shape of tabular_features after parent engineering: {tabular_features.shape}")

        # Ensure we have a clean index after dropping NaNs from parent feature engineering
        common_index = tabular_features.index.intersection(bars.index)
        bars_aligned = bars.loc[common_index]
        logger.debug(f"Shape of bars_aligned after common index intersection: {bars_aligned.shape}")

        # 2. Load Chronos components (if not already loaded)
        if self.chronos_model is None:
            # Use a smaller model for faster iteration, can be configured later
            chronos_model_name = self.config.get(
                "chronos_model_name", "amazon/chronos-t5-tiny"
            )

            # Load the whole pipeline from the chronos library
            pipeline = ChronosPipeline.from_pretrained(
                chronos_model_name,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16,
            )
            self.chronos_model = pipeline.model
            self.chronos_tokenizer = pipeline.tokenizer

        # 3. Chronos Preprocessing: Scaling, Quantization, and Tokenization
        # We need a time-series input for Chronos. Let's use 'close_price' from bars_aligned
        # For feature extraction, we process windows of the time series.
        # Let's define a window_size and a stride.
        window_size = self.config.get("chronos_window_size", 128)  # Example window size
        stride = self.config.get("chronos_stride", 1)  # Example stride

        chronos_embeddings = []
        for i in range(0, len(bars_aligned) - window_size + 1, stride):
            window = bars_aligned.iloc[i : i + window_size]
            time_series_data = window["close_price"].values.astype(np.float32)

            time_series_tensor = torch.from_numpy(time_series_data)

            # The tokenizer's internal transform method returns a tuple (input_ids, attention_mask)
            input_ids, attention_mask, _ = self.chronos_tokenizer._input_transform(
                time_series_tensor.unsqueeze(0)
            )

            # Move inputs to GPU if encoder is on GPU
            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")
                attention_mask = attention_mask.to("cuda")

            # Extract embeddings from the encoder
            with torch.no_grad():  # Disable gradient calculation for inference
                encoder_outputs = self.chronos_model.encode(
                    input_ids, attention_mask=attention_mask
                )

            # Pooling: Use the embedding of the last token (or mean pooling)
            # For simplicity, let's use mean pooling across the sequence dimension
            embedding = encoder_outputs.mean(dim=1).squeeze().float().cpu().numpy()
            chronos_embeddings.append(embedding)

        # Convert list of embeddings to DataFrame
        if chronos_embeddings:
            chronos_features_df = pd.DataFrame(
                np.array(chronos_embeddings),
                index=bars_aligned.index[window_size - 1 :: stride],
                columns=[f"chronos_embed_{j}" for j in range(embedding.shape[-1])],
            )
        else:
            chronos_features_df = pd.DataFrame(index=pd.Index([]))
        logger.debug(f"Shape of chronos_features_df after embedding: {chronos_features_df.shape}")

        # Align chronos_features_df to tabular_features index before concatenation
        final_common_index = tabular_features.index.intersection(
            chronos_features_df.index
        )
        aligned_tabular_features = tabular_features.loc[final_common_index]
        aligned_chronos_features = chronos_features_df.loc[final_common_index]
        logger.debug(f"Shape of aligned_tabular_features: {aligned_tabular_features.shape}")
        logger.debug(f"Shape of aligned_chronos_features: {aligned_chronos_features.shape}")

        # 4. Combine Chronos embeddings with tabular features
        combined_features = pd.concat(
            [aligned_tabular_features, aligned_chronos_features], axis=1
        )
        logger.debug(f"Shape of combined_features before final dropna: {combined_features.shape}")

        final_features = combined_features.dropna()
        logger.debug(f"Final shape of features after dropna: {final_features.shape}")
        return final_features

    def log_results(self, logger, model, X_test=None, y_test=None):
        """
        Log AutoGluon specific artifacts (Leaderboard).
        """
        if hasattr(model, "leaderboard") and X_test is not None and y_test is not None:
            logger.debug("\n--- AutoGluon Leaderboard ---")
            leaderboard_data = X_test.copy()
            leaderboard_data[model.label] = (
                y_test  # Use model.label for AutoGluon's target column
            )
            leaderboard = model.leaderboard(leaderboard_data, silent=True)
            logger.debug(leaderboard)

            # Log classification report
            y_pred = model.predict(X_test)
            report = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            )

            flat_report = {}
            for class_label, metrics in report.items():
                clean_class_label = str(class_label).replace(" ", "_")
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        clean_metric_name = metric_name.replace("-", "_")
                        flat_report[
                            f"report_{clean_class_label}_{clean_metric_name}"
                        ] = value
                else:
                    flat_report[f"report_{clean_class_label}"] = metrics

            logger.log_metrics(flat_report)

            # Log Best Model Score
            if leaderboard is not None and not leaderboard.empty:
                best_model_score = leaderboard.iloc[0]["score_test"]
                best_model_name = leaderboard.iloc[0]["model"]
                logger.log_metrics({"test_f1_best_model": best_model_score})
                logger.log_params({"best_model_name": best_model_name})

                # Optionally save leaderboard as CSV artifact
                lb_path = "autogluon_leaderboard.csv"
                leaderboard.to_csv(lb_path)
                logger.log_artifact(lb_path)
                # Cleanup local file
                if os.path.exists(lb_path):
                    os.remove(lb_path)


def main():
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
        "chronos_model_name": "amazon/chronos-t5-tiny",
        "chronos_window_size": 128,
        "chronos_stride": 1,
    }

    # Primary Model (XGBoost) parameters
    model_params = {
        "label": "label",
        "eval_metric": "f1_weighted",
        # "presets": "medium_quality", # Commented out for 'best_quality' preset
        "presets": "best_quality", # Using 'best_quality' as the highest known preset, 'extreme' is not a recognized preset.
        "time_limit": 600,
        "verbosity": 1,
        "path": "AutogluonModels/chronos_feature_pipeline_run",
    }

    pipeline = ChronosFeaturePipeline(pipeline_config)

    run_pipeline(
        pipeline=pipeline,
        model_cls=AutoGluonAdapter,  # The final model to train on Chronos features
        raw_data=raw_data,
        model_params=model_params,
        experiment_name="Chronos_Feature_AutoGluon_Pipeline",
        data_path=data_path,
    )


if __name__ == "__main__":
    main()
