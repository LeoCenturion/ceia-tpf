import os

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.data_analysis.data_analysis import fetch_historical_data, timer
from src.modeling import PurgedKFold
from src.modeling.pipeline_runner import run_pipeline
from src.modeling.chronos_pipeline_palazzo import (
    PalazzoChronosPipeline,
    PalazzoChronosClassificationPipeline,
)

_DEFAULT_BOLT_MODEL = "autogluon/chronos-bolt-small"


class PalazzoChronosBoltPipeline(PalazzoChronosPipeline):
    """
    Chronos-Bolt variant of PalazzoChronosPipeline.
    Identical logic; default model is autogluon/chronos-bolt-small.
    """

    def __init__(self, config):
        config.setdefault("chronos_model", _DEFAULT_BOLT_MODEL)
        super().__init__(config)
        self.problem_type = "forecasting"


class PalazzoChronosBoltClassificationPipeline(PalazzoChronosBoltPipeline):
    """
    Chronos-Bolt variant of PalazzoChronosClassificationPipeline.
    """

    def __init__(self, config):
        super().__init__(config)
        self.problem_type = "classification"

    def log_results(self, logger, model, X_test=None, y_test=None):
        super().log_results(logger, model, X_test, y_test)

        if hasattr(self, "y_true_all") and hasattr(self, "y_pred_all"):
            print("\n--- Final Classification Report (Chronos-Bolt CV Aggregated) ---")
            from sklearn.metrics import classification_report

            print(classification_report(self.y_true_all, self.y_pred_all))
            report = classification_report(
                self.y_true_all, self.y_pred_all, output_dict=True
            )
            logger.log_metrics(
                {
                    "test_accuracy": report["accuracy"],
                    "test_macro_f1": report["macro avg"]["f1-score"],
                    "test_weighted_f1": report["weighted avg"]["f1-score"],
                }
            )


class PalazzoChronosBoltBinaryClassificationPipeline(PalazzoChronosBoltPipeline):
    """
    Chronos-Bolt variant of PalazzoChronosBinaryClassificationPipeline.
    Fine-tunes Chronos-Bolt on labels {+1.0, -1.0} via TimeSeriesPredictor.
    Default model is autogluon/chronos-bolt-small.
    """

    def __init__(self, config):
        config.setdefault("chronos_model", _DEFAULT_BOLT_MODEL)
        super().__init__(config)
        self.problem_type = "classification_finetune"

    def step_3_labeling_and_weighting(self, bars):
        """Binary labels: +1.0 if next close > current close, else -1.0."""
        returns = bars["close_price"].pct_change().shift(-1)
        y = pd.Series(np.where(returns > 0, 1.0, -1.0), index=bars.index)
        y = y.iloc[:-1]
        t1 = pd.Series(bars.index, index=bars.index)
        return y, None, t1

    def fit_predictor(
        self,
        features: pd.DataFrame,
        y: pd.Series,
        model_path: str = "AutogluonModels/ChronosBolt_Binary_Live",
    ):
        """
        Fits a Chronos-Bolt TimeSeriesPredictor on all provided features/labels.
        Returns (predictor, known_covariates_names).
        """
        import shutil

        bolt_model = self.config.get("chronos_model", _DEFAULT_BOLT_MODEL)
        prediction_length = self.config.get("prediction_length", 2)

        train_df = features.copy()
        train_df["target"] = y
        train_df["item_id"] = "live_train"
        train_df["timestamp"] = pd.date_range(
            start="2000-01-01", periods=len(train_df), freq="min"
        )

        ts_train = TimeSeriesDataFrame.from_data_frame(
            train_df, id_column="item_id", timestamp_column="timestamp"
        )

        known_covariates_names = [
            c for c in train_df.columns if c not in ["target", "item_id", "timestamp"]
        ]

        if os.path.exists(model_path):
            shutil.rmtree(model_path, ignore_errors=True)

        predictor = TimeSeriesPredictor(
            prediction_length=prediction_length,
            path=model_path,
            target="target",
            eval_metric="MASE",
            known_covariates_names=known_covariates_names,
            freq="min",
            verbosity=0,
        )
        predictor.fit(
            ts_train,
            hyperparameters={
                "Chronos": {
                    "model_path": bolt_model,
                    "fine_tune": True,
                    "fine_tune_batch_size": 16,
                }
            },
            time_limit=300,
        )

        return predictor, known_covariates_names

    def predict_next(
        self,
        predictor,
        features: pd.DataFrame,
        y: pd.Series,
        known_covariates_names: list,
    ) -> int:
        """
        Predicts the direction of the next bar.
        Returns 1 if model expects price up, 0 if down/same.
        """
        prediction_length = self.config.get("prediction_length", 2)
        context_length = 512

        combined_df = features.copy()
        combined_df["target"] = y
        combined_df["item_id"] = "live_train"
        combined_df["timestamp"] = pd.date_range(
            start="2000-01-01", periods=len(combined_df), freq="min"
        )

        train_len = len(combined_df)
        cutoff_idx = train_len

        start_ctx = max(0, cutoff_idx - context_length)
        ctx_slice = combined_df.iloc[start_ctx:cutoff_idx].copy()
        ctx_slice["item_id"] = "seq_0"

        ts_context = TimeSeriesDataFrame.from_data_frame(
            ctx_slice, id_column="item_id", timestamp_column="timestamp"
        )

        future_slice = combined_df.iloc[
            cutoff_idx : cutoff_idx + prediction_length
        ].copy()
        if len(future_slice) < prediction_length:
            missing = prediction_length - len(future_slice)
            last_row = (
                future_slice.iloc[[-1]]
                if not future_slice.empty
                else ctx_slice.iloc[[-1]]
            )
            padding = pd.concat([last_row] * missing)
            last_ts = (
                future_slice["timestamp"].iloc[-1]
                if not future_slice.empty
                else ctx_slice["timestamp"].iloc[-1]
            )
            padding["timestamp"] = pd.date_range(
                start=last_ts + pd.Timedelta(minutes=1), periods=missing, freq="min"
            )
            future_slice = pd.concat([future_slice, padding])

        future_covs = future_slice[known_covariates_names + ["timestamp"]].copy()
        future_covs["item_id"] = "seq_0"

        known_covariates = TimeSeriesDataFrame.from_data_frame(
            future_covs, id_column="item_id", timestamp_column="timestamp"
        )

        prediction = predictor.predict(ts_context, known_covariates=known_covariates)
        pred_mean = prediction.loc["seq_0"]["mean"].iloc[0]

        return int(pred_mean > 0)

    def log_results(self, logger, model, X_test=None, y_test=None):
        if hasattr(self, "y_true_all") and hasattr(self, "y_pred_all"):
            print(
                "\n--- Final Classification Report (Fine-Tuned Chronos-Bolt CV Aggregated) ---"
            )
            from sklearn.metrics import classification_report

            print(
                classification_report(
                    self.y_true_all, self.y_pred_all, target_names=["Down/Same", "Up"]
                )
            )
            report = classification_report(
                self.y_true_all, self.y_pred_all, output_dict=True
            )
            logger.log_metrics(
                {
                    "test_accuracy": report["accuracy"],
                    "test_macro_f1": report["macro avg"]["f1-score"],
                    "test_weighted_f1": report["weighted avg"]["f1-score"],
                }
            )


def main():
    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    raw_data = fetch_historical_data(
        symbol="BTC/USDT", timeframe="1m", data_path=data_path
    )

    config = {
        "volume_threshold": 50000,
        "prediction_length": 2,
        "chronos_model": _DEFAULT_BOLT_MODEL,
        "n_splits": 3,
    }

    pipeline = PalazzoChronosBoltBinaryClassificationPipeline(config)

    run_pipeline(
        pipeline=pipeline,
        model_cls=None,
        raw_data=raw_data,
        model_params={},
        experiment_name="ChronosBolt_Palazzo_FinetuneToClass",
        data_path=data_path,
    )


if __name__ == "__main__":
    main()
