import os
import numpy as np
import logging

logger = logging.getLogger(__name__)
import pandas as pd
from sklearn.metrics import (
    classification_report,
    precision_score,
    f1_score,
)

from src.data_analysis.data_analysis import fetch_historical_data
from src.modeling.chronos_feature_pipeline import ChronosFeaturePipeline
from src.modeling.autogluon_adapter import AutoGluonAdapter
from src.modeling.pipeline_runner import run_pipeline
from src.modeling import PurgedKFold
from src.constants import VOLUME_COL, CLOSE_COL
from src.modeling.mlflow_utils import MLflowLogger


class ChronosMetaLabelingPipeline(ChronosFeaturePipeline):
    """
    Pipeline that implements Meta-Labeling on top of Chronos features using AutoGluon.
    It splits data into Train/Test, uses CV on Train to generate OOF predictions,
    trains a Meta-Model to predict correctness of Primary Model,
    and then evaluates on Test.
    """

    def __init__(self, config):
        super().__init__(config)
        self.primary_model_ = None
        self.meta_model_ = None
        self.t1_train_ = None

    def generate_oof_predictions(self, X, y, t1, sw, model):
        """
        Generates Out-Of-Fold predictions for the given dataset using PurgedKFold.
        Returns a DataFrame with [true_label, primary_pred, primary_prob]
        """
        cv = PurgedKFold(
            n_splits=self.config["n_splits"],
            t1=t1,
            pct_embargo=self.config["pct_embargo"],
        )

        oof_preds = pd.Series(index=X.index, dtype=float)
        oof_probs = pd.Series(index=X.index, dtype=float)
        fold_f1_scores = []

        logger.info(f"Generating OOF predictions with {self.config['n_splits']} folds...")

        for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            # Split
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]
            sw_train = sw.iloc[train_idx]
            logger.debug(f"Fold {i+1}: X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, y_train shape: {y_train.shape}, y_val shape: {y_val.shape}, sw_train shape: {sw_train.shape}")

            # Since model is AutoGluonAdapter, we instantiate it new for each fold
            # to ensure it's a fresh model.
            fold_model = type(model)(**model.get_params())

            # AutoGluon handles data scaling internally, so no manual scaling/PCA here
            fold_model.fit(X_train, y_train, sample_weight=sw_train.values)

            # Predict
            val_pred = fold_model.predict(X_val)
            val_prob = fold_model.predict_proba(X_val)[:, 1]

            # Score
            fold_f1 = f1_score(y_val, val_pred, average="weighted")
            fold_f1_scores.append(fold_f1)
            logger.info(f"Fold {i + 1} F1 Score: {fold_f1:.4f}")

            # Store OOF
            fold_indices = X.iloc[val_idx].index
            oof_preds.loc[fold_indices] = val_pred
            oof_probs.loc[fold_indices] = val_prob

        logger.debug(f"Average OOF CV F1 Score: {np.mean(fold_f1_scores):.4f}")

        return pd.DataFrame(
            {"true_label": y, "primary_pred": oof_preds, "primary_prob": oof_probs},
            index=X.index,
        )

    def log_results(
        self,
        mlflow_logger,
        primary_model,
        meta_model,
        metrics,
        X_test,
        y_test,
        t1_test,
        primary_test_pred,
        final_decision,
    ):
        """
        Log Meta-Labeling specific artifacts and metrics.
        """
        mlflow_logger.log_metrics(metrics)

        # Capture and log full classification reports as artifacts
        baseline_report = classification_report(
            y_test, primary_test_pred, output_dict=True, zero_division=0
        )
        meta_labeling_report = classification_report(
            y_test, final_decision, output_dict=True, zero_division=0
        )

        # Flatten and log baseline report metrics
        flat_baseline_report = {}
        for class_label, metrics_dict in baseline_report.items():
            clean_class_label = class_label.replace(" ", "_")
            if isinstance(metrics_dict, dict):
                for metric_name, value in metrics_dict.items():
                    clean_metric_name = metric_name.replace("-", "_")
                    flat_baseline_report[f"baseline_{clean_class_label}_{clean_metric_name}"] = value
            else:
                flat_baseline_report[f"baseline_{clean_class_label}"] = metrics_dict
        mlflow_logger.log_metrics(flat_baseline_report)


        # Flatten and log meta-labeling report metrics
        flat_meta_labeling_report = {}
        for class_label, metrics_dict in meta_labeling_report.items():
            clean_class_label = class_label.replace(" ", "_")
            if isinstance(metrics_dict, dict):
                for metric_name, value in metrics_dict.items():
                    clean_metric_name = metric_name.replace("-", "_")
                    flat_meta_labeling_report[f"metalabeling_{clean_class_label}_{clean_metric_name}"] = value
            else:
                flat_meta_labeling_report[f"metalabeling_{clean_class_label}"] = metrics_dict
        mlflow_logger.log_metrics(flat_meta_labeling_report)

        mlflow_logger.log_artifact_dict(baseline_report, "baseline_classification_report.json")
        mlflow_logger.log_artifact_dict(meta_labeling_report, "meta_labeling_classification_report.json")

        # Log AutoGluon Meta-Model Leaderboard
        if (
            hasattr(meta_model, "leaderboard")
            and X_test is not None
            and y_test is not None
        ):
            logger.debug("\n--- AutoGluon Meta-Model Leaderboard ---")
            leaderboard_data = X_test.copy()
            leaderboard_data["primary_prob"] = primary_model.predict_proba(X_test)[:, 1]
            leaderboard_data["primary_pred"] = primary_model.predict(X_test)
            # The meta-model was trained on meta_labels derived from primary model's correctness.
            # We need to recreate meta_labels for test set for leaderboard.
            meta_labels_test = (primary_test_pred == y_test).astype(int)
            leaderboard_data[meta_model.label] = meta_labels_test

            leaderboard = meta_model.leaderboard(leaderboard_data, silent=True)
            logger.debug(leaderboard)

            if leaderboard is not None and not leaderboard.empty:
                best_meta_model_score = leaderboard.iloc[0]["score_test"]
                best_meta_model_name = leaderboard.iloc[0]["model"]
                mlflow_logger.log_metrics({"test_f1_best_meta_model": best_meta_model_score})
                mlflow_logger.log_params({"best_meta_model_name": best_meta_model_name})

                lb_path = "autogluon_meta_leaderboard.csv"
                leaderboard.to_csv(lb_path)
                mlflow_logger.log_artifact(lb_path)
                if os.path.exists(lb_path):
                    os.remove(lb_path)

    def fit(self, raw_data: pd.DataFrame, model_cls, model_params):
        """
        Trains the primary and meta-models and stores them on the instance.
        """
        logger.debug(f"Starting fit method with raw_data of shape: {raw_data.shape}")
        primary_model_params = model_params.get("primary_model_params", {})
        meta_model_config = model_params.get("meta_model_config", {})

        primary_model_init = model_cls(**primary_model_params)

        bars = self.step_1_data_structuring(raw_data)
        logger.debug(f"After step_1_data_structuring, bars shape: {bars.shape}")
        if bars.empty:
            raise ValueError(
                "No volume bars created from the provided data. Cannot fit model."
            )

        features = self.step_2_feature_engineering(bars)
        logger.debug(
            f"After step_2_feature_engineering, features shape: {features.shape}"
        )
        labels, sample_weights, t1 = self.step_3_labeling_and_weighting(bars)
        logger.debug(
            f"After step_3_labeling_and_weighting, labels shape: {labels.shape}, sw shape: {sample_weights.shape}, t1 shape: {t1.shape}"
        )

        # Alignment
        common_idx = (
            features.index.intersection(labels.index)
            .intersection(sample_weights.index)
            .intersection(t1.index)
        )
        X = features.loc[common_idx]
        y = labels.loc[common_idx]
        sw = sample_weights.loc[common_idx]
        t1 = t1.loc[common_idx]
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        logger.debug(
            f"After alignment - X shape: {X.shape}, y shape: {y.shape}, sw shape: {sw.shape}, t1 shape: {t1.shape}"
        )
        if X.empty or y.empty:
            raise ValueError("Aligned features or labels are empty. Cannot fit model.")

        # Time-series split (for OOF and final training)
        split_idx = int(len(X) * 0.8)
        X_train, y_train, sw_train, t1_train = (
            X.iloc[:split_idx],
            y.iloc[:split_idx],
            sw.iloc[:split_idx],
            t1.iloc[:split_idx],
        )
        X_test, y_test, t1_test = (
            X.iloc[split_idx:],
            y.iloc[split_idx:],
            t1.iloc[split_idx:],
        )

        logger.debug(
            f"Before OOF generation - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, sw_train shape: {sw_train.shape}, t1_train shape: {t1_train.shape}"
        )
        if X_train.empty or y_train.empty:
            raise ValueError("X_train or y_train is empty after internal split. Cannot generate OOF predictions.")

        # Generate OOF predictions
        oof_df = self.generate_oof_predictions(
            X_train, y_train, t1_train, sw_train, primary_model_init
        )
        logger.debug(f"After generate_oof_predictions, oof_df shape: {oof_df.shape}")

        # Create meta-labels and meta-dataset
        meta_labels = (oof_df["primary_pred"] == oof_df["true_label"]).astype(int)
        X_meta_train = X_train.copy()
        X_meta_train["primary_prob"] = oof_df["primary_prob"]
        logger.debug(
            f"Before meta-model training, X_meta_train shape: {X_meta_train.shape}"
        )

        # Train meta-model (also AutoGluon)
        self.meta_model_ = model_cls(**meta_model_config)
        self.meta_model_.fit(X_meta_train, meta_labels)

        # Train final primary model on all training data
        self.primary_model_ = model_cls(**primary_model_params)
        self.primary_model_.fit(X_train, y_train, sample_weight=sw_train.values)
        self.t1_train_ = t1_train  # Store t1 for potential future use or debugging
        return X_test, y_test, t1_test

    def predict(self, data_window: pd.DataFrame) -> int:
        """
        Predicts a single signal (0 or 1) for the latest data point in the provided window.
        Assumes self.primary_model_ and self.meta_model_ are already trained.
        """
        logger.debug(
            f"Starting predict method with data_window of shape: {data_window.shape}"
        )
        if self.primary_model_ is None or self.meta_model_ is None:
            raise RuntimeError("Models not fitted. Call .fit() first.")

        # 1. Perform data structuring and feature engineering on the data_window
        bars = self.step_1_data_structuring(data_window)
        logger.debug(
            f"After step_1_data_structuring in predict, bars shape: {bars.shape}"
        )
        if bars.empty:
            return 0  # Cannot make a prediction if no bars are formed

        features = self.step_2_feature_engineering(bars)
        logger.debug(
            f"After step_2_feature_engineering in predict, features shape: {features.shape}"
        )

        if features.empty:
            return 0

        # Get the latest feature set for prediction
        latest_features = features.iloc[-1:]
        logger.debug(f"latest_features shape: {latest_features.shape}")

        # 2. Predict with primary model
        primary_pred = self.primary_model_.predict(latest_features)
        primary_prob = self.primary_model_.predict_proba(latest_features)[:, 1]

        # 3. Predict with meta-model
        X_meta_single = latest_features.copy()
        X_meta_single["primary_prob"] = primary_prob
        meta_pred = self.meta_model_.predict(X_meta_single)

        # 4. Final Decision
        final_decision = (primary_pred == 1) & (meta_pred == 1)

        return final_decision.astype(int).iloc[0]  # Return single int

    def run(
        self,
        raw_data: pd.DataFrame,
        model_cls,
        model_params,
        experiment_name="Chronos_Pipeline_Run",
    ):
        """
        Runs the full meta-labeling experiment, maintaining backward compatibility.
        Uses the new fit and predict methods internally.
        """
        mlflow_logger = MLflowLogger(experiment_name=experiment_name)
        with mlflow_logger.start_run(run_name=f"Chronos_MetaLabeling_Run", nested=True):
            mlflow_logger.log_params(
                {"pipeline_config": self.config, "model_params": model_params}
            )

            # Fit the pipeline and get test data
            X_test, y_test, t1_test = self.fit(raw_data, model_cls, model_params)
            logger.debug(
                f"After fit and split, X_test shape: {X_test.shape}, y_test shape: {y_test.shape}"
            )

            # Generate predictions for the test set using the fitted models
            primary_test_pred = self.primary_model_.predict(X_test)
            primary_test_prob = self.primary_model_.predict_proba(X_test)[:, 1]

            X_meta_test = X_test.copy()
            X_meta_test["primary_prob"] = primary_test_prob
            logger.debug(
                f"Before meta-model prediction, X_meta_test shape: {X_meta_test.shape}"
            )
            meta_test_pred = self.meta_model_.predict(X_meta_test)

            final_decision = (primary_test_pred == 1) & (meta_test_pred == 1)
            final_decision = final_decision.astype(int)

            # --- Metrics ---
            prec_baseline = precision_score(
                y_test, primary_test_pred, pos_label=1, zero_division=0
            )
            prec_meta = precision_score(
                y_test, final_decision, pos_label=1, zero_division=0
            )

            metrics = {
                "baseline_precision": prec_baseline,
                "metalabeling_precision": prec_meta,
                "baseline_f1_weighted": f1_score(
                    y_test, primary_test_pred, average="weighted"
                ),
                "metalabeling_f1_weighted": f1_score(
                    y_test, final_decision, average="weighted"
                ),
            }

            self.log_results(
                mlflow_logger,
                self.primary_model_,
                self.meta_model_,
                metrics,
                X_test,
                y_test,
                t1_test,
                primary_test_pred,
                final_decision,
            )

            return (
                self.primary_model_,
                self.meta_model_,
                metrics,
                X_test,
                y_test,
                t1_test,
                primary_test_pred,
                final_decision,
            )


def main():
    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    raw_data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        data_path=data_path,
    )

    # Configuration for the pipeline (mostly for feature engineering)
    pipeline_config = {
        "volume_threshold": 50000,
        "tau": 0.7,
        "n_splits": 3,
        "pct_embargo": 0.01,
        "use_pca": False,
        "chronos_model_name": "amazon/chronos-t5-tiny",
        "chronos_window_size": 128,
    }

    # Primary Model (AutoGluon) parameters
    primary_model_params = {
        "label": "label",
        "eval_metric": "f1_weighted",
        "presets": "medium_quality",
        "time_limit": 60,
        "path": "AutogluonModels/tmp_chronos_primary",  # tmp prefix
    }

    # Meta Model (AutoGluon) parameters
    meta_model_config = {
        "label": "label",
        "eval_metric": "f1",
        "presets": "best_quality",
        "time_limit": 30,
        "path": "AutogluonModels/tmp_chronos_meta",  # tmp prefix
    }

    model_params = {
        "primary_model_params": primary_model_params,
        "meta_model_config": meta_model_config,
    }

    pipeline = ChronosMetaLabelingPipeline(pipeline_config)

    run_pipeline(
        pipeline=pipeline,
        model_cls=AutoGluonAdapter,  # This is the class for both models
        raw_data=raw_data,
        model_params=model_params,
        experiment_name="Chronos_MetaLabeling_Pipeline",  # tmp prefix
        data_path=data_path,
    )


if __name__ == "__main__":
    main()
