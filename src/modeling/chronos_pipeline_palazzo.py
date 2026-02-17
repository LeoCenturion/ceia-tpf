import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from sklearn.metrics import f1_score, accuracy_score

# Make the script runnable from anywhere
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_analysis.data_analysis import fetch_historical_data, timer
from src.modeling.xgboost_pipeline_palazzo import PalazzoXGBoostPipeline
from src.modeling.pipeline_runner import run_pipeline
from src.modeling.mlflow_utils import MLflowLogger
from src.modeling import PurgedKFold
from src.constants import VOLUME_COL, CLOSE_COL, OPEN_COL, HIGH_COL, LOW_COL


class PalazzoChronosPipeline(PalazzoXGBoostPipeline):
    """
    Pipeline that adapts the Palazzo structure for Time Series Forecasting using Chronos.
    """

    def __init__(self, config):
        super().__init__(config)
        self.problem_type = "forecasting"

    def step_3_labeling_and_weighting(self, bars):
        y = bars["close_price"]
        # Use index as t1 (instantaneous event) for PurgedKFold
        t1 = pd.Series(bars.index, index=bars.index)
        return y, None, t1

    @timer
    def step_2_feature_engineering(self, bars):
        """
        Custom feature engineering for Chronos: Filter out non-stationary features.
        """
        # 1. Generate all features using parent logic
        features = super().step_2_feature_engineering(bars)

        # 2. Filter for Chronos: Keep only Stationary / Percentage-based features
        # We explicitly drop features that scale with price level or volume level

        # List of substrings or exact matches to DROP
        cols_to_drop = [
            f"feature_{VOLUME_COL}",
            "feature_avg_volume_20",
            "feature_AO",
            "feature_MACD",
            "feature_MACD_Signal",
            "feature_MACD_Hist",
            # We also drop absolute price lags if they exist (though usually they are pct lags in the parent)
            # The parent creates "feature_open_pct_lag_1" etc. which ARE stationary.
            # But just in case any absolute price lags crept in (unlikely based on code review).
        ]

        # Filter existing columns
        existing_cols_to_drop = [c for c in cols_to_drop if c in features.columns]

        if existing_cols_to_drop:
            print(
                f"Dropping {len(existing_cols_to_drop)} non-stationary features for Chronos: {existing_cols_to_drop}"
            )
            features = features.drop(columns=existing_cols_to_drop)

        return features

    @timer
    def run_cv(self, raw_tick_data, model=None):
        """
        Executes Purged Cross-Validation for Chronos.
        - Training: Concatenates all training folds into a single synthetic continuous series.
        - Validation: Appends validation fold to training and performs rolling 1-step prediction.
        """
        # 1. Pipeline Steps
        bars = self.step_1_data_structuring(raw_tick_data)
        features = self.step_2_feature_engineering(bars)
        y, _, t1 = self.step_3_labeling_and_weighting(bars)

        # 2. Align
        common_idx = features.index.intersection(y.index).intersection(t1.index)
        features = features.loc[common_idx]
        y = y.loc[common_idx]
        t1 = t1.loc[common_idx]

        # 3. Setup CV
        n_splits = self.config.get("n_splits", 3)
        pct_embargo = self.config.get("pct_embargo", 0.0)
        cv = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)

        prediction_length = self.config.get("prediction_length", 2)
        model_path = self.config.get("chronos_model", "amazon/chronos-t5-small")

        scores = []
        all_y_true = []
        all_y_pred = []

        print(f"Starting Purged Cross-Validation ({n_splits} folds)...")

        for i, (train_idx, test_idx) in enumerate(cv.split(features, y)):
            print(f"\n--- Fold {i + 1}/{n_splits} ---")

            # --- Prepare Training Data (Disjoint Segments) ---
            # Instead of concatenating, we identify contiguous segments and treat them as separate items

            # Find where indices are not continuous (diff > 1)
            # train_idx is sorted.
            gaps = np.where(np.diff(train_idx) > 1)[0]
            # split_indices points to the *start* of the new segment, so we add 1 to the index found by diff
            split_points = gaps + 1

            segments_indices = np.split(train_idx, split_points)

            train_items = []
            for seg_i, seg_idx in enumerate(segments_indices):
                if len(seg_idx) == 0:
                    continue

                # Extract segment data
                X_seg = features.iloc[seg_idx].copy()
                y_seg = y.iloc[seg_idx].copy()

                seg_df = X_seg.copy()
                seg_df["target"] = y_seg
                seg_df["item_id"] = f"train_fold_{i}_seg_{seg_i}"

                # Use SYNTHETIC timestamps for each segment starting from a fixed date
                # We restart the clock for each segment because Chronos treats items independently.
                # This ensures each segment looks like a valid continuous series starting at t=0.
                start_date = pd.Timestamp("2000-01-01")
                seg_timestamps = pd.date_range(
                    start=start_date, periods=len(seg_df), freq="min"
                )
                seg_df["timestamp"] = seg_timestamps

                train_items.append(seg_df)

            # Combine all segments into one DataFrame
            train_df = pd.concat(train_items)

            # Convert to TimeSeriesDataFrame
            ts_train = TimeSeriesDataFrame.from_data_frame(
                train_df, id_column="item_id", timestamp_column="timestamp"
            )

            known_covariates_names = [
                c
                for c in train_df.columns
                if c not in ["target", "item_id", "timestamp"]
            ]

            # Use specific model path for this fold to avoid conflicts
            fold_model_path = f"AutogluonModels/Chronos_Palazzo_Fold_{i}"
            if os.path.exists(fold_model_path):
                import shutil

                shutil.rmtree(fold_model_path, ignore_errors=True)

            # Fit Chronos
            predictor = TimeSeriesPredictor(
                prediction_length=prediction_length,
                path=fold_model_path,
                target="target",
                eval_metric="MASE",
                known_covariates_names=known_covariates_names,
                freq="min",
                verbosity=0,
            )

            predictor.fit(
                ts_train,
                hyperparameters={"Chronos": {"model_path": model_path}},
                time_limit=300,
            )

            # Also keep track of the last segment's timestamps to align the validation set correctly?
            # For validation rolling prediction, we append to the "end" of the training context.
            # Which training context?
            # Ideally, the "most recent" segment (the one just before the test set).
            # PurgedKFold: test set is usually "surrounded" or at the end.
            # If test is in the middle, "pre-test" data is one segment, "post-test" is another.
            # For autoregressive prediction of the test set, we should use the PRE-TEST segment as context.
            # Post-test data is "future" relative to test set, so it shouldn't be context.

            # Identify the segment that immediately precedes the test set
            # The test_idx[0] is the start of testing.
            # We look for the segment where seg_idx[-1] < test_idx[0] and is closest.

            # However, in PurgedKFold, train indices are: [0...test_start-gap] and [test_end+embargo...end]
            # The "context" for predicting test_start is the first segment.
            # We should NOT use the second segment (future) as context for prediction.

            # So, for the rolling prediction loop, we select the correct context segment.

            test_start_abs = test_idx[0]

            # Find the context segment (the one ending before test_start)
            context_df = None
            for seg_idx, seg_df in zip(segments_indices, train_items):
                if len(seg_idx) > 0 and seg_idx[-1] < test_start_abs:
                    # This segment is before the test set.
                    # We want the one that is *immediately* before (or the latest one before).
                    # Since segments are ordered, we can just keep updating 'context_df'
                    # until we hit a segment that starts after test_set.
                    context_df = seg_df

            if context_df is None:
                # Fallback: if no training data before test (e.g. test is at start),
                # we technically have no context. Chronos zero-shot?
                # But PurgedCV usually ensures some train data if not shuffle=True.
                # We'll assume context_df is found.
                # If not, use the first segment (though incorrect causally if it's future).
                context_df = train_items[0]

            # --- Prepare Validation Data (Rolling Prediction) ---
            # We need to predict y[test_idx[k]] using Context + Test[:k]
            # Context comes from `context_df` (the pre-gap training segment)

            X_test = features.iloc[test_idx].copy()
            y_test = y.iloc[test_idx].copy()

            test_df = X_test.copy()
            test_df["target"] = y_test
            # Ensure timestamps align with context
            # We continue the clock from context_df's last timestamp + 1 min
            last_ctx_ts = context_df["timestamp"].iloc[-1]
            start_test_ts = last_ctx_ts + pd.Timedelta(minutes=1)
            test_timestamps = pd.date_range(
                start=start_test_ts, periods=len(test_df), freq="min"
            )
            test_df["timestamp"] = test_timestamps

            # Combine Context + Test for rolling window extraction
            # context_df has 'item_id', 'target', 'timestamp', and features
            # test_df has features, 'target', 'timestamp'. Add matching item_id.
            test_df["item_id"] = context_df["item_id"].iloc[0]

            # Concat
            combined_df = pd.concat([context_df, test_df])

            # Identify indices relative to the combined dataframe
            train_len = len(context_df)
            test_len = len(test_df)

            print(f"Generating rolling predictions for {test_len} steps...")

            fold_y_pred = []

            # Optimization: We only need the last context_length points.
            context_length = 512

            # Batch Prediction Strategy
            # We create a TimeSeriesDataFrame where each 'item_id' is a specific test instance
            # Instance k: data up to index (train_len + k - 1). We want to predict index (train_len + k).

            batch_size = 64
            for start_k in tqdm(range(0, test_len, batch_size), desc="Predicting"):
                end_k = min(start_k + batch_size, test_len)
                batch_items = []
                batch_covariates_list = []

                for k in range(start_k, end_k):
                    # Cutoff index in combined_df: we include data up to train_len + k - 1
                    cutoff_idx = train_len + k

                    # --- Context (History) ---
                    # Context window: [cutoff_idx - context_length : cutoff_idx]
                    start_ctx = max(0, cutoff_idx - context_length)
                    ctx_slice = combined_df.iloc[start_ctx:cutoff_idx].copy()
                    ctx_slice["item_id"] = f"seq_{k}"
                    batch_items.append(ctx_slice)

                    # --- Known Covariates (Future) ---
                    # We need covariates for [cutoff_idx : cutoff_idx + prediction_length]
                    # Since prediction_length=2, we need t+1 and t+2 relative to context end.
                    future_slice = combined_df.iloc[
                        cutoff_idx : cutoff_idx + prediction_length
                    ].copy()

                    # Handle edge case where we run out of data at the end of the test set
                    if len(future_slice) < prediction_length:
                        missing_steps = prediction_length - len(future_slice)
                        if len(future_slice) > 0:
                            last_row = future_slice.iloc[[-1]]
                        else:
                            # If absolutely no future data (shouldn't happen for 1-step unless index error), use last context row
                            last_row = ctx_slice.iloc[[-1]]

                        # Pad with last available row to satisfy length requirement
                        padding = pd.concat([last_row] * missing_steps)
                        # Fix timestamps for padding
                        last_ts = (
                            future_slice["timestamp"].iloc[-1]
                            if not future_slice.empty
                            else ctx_slice["timestamp"].iloc[-1]
                        )
                        padding["timestamp"] = pd.date_range(
                            start=last_ts + pd.Timedelta(minutes=1),
                            periods=missing_steps,
                            freq="min",
                        )
                        future_slice = pd.concat([future_slice, padding])

                    # Ensure columns match known_covariates_names (exclude target)
                    future_covs = future_slice[
                        known_covariates_names + ["timestamp"]
                    ].copy()
                    future_covs["item_id"] = f"seq_{k}"
                    batch_covariates_list.append(future_covs)

                # Combine into TS DataFrames
                batch_df = pd.concat(batch_items)
                batch_ts = TimeSeriesDataFrame.from_data_frame(
                    batch_df, id_column="item_id", timestamp_column="timestamp"
                )

                batch_cov_df = pd.concat(batch_covariates_list)
                batch_covariates = TimeSeriesDataFrame.from_data_frame(
                    batch_cov_df, id_column="item_id", timestamp_column="timestamp"
                )

                # Predict with known covariates
                preds = predictor.predict(batch_ts, known_covariates=batch_covariates)

                # Extract the 1-step ahead forecast (first row for each item)
                for k in range(start_k, end_k):
                    item_id = f"seq_{k}"
                    item_preds = preds.loc[item_id]
                    pred_value = item_preds["mean"].iloc[0]
                    fold_y_pred.append(pred_value)

            # --- Scoring ---
            fold_y_pred = np.array(fold_y_pred)
            fold_y_true = y_test.values

            if self.problem_type == "classification":
                # Convert to directional labels
                # Direction = (Pred[t] > Actual[t-1])
                # Note: Actual[t-1] is the last observed price in the context.
                # For k=0, context ends at train_len-1. Last price is y_train.iloc[-1].
                # For k>0, context ends at train_len+k-1. Last price is y_test.iloc[k-1].

                # Construct "Previous Prices" vector aligned with y_test
                # prev_prices[0] = last context price
                # prev_prices[k] = y_test[k-1]
                last_context_price = context_df["target"].iloc[-1]
                prev_prices = np.concatenate([[last_context_price], y_test.values[:-1]])

                pred_class = (fold_y_pred > prev_prices).astype(int)
                true_class = (fold_y_true > prev_prices).astype(int)

                fold_score = f1_score(true_class, pred_class, average="weighted")
                acc = accuracy_score(true_class, pred_class)
                print(f"Fold {i + 1} F1: {fold_score:.4f}, Acc: {acc:.4f}")

                all_y_true.extend(true_class)
                all_y_pred.extend(pred_class)

            else:
                # MASE or similar for regression
                # Just simplified MAE for now as placeholder
                fold_score = -np.mean(np.abs(fold_y_true - fold_y_pred))  # Negative MAE
                print(f"Fold {i + 1} MAE: {-fold_score:.4f}")

            # Store last test data for leaderboard logging (from last fold)
            # We use test_df which is already set up with item_id and timestamps relative to context
            self.last_test_data = TimeSeriesDataFrame.from_data_frame(
                test_df, id_column="item_id", timestamp_column="timestamp"
            )

            scores.append(fold_score)

        # Store aggregated results for final logging
        self.y_true_all = np.array(all_y_true)
        self.y_pred_all = np.array(all_y_pred)

        return predictor, scores, features, y, None, None, None

    def log_results(self, logger, model, X_test=None, y_test=None):
        """
        Log Time Series specific artifacts.
        """
        if hasattr(model, "leaderboard"):
            print("\n--- Chronos Leaderboard ---")
            leaderboard = model.leaderboard(self.last_test_data, silent=True)
            print(leaderboard)

            if not leaderboard.empty:
                best_model = leaderboard.iloc[0]
                logger.log_metrics({"test_mase_best_model": best_model["score_test"]})
                logger.log_params({"best_ts_model": best_model["model"]})

                lb_path = "chronos_leaderboard.csv"
                leaderboard.to_csv(lb_path)
                logger.log_artifact(lb_path)
                os.remove(lb_path)


class PalazzoChronosClassificationPipeline(PalazzoChronosPipeline):
    """
    Inherits from the Forecasting Pipeline but interprets the output as a Classification task.
    It predicts Future Price, then converts (Predicted - Current) > 0 -> Label 1, else 0.
    """

    def __init__(self, config):
        super().__init__(config)
        self.problem_type = "classification"

    def log_results(self, logger, model, X_test=None, y_test=None):
        """
        Log custom classification report for the time-series derived labels.
        """
        super().log_results(logger, model, X_test, y_test)

        if hasattr(self, "y_true_all") and hasattr(self, "y_pred_all"):
            print("\n--- Final Classification Report (Chronos CV Aggregated) ---")
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

class PalazzoChronosBinaryClassificationPipeline(PalazzoChronosPipeline):
    """
    Fine-tunes Chronos for binary classification by recasting the problem
    as a regression task on labels {+1.0, -1.0}.
    """

    def __init__(self, config):
        super().__init__(config)
        self.problem_type = "classification_finetune"

    def step_3_labeling_and_weighting(self, bars):
        """
        Generates binary labels for price movement.
        Target is +1.0 if next close > current close, else -1.0.
        """
        # Calculate future returns to determine the label
        returns = bars["close_price"].pct_change().shift(-1)
        # Create binary labels: +1.0 for up, -1.0 for down/same
        y = pd.Series(np.where(returns > 0, 1.0, -1.0), index=bars.index)
        y = y.iloc[:-1]  # Drop last row as it has no future

        # Use index as t1 for PurgedKFold
        t1 = pd.Series(bars.index, index=bars.index)
        return y, None, t1

    @timer
    def run_cv(self, raw_tick_data, model=None):
        """
        Executes Purged Cross-Validation, fine-tuning Chronos on the +1/-1 target.
        The model's continuous output is then thresholded at 0 to get binary predictions.
        """
        # 1. Pipeline Steps (Leverages the new labeling)
        bars = self.step_1_data_structuring(raw_tick_data)
        features = self.step_2_feature_engineering(bars)
        y, _, t1 = self.step_3_labeling_and_weighting(bars)

        # 2. Align
        common_idx = features.index.intersection(y.index).intersection(t1.index)
        features = features.loc[common_idx]
        y = y.loc[common_idx]
        t1 = t1.loc[common_idx]

        # 3. Setup CV
        n_splits = self.config.get("n_splits", 3)
        pct_embargo = self.config.get("pct_embargo", 0.0)
        cv = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)

        prediction_length = self.config.get("prediction_length", 2)
        model_path = self.config.get("chronos_model", "amazon/chronos-t5-small")

        scores = []
        all_y_true = []
        all_y_pred = []

        print(f"Starting Purged Cross-Validation ({n_splits} folds) for Binary Classification...")

        for i, (train_idx, test_idx) in enumerate(cv.split(features, y)):
            print(f"\n--- Fold {i + 1}/{n_splits} ---")

            # --- Prepare Training Data (Identical logic to parent) ---
            gaps = np.where(np.diff(train_idx) > 1)[0]
            split_points = gaps + 1
            segments_indices = np.split(train_idx, split_points)

            train_items = []
            for seg_i, seg_idx in enumerate(segments_indices):
                if len(seg_idx) == 0: continue
                X_seg, y_seg = features.iloc[seg_idx], y.iloc[seg_idx]
                seg_df = X_seg.copy()
                seg_df["target"] = y_seg
                seg_df["item_id"] = f"train_fold_{i}_seg_{seg_i}"
                start_date = pd.Timestamp("2000-01-01")
                seg_timestamps = pd.date_range(start=start_date, periods=len(seg_df), freq="min")
                seg_df["timestamp"] = seg_timestamps
                train_items.append(seg_df)

            train_df = pd.concat(train_items)
            ts_train = TimeSeriesDataFrame.from_data_frame(
                train_df, id_column="item_id", timestamp_column="timestamp"
            )

            known_covariates_names = [c for c in train_df.columns if c not in ["target", "item_id", "timestamp"]]

            # --- Fit Chronos on {+1, -1} labels ---
            fold_model_path = f"AutogluonModels/Chronos_Binary_Fold_{i}"
            if os.path.exists(fold_model_path):
                import shutil
                shutil.rmtree(fold_model_path, ignore_errors=True)

            predictor = TimeSeriesPredictor(
                prediction_length=prediction_length,
                path=fold_model_path,
                target="target",
                eval_metric="MASE",
                known_covariates_names=known_covariates_names,
                freq="min",
                verbosity=0,
            )
            predictor.fit(
                ts_train,
                hyperparameters={"Chronos": {"model_path": model_path}},
                time_limit=300,
            )

            # --- Rolling Prediction (Identical logic to parent) ---
            test_start_abs = test_idx[0]
            context_df = None
            for seg_idx, seg_df in zip(segments_indices, train_items):
                if len(seg_idx) > 0 and seg_idx[-1] < test_start_abs:
                    context_df = seg_df
            if context_df is None: context_df = train_items[0]

            X_test, y_test = features.iloc[test_idx], y.iloc[test_idx]
            test_df = X_test.copy()
            test_df["target"] = y_test
            last_ctx_ts = context_df["timestamp"].iloc[-1]
            start_test_ts = last_ctx_ts + pd.Timedelta(minutes=1)
            test_timestamps = pd.date_range(start=start_test_ts, periods=len(test_df), freq="min")
            test_df["timestamp"] = test_timestamps
            test_df["item_id"] = context_df["item_id"].iloc[0]

            combined_df = pd.concat([context_df, test_df])
            train_len, test_len = len(context_df), len(test_df)

            print(f"Generating rolling predictions for {test_len} steps...")
            fold_y_pred_continuous = []
            context_length = 512
            batch_size = 64

            for start_k in tqdm(range(0, test_len, batch_size), desc="Predicting"):
                # (Batching logic is identical to parent, so it's condensed for brevity)
                end_k = min(start_k + batch_size, test_len)
                batch_items, batch_covariates_list = [], []
                for k in range(start_k, end_k):
                    cutoff_idx = train_len + k
                    start_ctx = max(0, cutoff_idx - context_length)
                    ctx_slice = combined_df.iloc[start_ctx:cutoff_idx].copy()
                    ctx_slice["item_id"] = f"seq_{k}"
                    batch_items.append(ctx_slice)

                    future_slice = combined_df.iloc[cutoff_idx : cutoff_idx + prediction_length].copy()
                    if len(future_slice) < prediction_length: # Padding
                        missing = prediction_length - len(future_slice)
                        last_row = future_slice.iloc[[-1]] if not future_slice.empty else ctx_slice.iloc[[-1]]
                        padding = pd.concat([last_row] * missing)
                        last_ts = future_slice["timestamp"].iloc[-1] if not future_slice.empty else ctx_slice["timestamp"].iloc[-1]
                        padding["timestamp"] = pd.date_range(start=last_ts + pd.Timedelta(minutes=1), periods=missing, freq="min")
                        future_slice = pd.concat([future_slice, padding])
                    
                    future_covs = future_slice[known_covariates_names + ["timestamp"]].copy()
                    future_covs["item_id"] = f"seq_{k}"
                    batch_covariates_list.append(future_covs)
                
                batch_ts = TimeSeriesDataFrame.from_data_frame(pd.concat(batch_items), "item_id", "timestamp")
                batch_covariates = TimeSeriesDataFrame.from_data_frame(pd.concat(batch_covariates_list), "item_id", "timestamp")
                preds = predictor.predict(batch_ts, known_covariates=batch_covariates)

                for k in range(start_k, end_k):
                    item_preds = preds.loc[f"seq_{k}"]
                    fold_y_pred_continuous.append(item_preds["mean"].iloc[0])

            # --- Scoring ---
            # Convert continuous prediction to binary class
            fold_y_pred_class = (np.array(fold_y_pred_continuous) > 0).astype(int)
            # True classes are 1 if target was 1.0, else 0
            fold_y_true_class = (y_test.values > 0).astype(int)

            fold_score = f1_score(fold_y_true_class, fold_y_pred_class, average="weighted")
            acc = accuracy_score(fold_y_true_class, fold_y_pred_class)
            print(f"Fold {i + 1} F1: {fold_score:.4f}, Acc: {acc:.4f}")

            all_y_true.extend(fold_y_true_class)
            all_y_pred.extend(fold_y_pred_class)
            scores.append(fold_score)

        # Store aggregated results for final logging
        self.y_true_all = np.array(all_y_true)
        self.y_pred_all = np.array(all_y_pred)

        # Return a dummy predictor for compatibility with run_pipeline
        # The actual predictors are fold-specific and discarded.
        return "fine-tuned-classifier", scores, features, y, None, None, None

    def log_results(self, logger, model, X_test=None, y_test=None):
        """
        Log classification report for the fine-tuned model.
        """
        if hasattr(self, "y_true_all") and hasattr(self, "y_pred_all"):
            print("\n--- Final Classification Report (Fine-Tuned Chronos CV Aggregated) ---")
            from sklearn.metrics import classification_report

            print(classification_report(self.y_true_all, self.y_pred_all, target_names=['Down/Same', 'Up']))

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
    raw_data.rename(columns={VOLUME_COL: "volume", CLOSE_COL: "close"}, inplace=True)

    config = {
        "volume_threshold": 50000,
        "prediction_length": 2,
        "chronos_model": "amazon/chronos-t5-small",
        "n_splits": 3,
    }

    # --- CHOOSE THE PIPELINE TO RUN ---
    # 1. Original Forecasting
    # pipeline = PalazzoChronosPipeline(config)
    # experiment = "Chronos_Palazzo_Forecasting"

    # 2. Forecasting-to-Classification (Original method)
    # pipeline = PalazzoChronosClassificationPipeline(config)
    # experiment = "Chronos_Palazzo_ForecastToClass"

    # 3. Fine-tuning for Binary Classification (New method)
    # pipeline = PalazzoChronosBinaryClassificationPipeline(config)
    # experiment = "Chronos_Palazzo_FinetuneToClass"


    run_pipeline(
        pipeline=pipeline,
        model_cls=None,
        raw_data=raw_data,
        model_params={},
        experiment_name=experiment,
        data_path=data_path,
    )


if __name__ == "__main__":
    main()
