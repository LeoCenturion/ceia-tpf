import logging
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report

from itertools import combinations
import pandas as pd

from src.data_analysis.data_analysis import fetch_historical_data
from src.modeling.xgboost_pipeline_palazzo import PalazzoXGBoostPipeline
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class PalazzoXGBoostCPCVPipeline(PalazzoXGBoostPipeline):
    """
    Extends the PalazzoXGBoostPipeline to include a method for running
    Combinatorially Purged Cross-Validation.
    """

    def run_cpcv(self, raw_data, model_cls, model_params):
        """
        Runs Combinatorially Purged Cross-Validation.
        """
        # pylint: disable=too-many-locals
        logger.info("Starting CPCV process...")

        # Step 1-3: Use parent methods to get processed data
        logger.info("Step 1: Data Structuring")
        bars = self.step_1_data_structuring(raw_data)
        logger.info("Step 2: Feature Engineering")
        features = self.step_2_feature_engineering(bars)
        logger.info("Step 3: Labeling and Weighting")
        y, sample_weights, t1_from_labeling = self.step_3_labeling_and_weighting(bars)

        # Align data
        common_index = features.index.intersection(y.index)
        X = features.loc[common_index]
        y = y.loc[common_index]
        sample_weights = sample_weights.loc[common_index]
        t1 = t1_from_labeling.loc[common_index]

        logger.info(f"Data aligned. X shape: {X.shape}, y shape: {y.shape}")

        # CPCV parameters
        n_paths = self.config.get("n_paths", 10)
        k_test_paths = self.config.get("k_test_paths", 2)
        pct_embargo = self.config.get("pct_embargo", 0.01)

        logger.info(
            f"Running CPCV with n_paths={n_paths}, k_test_paths={k_test_paths}, pct_embargo={pct_embargo}"
        )

        # Split data into N paths based on time, not by number of samples
        start_time = X.index.min()
        end_time = X.index.max()
        path_duration = (end_time - start_time) / n_paths

        time_splits = [start_time + i * path_duration for i in range(n_paths + 1)]
        # Ensure the last split aligns with the actual end time to include all data
        time_splits[-1] = end_time

        path_indices = []
        for i in range(n_paths):
            start_split = time_splits[i]
            end_split = time_splits[i + 1]

            # The last path includes its end time
            if i < n_paths - 1:
                path_mask = (X.index >= start_split) & (X.index < end_split)
            else:
                path_mask = (X.index >= start_split) & (X.index <= end_split)

            indices = np.where(path_mask)[0]
            path_indices.append(indices)
        test_path_combinations = list(combinations(range(n_paths), k_test_paths))

        logger.info(f"Total combinations to test: {len(test_path_combinations)}")

        scores = []
        fold = 0
        for test_path_idxs in test_path_combinations:
            fold += 1
            logger.info(f"--- Fold {fold}/{len(test_path_combinations)} ---")
            logger.info(f"Test path indices: {test_path_idxs}")

            # Define train and test paths
            train_path_idxs = np.setdiff1d(range(n_paths), test_path_idxs)
            train_indices_orig = np.concatenate(
                [path_indices[i] for i in train_path_idxs if len(path_indices[i]) > 0]
            )
            test_indices = np.concatenate(
                [path_indices[i] for i in test_path_idxs if len(path_indices[i]) > 0]
            )

            if len(test_indices) == 0 or len(train_indices_orig) == 0:
                logger.warning(f"Skipping fold {fold} due to empty train or test set.")
                continue

            # Get time ranges for each test path for purging/embargo
            test_path_time_ranges = []
            for path_idx in test_path_idxs:
                path_inds = path_indices[path_idx]
                if len(path_inds) > 0:
                    test_path_time_ranges.append(
                        (X.index[path_inds[0]], X.index[path_inds[-1]])
                    )

            # Purging: remove training samples that overlap with test period
            train_times = X.index[train_indices_orig]
            train_t1 = t1.loc[train_times]

            purge_mask = pd.Series(False, index=train_times)
            for start_time, end_time in test_path_time_ranges:
                purge_mask |= (train_times >= start_time) & (train_times <= end_time)
                purge_mask |= (train_t1 >= start_time) & (train_t1 <= end_time)

            train_indices_purged = train_indices_orig[~purge_mask.values]

            # Embargo: remove training samples that immediately follow a test period
            embargo_td = (X.index[-1] - X.index[0]) * pct_embargo
            train_indices_final = train_indices_purged
            if embargo_td.total_seconds() > 0 and len(train_indices_purged) > 0:
                embargo_mask = pd.Series(False, index=X.index[train_indices_purged])
                for _, end_time in test_path_time_ranges:
                    embargo_start = end_time
                    embargo_end = end_time + embargo_td
                    embargo_mask |= (X.index[train_indices_purged] > embargo_start) & (
                        X.index[train_indices_purged] <= embargo_end
                    )
                train_indices_final = train_indices_purged[~embargo_mask.values]

            if len(train_indices_final) == 0:
                logger.warning(
                    f"Skipping fold {fold} as all training data was purged/embargoed."
                )
                continue

            # Final train/test sets
            X_train, X_test = (
                X.iloc[train_indices_final],
                X.iloc[test_indices],
            )
            y_train, y_test = (
                y.iloc[train_indices_final],
                y.iloc[test_indices],
            )
            sw_train = sample_weights.iloc[train_indices_final]

            logger.info(
                f"Train size: {len(X_train)} (after purge/embargo), Test size: {len(X_test)}"
            )

            model = model_cls(**model_params)
            model.fit(X_train, y_train, sample_weight=sw_train)
            preds = model.predict(X_test)
            score = f1_score(y_test, preds, zero_division=0)
            scores.append(score)

            logger.info(f"Fold {fold} F1 Score: {score:.4f}")
            logger.info(
                f"Classification report for fold {fold}:\n"
                f"{classification_report(y_test, preds, zero_division=0)}"
            )

        logger.info("--- CPCV Results ---")
        if scores:
            logger.info(f"Individual F1 Scores: {[f'{s:.4f}' for s in scores]}")
            logger.info(f"Mean F1 Score: {np.mean(scores):.4f}")
            logger.info(f"Std Dev of F1 Scores: {np.std(scores):.4f}")
        else:
            logger.warning("No scores were generated, all folds might have been skipped.")
        logger.info("CPCV process finished.")
        return np.mean(scores) if scores else 0


@setup_logging
def main():
    """
    Main function to run the CPCV backtest for the PalazzoXGBoostCPCVStrategy.
    """
    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        data_path=data_path,
        start_date="2024-01-01T00:00:00Z",  # Use a smaller subset for quicker testing
    )
    # The pipeline expects lowercase 'volume' and 'close'
    # data.rename(columns={"Volume": "volume", "Close": "close"}, inplace=True)

    logger.info(
        f"Running CPCV for Palazzo XGBoost strategy with {len(data)} datapoints"
    )

    # Configuration for the pipeline
    pipeline_config = {
        "volume_threshold": 50000,
        "tau": 0.7,
        "n_paths": 10,
        "k_test_paths": 2,
        "pct_embargo": 0.01,
    }

    # XGBoost model parameters
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

    cpcv_pipeline = PalazzoXGBoostCPCVPipeline(pipeline_config)
    cpcv_pipeline.run_cpcv(data, xgb.XGBClassifier, model_params)


if __name__ == "__main__":
    main()
