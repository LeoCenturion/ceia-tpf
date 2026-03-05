import logging
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report

import pandas as pd

from src.backtesting.cpcv import (
    construct_backtest_paths,
    generate_combinatorial_splits,
    purge_and_embargo_split,
    time_based_partition,
)
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
        Runs Combinatorially Purged Cross-Validation and constructs backtest paths.
        """
        # pylint: disable=too-many-locals
        logger.info("Starting CPCV process...")

        # Steps 1-3: Data processing
        bars = self.step_1_data_structuring(raw_data)
        features = self.step_2_feature_engineering(bars)
        y, sample_weights, t1_from_labeling = self.step_3_labeling_and_weighting(bars)

        common_index = features.index.intersection(y.index)
        X, y, sample_weights, t1 = (
            features.loc[common_index],
            y.loc[common_index],
            sample_weights.loc[common_index],
            t1_from_labeling.loc[common_index],
        )
        logger.info(f"Data aligned. X shape: {X.shape}, y shape: {y.shape}")

        # CPCV parameters
        n_groups = self.config.get("n_groups", 10)
        k_test_groups = self.config.get("k_test_groups", 2)
        pct_embargo = self.config.get("pct_embargo", 0.01)

        # Step 1: Data Partitioning (Time-based)
        path_indices = time_based_partition(X.index, n_groups)

        # Step 2: Combinatorial Splitting
        splits = generate_combinatorial_splits(n_groups, k_test_groups)
        logger.info(f"Total combinations to test: {len(splits)}")

        # Step 3 & 4: Purging, Embargoing, Training, and Forecasting
        split_predictions = []
        for fold, (train_group_idxs, test_group_idxs) in enumerate(splits):
            logger.info(
                f"--- Fold {fold + 1}/{len(splits)} --- Test groups: {test_group_idxs}"
            )

            train_indices, test_indices = purge_and_embargo_split(
                X, t1, path_indices, train_group_idxs, test_group_idxs, pct_embargo
            )

            if test_indices.size == 0 or train_indices.size == 0:
                logger.warning(
                    f"Skipping fold {fold + 1} due to empty train or test set after purging."
                )
                continue

            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            sw_train = sample_weights.iloc[train_indices]

            model = model_cls(**model_params)
            model.fit(X_train, y_train, sample_weight=sw_train)
            preds = model.predict(X_test)

            split_predictions.append(
                {
                    "test_path_idxs": test_group_idxs,
                    "preds": preds,
                    "y_test": y_test,
                }
            )

        # Step 5: Backtest Path Construction
        logger.info("--- Constructing and Evaluating Backtest Paths ---")
        path_results = construct_backtest_paths(split_predictions, n_groups)

        path_scores = []
        for i, result in enumerate(path_results):
            score = f1_score(result["y_true"], result["y_pred"], zero_division=0)
            path_scores.append(score)
            logger.info(f"Path {i+1}/{len(path_results)} F1 Score: {score:.4f}")

        logger.info("--- CPCV Path Results ---")
        if path_scores:
            logger.info(
                f"Individual Path F1 Scores: {[f'{s:.4f}' for s in path_scores]}"
            )
            logger.info(f"Mean Path F1 Score: {np.mean(path_scores):.4f}")
            logger.info(f"Std Dev of Path F1 Scores: {np.std(path_scores):.4f}")
        else:
            logger.warning("No complete backtest paths were evaluated.")

        logger.info("CPCV process finished.")
        return path_scores


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
        "n_groups": 10,
        "k_test_groups": 2,
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
    path_scores = cpcv_pipeline.run_cpcv(data, xgb.XGBClassifier, model_params)

    if not path_scores:
        logger.error("CPCV execution resulted in no valid paths. Please check data and configuration.")


if __name__ == "__main__":
    main()
