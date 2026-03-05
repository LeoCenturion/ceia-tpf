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

    def _find_paths(self, splits, n_groups):
        """
        Finds all unique sets of splits that form a complete partition of the N groups.
        This is a recursive backtracking algorithm.
        """
        memo = {}

        def solve(groups_tuple, available_splits_indices):
            if not groups_tuple:
                return [[]]
            groups_tuple = tuple(sorted(groups_tuple))
            state = (groups_tuple, available_splits_indices)
            if state in memo:
                return memo[state]

            res = []
            first_group = groups_tuple[0]
            
            for i in available_splits_indices:
                split = splits[i]
                if first_group in split:
                    remaining_groups = tuple(g for g in groups_tuple if g not in split)
                    
                    # New available splits are those that do not overlap with the current split
                    new_available_splits_indices = tuple(
                        j for j in available_splits_indices if not set(splits[j]).intersection(split)
                    )
                    
                    sub_partitions = solve(remaining_groups, new_available_splits_indices)
                    for p in sub_partitions:
                        res.append([split] + p)
            
            memo[state] = res
            return res

        all_splits_indices = tuple(range(len(splits)))
        all_groups = tuple(range(n_groups))
        raw_paths = solve(all_groups, all_splits_indices)
        
        # Deduplicate paths (the solver might find the same path with splits in a different order)
        unique_paths = set()
        for p in raw_paths:
            canonical_path = tuple(sorted(p))
            unique_paths.add(canonical_path)
            
        return [list(p) for p in unique_paths]

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
        n_paths = self.config.get("n_paths", 10)
        k_test_paths = self.config.get("k_test_paths", 2)
        pct_embargo = self.config.get("pct_embargo", 0.01)

        # Time-based path splitting
        start_time, end_time = X.index.min(), X.index.max()
        path_duration = (end_time - start_time) / n_paths
        time_splits = [start_time + i * path_duration for i in range(n_paths + 1)]
        time_splits[-1] = end_time
        path_indices = [
            np.where((X.index >= time_splits[i]) & (X.index < time_splits[i+1]))[0]
            if i < n_paths - 1 else np.where((X.index >= time_splits[i]) & (X.index <= time_splits[i+1]))[0]
            for i in range(n_paths)
        ]
        
        test_path_combinations = list(combinations(range(n_paths), k_test_paths))
        logger.info(f"Total combinations to test: {len(test_path_combinations)}")

        # Step 4: Training and Forecasting for each split
        split_predictions = []
        for fold, test_path_idxs in enumerate(test_path_combinations):
            logger.info(f"--- Fold {fold + 1}/{len(test_path_combinations)} --- Test groups: {test_path_idxs}")

            train_path_idxs = np.setdiff1d(range(n_paths), test_path_idxs)
            train_indices_orig = np.concatenate([path_indices[i] for i in train_path_idxs if path_indices[i].size > 0])
            test_indices = np.concatenate([path_indices[i] for i in test_path_idxs if path_indices[i].size > 0])

            if test_indices.size == 0 or train_indices_orig.size == 0:
                logger.warning(f"Skipping fold {fold + 1} due to empty train or test set.")
                continue

            # Purging and Embargo
            test_path_time_ranges = [(X.index[path_indices[i][0]], X.index[path_indices[i][-1]]) for i in test_path_idxs if path_indices[i].size > 0]
            train_times = X.index[train_indices_orig]
            train_t1 = t1.loc[train_times]
            purge_mask = pd.Series(False, index=train_times)
            for start, end in test_path_time_ranges:
                purge_mask |= (train_times >= start) & (train_times <= end) | (train_t1 >= start) & (train_t1 <= end)
            train_indices_purged = train_indices_orig[~purge_mask.values]

            embargo_td = (end_time - start_time) * pct_embargo
            train_indices_final = train_indices_purged
            if embargo_td.total_seconds() > 0 and train_indices_purged.size > 0:
                embargo_mask = pd.Series(False, index=X.index[train_indices_purged])
                for _, end in test_path_time_ranges:
                    embargo_mask |= (X.index[train_indices_purged] > end) & (X.index[train_indices_purged] <= end + embargo_td)
                train_indices_final = train_indices_purged[~embargo_mask.values]

            if train_indices_final.size == 0:
                logger.warning(f"Skipping fold {fold + 1} as all training data was purged/embargoed.")
                continue

            X_train, X_test = X.iloc[train_indices_final], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices_final], y.iloc[test_indices]
            sw_train = sample_weights.iloc[train_indices_final]

            model = model_cls(**model_params)
            model.fit(X_train, y_train, sample_weight=sw_train)
            preds = model.predict(X_test)
            
            split_predictions.append({'test_path_idxs': test_path_idxs, 'preds': preds, 'y_test': y_test})

        # Step 5: Backtest Path Construction
        logger.info("--- Constructing and Evaluating Backtest Paths ---")
        all_preds = {p['test_path_idxs']: p for p in split_predictions}
        all_splits = list(all_preds.keys())
        
        paths = self._find_paths(all_splits, n_paths)
        logger.info(f"Constructed {len(paths)} unique backtest paths.")

        path_scores = []
        for i, path in enumerate(paths):
            path_y_true, path_y_pred = [], []
            for split_groups in path:
                if split_groups in all_preds:
                    split_results = all_preds[split_groups]
                    path_y_true.append(split_results['y_test'])
                    path_y_pred.append(split_results['preds'])
            
            if not path_y_true: continue

            path_y_true = np.concatenate(path_y_true)
            path_y_pred = np.concatenate(path_y_pred)
            
            score = f1_score(path_y_true, path_y_pred, zero_division=0)
            path_scores.append(score)
            logger.info(f"Path {i+1}/{len(paths)} F1 Score: {score:.4f}")

        logger.info("--- CPCV Path Results ---")
        if path_scores:
            logger.info(f"Individual Path F1 Scores: {[f'{s:.4f}' for s in path_scores]}")
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
    path_scores = cpcv_pipeline.run_cpcv(data, xgb.XGBClassifier, model_params)

    if not path_scores:
        logger.error("CPCV execution resulted in no valid paths. Please check data and configuration.")


if __name__ == "__main__":
    main()
