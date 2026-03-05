import logging
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report

from src.data_analysis.data_analysis import fetch_historical_data
from src.modeling import PurgedKFold
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
        t1 = t1_from_labeling.loc[common_index]  # Use t1 from labeling, aligned

        logger.info(f"Data aligned. X shape: {X.shape}, y shape: {y.shape}")

        # Setup PurgedKFold
        n_splits = self.config.get("n_splits", 5)
        pct_embargo = self.config.get("pct_embargo", 0.01)
        cv = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)
        logger.info(
            f"Using PurgedKFold with n_splits={n_splits}, pct_embargo={pct_embargo}"
        )

        scores = []
        fold = 0
        for train_indices, test_indices in cv.split(X):
            fold += 1
            logger.info(f"--- Fold {fold}/{n_splits} ---")
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            sw_train = sample_weights.iloc[train_indices]

            logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

            model = model_cls(**model_params)
            model.fit(X_train, y_train, sample_weight=sw_train)

            preds = model.predict(X_test)

            score = f1_score(y_test, preds)
            scores.append(score)

            logger.info(f"Fold {fold} F1 Score: {score:.4f}")
            logger.info(
                f"Classification report for fold {fold}:\n"
                f"{classification_report(y_test, preds)}"
            )

        logger.info("--- CPCV Results ---")
        logger.info(f"Individual F1 Scores: {[f'{s:.4f}' for s in scores]}")
        logger.info(f"Mean F1 Score: {np.mean(scores):.4f}")
        logger.info(f"Std Dev of F1 Scores: {np.std(scores):.4f}")
        logger.info("CPCV process finished.")
        return np.mean(scores)


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
        "n_splits": 5,
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
