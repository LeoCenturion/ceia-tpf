import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
from typing import cast
from src.constants import (
    CLOSE_COL,
    HIGH_COL,
    LOW_COL,
    OPEN_COL,
    VOLUME_COL,
)
from src.data_analysis.data_analysis import fetch_historical_data, timer
from src.modeling.pipeline import AbstractMLPipeline
from src.modeling.pipeline_runner import run_optuna_optimization, run_pipeline
from src.modeling.machine_learning.xgboost_price_reversal_palazzo import (
    _create_reversal_features,
    aggregate_to_volume_bars,
)
from src.modeling.machine_learning.xgboost_price_reversal_palazzo import (
    create_labels as palazzo_create_labels,
)

# --- Pipeline Class ---


class PalazzoXGBoostPipeline(AbstractMLPipeline):
    def __init__(self, config):
        super().__init__(config)

    @timer
    def step_1_data_structuring(self, raw_tick_data) -> pd.DataFrame:
        # Reuse aggregate_to_volume_bars from palazzo script
        df = aggregate_to_volume_bars(raw_tick_data, self.config["volume_threshold"])
        # Ensure index is datetime (aggregate_to_volume_bars returns RangeIndex with close_time col)
        if not df.empty and "close_time" in df.columns:
            df.set_index("close_time", inplace=True)
        return df

    @timer
    def step_2_feature_engineering(self, bars) -> pd.DataFrame:
        # print("Step 2: Creating features...")
        # Prepare temp df with standard column names for the shared feature creator
        temp_df = pd.DataFrame(index=bars.index)
        temp_df[OPEN_COL] = bars["open_price"]
        temp_df[HIGH_COL] = bars["High"]
        temp_df[LOW_COL] = bars["Low"]
        temp_df[CLOSE_COL] = bars["close_price"]
        if "total_volume" in bars.columns:
            temp_df[VOLUME_COL] = bars["total_volume"]

        # Reuse _create_reversal_features from palazzo script
        features = _create_reversal_features(temp_df)

        # Replicate feature post-processing from palazzo's create_features
        # but maintaining the index for PurgedKFold compatibility
        final_features = features.add_prefix("feature_")

        final_features["feature_return_lag_1"] = bars["bar_return"].shift(1)
        final_features["feature_volatility_lag_1"] = bars["intra_bar_std"].shift(1)
        final_features["feature_rolling_mean_return_5"] = (
            bars["bar_return"].shift(1).rolling(window=5).mean()
        )
        final_features["feature_rolling_std_return_5"] = (
            bars["bar_return"].shift(1).rolling(window=5).std()
        )

        return final_features.dropna()

    @classmethod
    def get_optuna_params(cls, trial) -> dict:
        return {
            "volume_threshold": trial.suggest_int("volume_threshold", 25000, 75000),
            "tau": trial.suggest_float("tau", 0.7, 1.3),
        }

    @timer
    def step_3_labeling_and_weighting(
        self, bars
    ) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
        # print("Step 3: Creating target labels and sample weights...")

        # The `palazzo_create_labels` function is the correct source for the event
        # end times (t1). It must be modified to return the `t1` series along
        # with the labeled dataframe.
        df_labeled, t1 = palazzo_create_labels(bars.copy(), tau=self.config["tau"])

        # Balanced weights
        y: pd.Series = cast(pd.Series, df_labeled["label"])
        weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
        class_weight_dict = dict(zip(np.unique(y), weights))
        sample_weights = y.map(class_weight_dict)

        return y, sample_weights, t1


class XGBClassifierPalazzo(xgb.XGBClassifier):
    @classmethod
    def get_optuna_params(cls, trial) -> dict:
        return {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "device": "cuda",
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "seed": 42,
        }


def main():
    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    raw_data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        data_path=data_path,
    )
    raw_data.rename(columns={VOLUME_COL: "volume", CLOSE_COL: "close"}, inplace=True)

    config = {
        "n_splits": 3,
        "pct_embargo": 0.01,
        "use_pca": True,
        "pca_components": 0.95,
    }

    run_optuna_optimization(
        pipeline_cls=PalazzoXGBoostPipeline,
        model_cls=XGBClassifierPalazzo,
        raw_data=raw_data,
        pipeline_config=config,
        experiment_name="Palazzo_XGBoost_Optimization",
        n_trials=30,
        run_name_prefix="palazzo_xgb",
        data_path=data_path,
    )


if __name__ == "__main__":
    main()
