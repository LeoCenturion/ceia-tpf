import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Make the script runnable from anywhere
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_analysis.data_analysis import fetch_historical_data, timer
from src.modeling import PurgedKFold
from src.constants import (
    OPEN_COL,
    HIGH_COL,
    LOW_COL,
    CLOSE_COL,
    VOLUME_COL,
)
from src.modeling.xgboost_price_reversal_palazzo import (
    aggregate_to_volume_bars,
    _create_reversal_features,
    create_labels as palazzo_create_labels
)

# --- Pipeline Class ---

class PalazzoXGBoostPipeline:
    def __init__(self, config):
        self.config = config

    @timer
    def step_1_data_structuring(self, raw_tick_data):
        # Reuse aggregate_to_volume_bars from palazzo script
        df = aggregate_to_volume_bars(raw_tick_data, self.config["volume_threshold"])
        # Ensure index is datetime (aggregate_to_volume_bars returns RangeIndex with close_time col)
        if not df.empty and "close_time" in df.columns:
            df.set_index("close_time", inplace=True)
        return df

    @timerh
    def step_2_feature_engineering(self, bars):
        print("Step 2: Creating features...")
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
        final_features = pd.DataFrame(index=bars.index)
        for col in features.columns: 
            final_features[f"feature_{col}"] = features[col]
            
        final_features["feature_return_lag_1"] = bars["bar_return"].shift(1)
        final_features["feature_volatility_lag_1"] = bars["intra_bar_std"].shift(1)
        final_features["feature_rolling_mean_return_5"] = bars["bar_return"].shift(1).rolling(window=5).mean()
        final_features["feature_rolling_std_return_5"] = bars["bar_return"].shift(1).rolling(window=5).std()

        return final_features.dropna()

    @timer
    def step_3_labeling_and_weighting(self, bars):
        print("Step 3: Creating target labels and sample weights...")
        
        # Calculate t1 (event end time) BEFORE dropping rows in labeling
        # For this strategy, label at t depends on t+1, so event covers [t, t+1]
        t1_full = pd.Series(bars.index, index=bars.index).shift(-1)
        
        # Reuse create_labels from palazzo script
        # This function creates 'label' column and drops NaNs/unused columns inplace
        df_labeled = palazzo_create_labels(bars.copy(), tau=self.config["tau"])
        
        # Align t1 to the labeled index
        t1 = t1_full.loc[df_labeled.index]
        
        # Balanced weights
        y = df_labeled["label"]
        weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
        class_weight_dict = dict(zip(np.unique(y), weights))
        sample_weights = y.map(class_weight_dict)
        
        return df_labeled[["label"]], sample_weights, t1

    @timer
    def run(self, raw_tick_data, model_params):
        bars = self.step_1_data_structuring(raw_tick_data)
        features = self.step_2_feature_engineering(bars)
        labels, weights, t1 = self.step_3_labeling_and_weighting(bars)
        
        # Align all series
        common_idx = features.index.intersection(labels.index).intersection(weights.index).intersection(t1.index)
        X, y, sw, t1_series = features.loc[common_idx], labels.loc[common_idx, "label"], weights.loc[common_idx], t1.loc[common_idx]
        
        cv = PurgedKFold(n_splits=self.config["n_splits"], t1=t1_series, pct_embargo=self.config["pct_embargo"])
        scores = []
        
        print(f"Starting Purged Cross-Validation ({self.config['n_splits']} folds)...")
        for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            sw_train = sw.iloc[train_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Using cupy for XGBoost if device is cuda (handled by xgb internals)
            model = xgb.XGBClassifier(**model_params)
            model.fit(X_train_scaled, y_train, sample_weight=sw_train.values)
            y_pred = model.predict(X_test_scaled)
            scores.append(f1_score(y_test, y_pred, average="weighted"))
            print(f"Fold {i+1} F1: {scores[-1]:.4f}")
            
        return scores, X, y

def main():
    raw_data = fetch_historical_data(
        symbol="BTC/USDT", timeframe="1m",
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    )
    # The palazzo aggregation expects lowercase columns or specific mapped columns
    # aggregate_to_volume_bars uses 'volume' and OPEN_COL ('Open') but also 'close' explicitly.
    # fetch_historical_data returns Uppercase Columns (Open, High, Low, Close, Volume)
    raw_data.rename(columns={VOLUME_COL: "volume", CLOSE_COL: "close"}, inplace=True)
    
    config = {
        "volume_threshold": 50000, "tau": 0.7,
        "n_splits": 3, "pct_embargo": 0.01
    }
    
    model_params = {
        "objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
        "device": "cuda", "n_estimators": 100, "learning_rate": 0.1, "max_depth": 6
    }
    
    pipeline = PalazzoXGBoostPipeline(config)
    scores, X, y = pipeline.run(raw_data, model_params)
    
    print(f"\nAverage Purged CV F1 Score: {np.mean(scores):.4f}")
    print("\nFinal Classification Report (Sample Split):")
    # Simple final split for report
    split = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
