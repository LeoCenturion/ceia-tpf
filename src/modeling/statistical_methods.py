import os
import sys
import time
from functools import wraps

# Make the script runnable from anywhere
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from statsmodels.tsa.stattools import adfuller
from joblib import Parallel, delayed, cpu_count

from src.data_analysis.bar_aggregation import create_dollar_bars
from src.data_analysis.data_analysis import (
    fetch_historical_data,
    get_weights_ffd,
    frac_diff_ffd,
    find_minimum_d,
)
from src.data_analysis.indicators import create_features
from src.modeling import PurgedKFold
from src.constants import (
    OPEN_COL,
    HIGH_COL,
    LOW_COL,
    CLOSE_COL,
    VOLUME_COL,
    TIMESTAMP_COL,
)
from src.modeling.feature_importance import (
    feature_importance_mdi,
    feature_importance_mda,
    feature_importance_sfi,
    feature_importance_orthogonal,
    weighted_kendalls_tau,
)


def timer(func):
    """Decorator that prints the execution time of the function it decorates."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"\n>>> Function '{func.__name__}' executed in {duration:.4f} seconds")
        return result

    return wrapper


class MachineLearningPipeline:
    def __init__(self, config):
        self.config = config

    # Part I: Data Analysis
    # Step 1: Data Structuring
    @timer
    def step_1_data_structuring(self, raw_tick_data):
        """
        Generate information-driven bars (Dollar Bars).
        """
        bars = create_dollar_bars(raw_tick_data, self.config["dollar_threshold"])
        return bars

    # Step 2: Feature Engineering
    @timer
    def filter_features_whitelist(self, stationary_features, whitelist):
        """
        Filter features based on a predefined whitelist.
        """
        print("\n--- Filtering Features by Whitelist ---")

        # Check which features from the whitelist are present
        available_features = [f for f in whitelist if f in stationary_features.columns]
        missing_features = set(whitelist) - set(available_features)

        if missing_features:
            print(f"Warning: {len(missing_features)} features from whitelist not found in data: {missing_features}")

        print(f"Original Feature Count: {len(stationary_features.columns)}")
        print(f"Keeping {len(available_features)} features: {available_features}")

        return stationary_features[available_features]

    @timer
    def orthogonalize_pca(self, stationary_features, n_components=0.95):
        # Standardize features before PCA
        scaler = StandardScaler()
        stationary_features_scaled = pd.DataFrame(
            scaler.fit_transform(stationary_features),
            index=stationary_features.index,
            columns=stationary_features.columns
        )
        
        # Orthogonalize features using PCA
        pca = PCA(n_components=None, random_state=42)
        orthogonal_features_data = pca.fit_transform(stationary_features_scaled)
        orthogonal_features = pd.DataFrame(
            orthogonal_features_data,
            index=stationary_features.index,
            columns=[f"PC{i + 1}" for i in range(orthogonal_features_data.shape[1])],
        )
        return orthogonal_features, pca

    @timer
    def step_2_feature_engineering(self, bars):
        """
        Create features and make them stationary.
        """
        features = create_features(bars)
        features = features.dropna()

        # Filter features based on whitelist
        if self.config.get("feature_whitelist") is not None:
            features = self.filter_features_whitelist(features, self.config["feature_whitelist"])

        # Fractional differentiation to reach stationarity
        d_star, stationary_features = find_minimum_d(features)
        print(f'minimum d: {d_star}')
        return features, stationary_features

    # Step 3: Labeling and Weighting
    def get_daily_vol(self, close, lookback=100):
        """
        Compute daily volatility.
        """
        df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
        df0 = df0[df0 > 0]
        df0 = pd.Series(
            close.index[df0 - 1], index=close.index[close.shape[0] - len(df0) :]
        )
        df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # Daily returns
        df0 = df0.ewm(span=lookback).std()
        return df0

    def get_t1(self, close, t_events, num_days):
        """
        Get vertical barrier timestamps.
        """
        t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
        t1 = t1[t1 < close.shape[0]]
        t1 = pd.Series(close.index[t1], index=t_events[: t1.shape[0]])
        return t1

    def get_events(self, close, t_events, pt_sl, target, min_ret, t1=None):
        """
        Get Triple-Barrier events.
        """
        # Align target index with t_events index
        target = target.reindex(t_events, method="ffill")
        target = target.loc[t_events]
        target = target[target > min_ret]
        if t1 is None:
            t1 = pd.Series(pd.NaT, index=t_events)

        side_ = pd.Series(1.0, index=target.index)
        events = pd.concat({"t1": t1, "trgt": target, "side": side_}, axis=1).dropna(
            subset=["trgt"]
        )
        df0 = events[["t1"]].copy(deep=True)
        for loc, t_1 in events["t1"].fillna(close.index[-1]).items():
            path_prices = close[loc:t_1]
            path_prices = (path_prices / close[loc] - 1) * events.at[loc, "side"]
            df0.loc[loc, "sl"] = path_prices[path_prices < -pt_sl[0]].index.min()
            df0.loc[loc, "pt"] = path_prices[path_prices > pt_sl[1]].index.min()
        events["t1"] = df0.min(axis=1)
        events = events.drop(columns=["side"])
        return events

    def get_bins(self, events, close):
        """
        Generate labels from events.
        """
        events_ = events.dropna(subset=["t1"])
        px = events_.index.union(events_["t1"].values).drop_duplicates()
        px = close.reindex(px, method="ffill")

        out = pd.DataFrame(index=events_.index)
        out["ret"] = px.loc[events_["t1"].values].values / px.loc[events_.index] - 1
        out["bin"] = np.sign(out["ret"])
        out["t1"] = events_["t1"]
        return out

    def get_num_co_events(self, close_idx, t1, molecule):
        """
        Compute number of concurrent events.
        """
        t1 = t1.fillna(close_idx[-1])
        t1 = t1[t1.index.isin(molecule)]
        t1 = t1.loc[molecule]
        # count = pd.Series(0, index=close_idx[t1.index[0] : t1.max()])

        # Use searchsorted to find integer locations for slicing the DatetimeIndex
        idx_start = close_idx.searchsorted(t1.index[0])
        idx_end = close_idx.searchsorted(t1.max())

        # If the exact timestamp t1.max() exists, we want to include it.
        if idx_end < len(close_idx) and close_idx[idx_end] == t1.max():
            idx_end += 1

        count = pd.Series(0, index=close_idx[idx_start:idx_end])
        for t_in, t_out in t1.items():
            count.loc[t_in:t_out] += 1
        return count.loc[molecule]

    def get_avg_uniqueness(self, t1, num_co_events):
        """
        Compute average uniqueness.
        """
        t1 = t1.dropna()
        weights = pd.Series(index=t1.index)
        for t_in, t_out in t1.items():
            weights.loc[t_in] = (1.0 / num_co_events.loc[t_in:t_out]).mean()
        return weights

    def get_sample_weights(self, t1, num_co_events, close):
        """
        Compute sample weights by uniqueness and return.
        """
        ret = np.log(close).diff().dropna()
        weights = self.get_avg_uniqueness(t1, num_co_events)
        weights *= np.abs(ret.loc[weights.index])
        weights = weights / weights.sum()
        weights.name = "sample_weight"
        return weights

    @timer
    def step_3_labeling_and_weighting(self, bars):
        """
        Apply Triple-Barrier method, compute uniqueness, and sample weights.
        """
        close = bars[CLOSE_COL]
        vol = self.get_daily_vol(close)
        cusum_events = bars.index  # Simplified event trigger
        t1 = self.get_t1(close, cusum_events, num_days=self.config["horizon"])
        target = vol

        events = self.get_events(
            close,
            cusum_events,
            pt_sl=[self.config["pt"], self.config["sl"]],
            target=target,
            min_ret=self.config["min_ret"],
            t1=t1,
        )
        labels = self.get_bins(events, close)
        labels = labels.dropna()

        events_for_weights = events.loc[labels.index]
        num_co_events = self.get_num_co_events(
            close.index, events_for_weights["t1"], labels.index
        )
        sample_weights = self.get_sample_weights(events_for_weights["t1"], num_co_events, close)

        return labels, sample_weights

    # Part II: Modeling
    @timer
    def run(self, raw_tick_data, model):
        """
        Execute the full machine learning pipeline with leakage-free CV.
        """
        # Part I: Data Analysis
        # Step 1: Data Structuring
        bars = self.step_1_data_structuring(raw_tick_data)

        # Step 2: Feature Engineering (Stationary Features Only)
        features, stationary_features = self.step_2_feature_engineering(bars)

        # Step 3: Labeling and Weighting
        labels, sample_weights = self.step_3_labeling_and_weighting(bars)
        
        # Align data for labeling
        t1 = labels["t1"]
        
        # We first align the features with the labels/weights before splitting
        # This ensures indices match across X, y, and weights
        combined = pd.concat([labels["bin"], stationary_features, sample_weights], axis=1).dropna()
        
        # Separate back into components
        # X_raw contains the stationary features (not yet scaled/PCA'd)
        X_raw = combined[stationary_features.columns]
        y = combined["bin"]
        sample_weights_series = combined["sample_weight"]
        t1_series = t1.loc[X_raw.index]

        # Part II: Modeling
        # Manual Cross-validation with PurgedKFold
        cv = PurgedKFold(
            n_splits=self.config["n_splits"], t1=t1_series, pct_embargo=self.config["pct_embargo"]
        )

        scores = []
        
        # We need to store the columns of the PCA for consistency
        pca_columns = None

        for train_idx, test_idx in cv.split(X_raw, y, groups=t1_series):
            # 1. Split Raw Data
            X_train_raw, X_test_raw = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            sample_weight_train = sample_weights_series.iloc[train_idx]

            # 2. Fit Scaler on TRAIN only, apply to both
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw) # Leakage prevented

            # 3. Fit PCA on TRAIN only, apply to both
            # Note: We need to handle the case where n_components=None (all components)
            pca_fold = PCA(n_components=None, random_state=42)
            X_train_pca = pca_fold.fit_transform(X_train_scaled)
            X_test_pca = pca_fold.transform(X_test_scaled)

            # Convert to DataFrame to handle column selection easily
            # We name columns PC1, PC2...
            num_pcs = X_train_pca.shape[1]
            pc_cols = [f"PC{i + 1}" for i in range(num_pcs)]
            X_train_df = pd.DataFrame(X_train_pca, index=X_train_raw.index, columns=pc_cols)
            X_test_df = pd.DataFrame(X_test_pca, index=X_test_raw.index, columns=pc_cols)

            # 4. Filter PCA features based on whitelist if provided
            if self.config.get("pca_whitelist"):
                available_pcs = [c for c in self.config["pca_whitelist"] if c in X_train_df.columns]
                X_train_df = X_train_df[available_pcs]
                X_test_df = X_test_df[available_pcs]
            
            pca_columns = X_train_df.columns # Save for final consistent output

            # Use clone to ensure a fresh instance for each fold
            fold_model = clone(model)
            fold_model.fit(X_train_df, y_train, sample_weight=sample_weight_train.values)
            y_pred = fold_model.predict(X_test_df)

            scores.append(f1_score(y_test, y_pred, average="weighted"))

        # --- Final Fit on Full Dataset (for Feature Importance Analysis) ---
        # We must repeat the transformation pipeline on the FULL dataset
        # so the returned 'X' and 'pca' are consistent with the logic.
        scaler_final = StandardScaler()
        X_raw_scaled = scaler_final.fit_transform(X_raw)
        
        pca_final = PCA(n_components=None, random_state=42)
        X_pca_final = pca_final.fit_transform(X_raw_scaled)
        
        num_pcs_final = X_pca_final.shape[1]
        pc_cols_final = [f"PC{i + 1}" for i in range(num_pcs_final)]
        X_final = pd.DataFrame(X_pca_final, index=X_raw.index, columns=pc_cols_final)
        
        if self.config.get("pca_whitelist"):
             available_pcs = [c for c in self.config["pca_whitelist"] if c in X_final.columns]
             X_final = X_final[available_pcs]

        trained_model = clone(model)
        trained_model.fit(X_final, y, sample_weight=sample_weights_series.values)

        # Return the transformed X (PCA features) and the fitted PCA object
        return trained_model, scores, X_final, y, sample_weights_series, t1_series, features, pca_final


def main():
    """
    Main function to run the ML pipeline.
    """
    raw_tick_data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2020-01-01T00:00:00Z",
        end_date="2025-08-01T00:00:00Z",
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1h/BTCUSDT_consolidated_klines.csv",
    )

    print(raw_tick_data.head())
    raw_tick_data.index = pd.to_datetime(raw_tick_data.index)

    model = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1, max_features=1)
    
    feature_whitelist = [
        "avg_volume_20",
        "ADX_14",
        "Stoch_D_pct",
        "open_pct_lag_1",
        "VPT",
        "low_pct_lag_1",
        "high_pct_lag_1",
        "pct_change",
        "UO_7_14_28",
        "MACD",
        "RSI_pct",
        "close_pct_lag_3",
        "high_pct_lag_5",
        "KAMA",
        "Volume",
        "VO",
        "Stoch_K_pct",
        "run",
        "MACD_pct",
        "high_pct_lag_2",
        "CMF",
        "AROONU_14",
        "close_pct_lag_5",
        "DMP_14",
        "open_pct_lag_4",
        "high_pct_lag_3",
        "TSI_25_13",
        "above_ema_40",
        "low_pct_lag_3",
        "TSIs_25_13",
        "low_pct_lag_4",
    ]

    pca_whitelist = [
        "PC2",
        "PC3",
        "PC4",
        "PC6",
        "PC1",
        "PC5"
    ]
    config = {
        "dollar_threshold": 1e9,
        "horizon": 8,
        "pt": 1,
        "sl": 1,
        "min_ret": 0.0005,
        "n_splits": 3,
        "pct_embargo": 0.01,
        "feature_whitelist": None,
        "pca_whitelist": None
    }

    pipeline = MachineLearningPipeline(config)
    trained_model, scores, X, y, sample_weights, t1, features, pca = (
        pipeline.run(raw_tick_data, model)
    )

    print(f"Model: {trained_model}")
    print(f"Cross-validation F1 scores: {scores}")
    print(f"Average F1 score: {np.mean(scores)}")

    # --- Feature Importance Analysis ---
    print("\n--- Feature Importance Analysis ---")

    # Get PurgedKFold for MDA and SFI
    cv = PurgedKFold(
        n_splits=config["n_splits"], t1=t1, pct_embargo=config["pct_embargo"]
    )

    # 1. Mean Decrease Impurity (MDI) - on orthogonal features (PCs) used in training
    print("\n1. Mean Decrease Impurity (MDI) on orthogonal features (PCs):")
    mdi_importance = feature_importance_mdi(trained_model, X, y)
    print(mdi_importance)

    # 2. Mean Decrease Accuracy (MDA) - on orthogonal features
    print("\n2. Mean Decrease Accuracy (MDA) on orthogonal features:")
    mda_importance = feature_importance_mda(trained_model, X, y, cv, sample_weights, t1)
    print(mda_importance)

    # 2b. Mean Decrease Accuracy (MDA) - on original features
    print("\n2b. Mean Decrease Accuracy (MDA) on original features:")
    # Use a fresh model to calculate MDA on original features
    # The trained_model is fitted on orthogonal features (X), so it cannot be used here.
    mda_original_model = clone(model)
    mda_original = feature_importance_mda(
        mda_original_model,
        features.loc[X.index],
        y,
        cv,
        sample_weights,
        t1,
    )
    print(mda_original)

    # 3. Single Feature Importance (SFI) - on original features for interpretability
    # SFI on features that survived alignment with X (which are orthogonal)
    # But SFI usually wants original features.
    # We have 'features' returned from machine_learning_cycle which are the raw features.
    # However, 'features' might not align perfectly with X if rows were dropped.
    # Let's align.
    print("\n3. Single Feature Importance (SFI) on original features:")
    # Using features.loc[X.index] ensures alignment
    sfi_importance = feature_importance_sfi(
        trained_model, features.loc[X.index], y, cv, sample_weights, t1
    )
    print(sfi_importance)

    # 4. Orthogonal Feature Importance
    print("\n4. Orthogonal Feature Importance (PCA-based):")
    ortho_importance = feature_importance_orthogonal(
        trained_model, X, y, sample_weights, pca
    )
    print(ortho_importance)

    # 5. Rank Correlation between ML Importance and PCA Eigenvalues
    ml_importance = ortho_importance["Orthogonal Importance"]
    eigen_importance = ortho_importance["Explained Variance"]

    tau, p_value = weighted_kendalls_tau(ml_importance, eigen_importance)
    print("\n5. Weighted Kendall's Tau between ML Importance and PCA Eigenvalues:")
    print(f"Correlation: {tau:.4f} (p-value: {p_value:.4f})")


if __name__ == "__main__":
    main()
