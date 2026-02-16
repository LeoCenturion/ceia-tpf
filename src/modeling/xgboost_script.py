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
import xgboost as xgb
from sklearn.decomposition import PCA
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
        print(f">>> Function '{func.__name__}' executed in {duration:.4f} seconds")
        return result

    return wrapper


# Part I: Data Analysis
# Step 1: Data Structuring
@timer
def step_1_data_structuring(raw_tick_data, dollar_threshold):
    """
    Generate information-driven bars (Dollar Bars).
    """
    bars = create_dollar_bars(raw_tick_data, dollar_threshold)
    return bars


# Step 2: Feature Engineering
@timer
def filter_features_whitelist(stationary_features, whitelist):
    """
    Filter features based on a predefined whitelist.
    """
    print("--- Filtering Features by Whitelist ---")

    # Check which features from the whitelist are present
    available_features = [f for f in whitelist if f in stationary_features.columns]
    missing_features = set(whitelist) - set(available_features)

    if missing_features:
        print(f"Warning: {len(missing_features)} features from whitelist not found in data: {missing_features}")

    print(f"Original Feature Count: {len(stationary_features.columns)}")
    print(f"Keeping {len(available_features)} features: {available_features}")

    return stationary_features[available_features]


@timer
def orthogonalize_pca(stationary_features, n_components=0.95):
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
def step_2_feature_engineering(bars, feature_whitelist=None):
    """
    Create features and make them stationary.
    """
    features = create_features(bars)
    features = features.dropna()

    # Filter features based on whitelist
    if feature_whitelist is not None:
        features = filter_features_whitelist(features, feature_whitelist)

    # Fractional differentiation to reach stationarity
    d_star, stationary_features = find_minimum_d(features)
    print(f'minimum d: {d_star}')
    return features, stationary_features


# Step 3: Labeling and Weighting
def get_daily_vol(close, lookback=100):
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


def get_t1(close, t_events, num_days):
    """
    Get vertical barrier timestamps.
    """
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[: t1.shape[0]])
    return t1


def get_events(close, t_events, pt_sl, target, min_ret, t1=None):
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


def get_bins(events, close):
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


def get_num_co_events(close_idx, t1, molecule):
    """
    Compute number of concurrent events.
    """
    t1 = t1.fillna(close_idx[-1])
    t1 = t1[t1.index.isin(molecule)]
    t1 = t1.loc[molecule]
    
    # Use searchsorted to find integer locations for slicing the DatetimeIndex
    idx_start = close_idx.searchsorted(t1.index[0])
    idx_end = close_idx.searchsorted(t1.max())

    if idx_end < len(close_idx) and close_idx[idx_end] == t1.max():
        idx_end += 1

    count = pd.Series(0, index=close_idx[idx_start:idx_end])
    for t_in, t_out in t1.items():
        count.loc[t_in:t_out] += 1
    return count.loc[molecule]


def get_avg_uniqueness(t1, num_co_events):
    """
    Compute average uniqueness.
    """
    t1 = t1.dropna()
    weights = pd.Series(index=t1.index)
    for t_in, t_out in t1.items():
        weights.loc[t_in] = (1.0 / num_co_events.loc[t_in:t_out]).mean()
    return weights


def get_sample_weights(t1, num_co_events, close):
    """
    Compute sample weights by uniqueness and return.
    """
    ret = np.log(close).diff().dropna()
    weights = get_avg_uniqueness(t1, num_co_events)
    weights *= np.abs(ret.loc[weights.index])
    weights = weights / weights.sum()
    weights.name = "sample_weight"
    return weights


@timer
def step_3_labeling_and_weighting(bars, config):
    """
    Apply Triple-Barrier method, compute uniqueness, and sample weights.
    """
    close = bars[CLOSE_COL]
    vol = get_daily_vol(close)
    cusum_events = bars.index  # Simplified event trigger
    t1 = get_t1(close, cusum_events, num_days=config["horizon"])
    target = vol

    events = get_events(
        close,
        cusum_events,
        pt_sl=[config["pt"], config["sl"]],
        target=target,
        min_ret=config["min_ret"],
        t1=t1,
    )
    labels = get_bins(events, close)
    labels = labels.dropna()

    events_for_weights = events.loc[labels.index]
    num_co_events = get_num_co_events(
        close.index, events_for_weights["t1"], labels.index
    )
    sample_weights = get_sample_weights(events_for_weights["t1"], num_co_events, close)

    return labels, sample_weights


# Part II: Modeling
@timer
def machine_learning_cycle(raw_tick_data, model, config):
    """
    Execute the full machine learning pipeline with leakage-free CV.
    """
    # Part I: Data Analysis
    # Step 1: Data Structuring
    bars = step_1_data_structuring(raw_tick_data, config["dollar_threshold"])

    # Step 2: Feature Engineering (Stationary Features Only)
    features, stationary_features = step_2_feature_engineering(bars, config.get("feature_whitelist"))

    # Step 3: Labeling and Weighting
    labels, sample_weights = step_3_labeling_and_weighting(bars, config)
    
    # Align data for labeling
    t1 = labels["t1"]
    
    # We first align the features with the labels/weights before splitting
    # This ensures indices match across X, y, and weights
    combined = pd.concat([labels["bin"], stationary_features, sample_weights], axis=1).dropna()
    
    # Separate back into components
    # X_raw contains the stationary features (not yet scaled/PCA'd)
    X_raw = combined[stationary_features.columns]
    y = combined["bin"]
    
    # XGBoost Requirement: Map labels to [0, num_classes-1]
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_mapped_values = le.fit_transform(y)
    y_mapped = pd.Series(y_mapped_values, index=y.index)
    num_classes = len(le.classes_)
    print(f"Detected {num_classes} classes: {le.classes_}. Mapped to: {np.unique(y_mapped_values)}")

    # Update model num_class if it's an XGBClassifier
    if hasattr(model, "set_params"):
        if isinstance(model, xgb.XGBClassifier):
             model.set_params(num_class=num_classes)
             model.set_params(objective="multi:softmax")

    sample_weights_series = combined["sample_weight"]
    t1_series = t1.loc[X_raw.index]

    # Part II: Modeling
    # Manual Cross-validation with PurgedKFold
    cv = PurgedKFold(
        n_splits=config["n_splits"], t1=t1_series, pct_embargo=config["pct_embargo"]
    )

    scores = []
    
    # We need to store the columns of the PCA for consistency
    pca_columns = None

    for train_idx, test_idx in cv.split(X_raw, y_mapped, groups=t1_series):
        # 1. Split Raw Data
        X_train_raw, X_test_raw = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        y_train, y_test = y_mapped.iloc[train_idx], y_mapped.iloc[test_idx]
        sample_weight_train = sample_weights_series.iloc[train_idx]

        # 2. Fit Scaler on TRAIN only, apply to both
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw) # Leakage prevented

        # 3. Fit PCA on TRAIN only, apply to both
        pca_fold = PCA(n_components=None, random_state=42)
        X_train_pca = pca_fold.fit_transform(X_train_scaled)
        X_test_pca = pca_fold.transform(X_test_scaled)

        # Convert to DataFrame to handle column selection easily
        num_pcs = X_train_pca.shape[1]
        pc_cols = [f"PC{i + 1}" for i in range(num_pcs)]
        X_train_df = pd.DataFrame(X_train_pca, index=X_train_raw.index, columns=pc_cols)
        X_test_df = pd.DataFrame(X_test_pca, index=X_test_raw.index, columns=pc_cols)

        # 4. Filter PCA features based on whitelist if provided
        if config.get("pca_whitelist"):
            available_pcs = [c for c in config["pca_whitelist"] if c in X_train_df.columns]
            X_train_df = X_train_df[available_pcs]
            X_test_df = X_test_df[available_pcs]
        
        pca_columns = X_train_df.columns # Save for final consistent output

        # Use clone to ensure a fresh instance for each fold
        fold_model = clone(model)
        # We need to re-ensure num_class/objective on the clone because clone doesn't always preserve dynamic changes
        if isinstance(fold_model, xgb.XGBClassifier):
            fold_model.set_params(num_class=num_classes)
            fold_model.set_params(objective="multi:softmax")

        fold_model.fit(X_train_df, y_train, sample_weight=sample_weight_train.values)
        y_pred = fold_model.predict(X_test_df)

        scores.append(f1_score(y_test, y_pred, average="weighted"))

    # --- Final Fit on Full Dataset (for Feature Importance Analysis) ---
    scaler_final = StandardScaler()
    X_raw_scaled = scaler_final.fit_transform(X_raw)
    
    pca_final = PCA(n_components=None, random_state=42)
    X_pca_final = pca_final.fit_transform(X_raw_scaled)
    
    num_pcs_final = X_pca_final.shape[1]
    pc_cols_final = [f"PC{i + 1}" for i in range(num_pcs_final)]
    X_final = pd.DataFrame(X_pca_final, index=X_raw.index, columns=pc_cols_final)
    
    if config.get("pca_whitelist"):
         available_pcs = [c for c in config["pca_whitelist"] if c in X_final.columns]
         X_final = X_final[available_pcs]

    trained_model = clone(model)
    if isinstance(trained_model, xgb.XGBClassifier):
        trained_model.set_params(num_class=num_classes)
        trained_model.set_params(objective="multi:softmax")

    trained_model.fit(X_final, y_mapped, sample_weight=sample_weights_series.values)

    # Return y_mapped so subsequent functions know the true labels used by the model
    return trained_model, scores, X_final, y_mapped, sample_weights_series, t1_series, features, pca_final


def main():
    """
    Main function to run the ML pipeline with XGBoost.
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

    model = xgb.XGBClassifier(
        n_estimators=1000,
        random_state=42,
        n_jobs=-1,
        objective="multi:softmax",
        num_class=3,
        colsample_bytree=0.5  # Analogous to selecting subset of features like max_features in RF
    )
    
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
    }

    trained_model, scores, X, y, sample_weights, t1, features, pca = (
        machine_learning_cycle(raw_tick_data, model, config)
    )

    print(f"Model: {trained_model}")
    print(f"Cross-validation F1 scores: {scores}")
    print(f"Average F1 score: {np.mean(scores)}")

    # --- Feature Importance Analysis ---
    # print("--- Feature Importance Analysis ---")

    # # Get PurgedKFold for MDA and SFI
    # cv = PurgedKFold(
    #     n_splits=config["n_splits"], t1=t1, pct_embargo=config["pct_embargo"]
    # )

    # # 1. Mean Decrease Impurity (MDI) - on orthogonal features (PCs) used in training
    # print("1. Mean Decrease Impurity (MDI) on orthogonal features (PCs):")
    # # Note: feature_importance_mdi internally uses RandomForestClassifier.
    # # We use this as a proxy for structural importance.
    # mdi_importance = feature_importance_mdi(trained_model, X, y)
    # print(mdi_importance)

    # # 2. Mean Decrease Accuracy (MDA) - on orthogonal features
    # print("2. Mean Decrease Accuracy (MDA) on orthogonal features:")
    # mda_importance = feature_importance_mda(trained_model, X, y, cv, sample_weights, t1)
    # print(mda_importance)

    # # 2b. Mean Decrease Accuracy (MDA) - on original features
    # print("2b. Mean Decrease Accuracy (MDA) on original features:")
    # # Use a fresh model to calculate MDA on original features
    # mda_original_model = clone(model)
    # mda_original = feature_importance_mda(
    #     mda_original_model,
    #     features.loc[X.index],
    #     y,
    #     cv,
    #     sample_weights,
    #     t1,
    # )
    # print(mda_original)

    # # 3. Single Feature Importance (SFI) - on original features for interpretability
    # print("3. Single Feature Importance (SFI) on original features:")
    # # Using features.loc[X.index] ensures alignment
    # sfi_importance = feature_importance_sfi(
    #     trained_model, features.loc[X.index], y, cv, sample_weights, t1
    # )
    # print(sfi_importance)

    # # --- Ensemble Feature Selection ---
    # print("--- Ensemble Feature Selection (Original Features) ---")
    
    # # Calculate MDI on original features for comparison
    # print("Computing MDI on original features...")
    # # Again, MDI uses RF proxy
    # mdi_original_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    # mdi_original = feature_importance_mdi(mdi_original_model, features.loc[X.index], y)

    # # Consolidate scores
    # feature_board = pd.DataFrame({
    #     'MDI': mdi_original,
    #     'MDA': mda_original,
    #     'SFI': sfi_importance
    # })

    # # Fill NaNs with 0
    # feature_board = feature_board.fillna(0)

    # # Rank-based scoring (average percentile rank)
    # feature_board_rank = feature_board.rank(pct=True)
    # # Weighted Ensemble Score emphasizing Interactions (MDA) over Isolation (SFI)
    # # Weights: MDA (50%), MDI (30%), SFI (20%)
    # feature_board['Ensemble_Score'] = (
    #     feature_board_rank['MDA'] * 0.5 + 
    #     feature_board_rank['MDI'] * 0.3 + 
    #     feature_board_rank['SFI'] * 0.2
    # )

    # # Sort by Ensemble Score
    # feature_board = feature_board.sort_values(by='Ensemble_Score', ascending=False)
    
    # print("Top 20 Features by Ensemble Score (Average Percentile Rank):")
    # print(feature_board.head(20))

    # # 4. Orthogonal Feature Importance
    # print("4. Orthogonal Feature Importance (PCA-based):")
    # # This expects model.feature_importances_, which XGBClassifier has.
    # ortho_importance = feature_importance_orthogonal(
    #     trained_model, X, y, sample_weights, pca
    # )
    # print(ortho_importance)

    # # 5. Rank Correlation between ML Importance and PCA Eigenvalues
    # ml_importance = ortho_importance["Orthogonal Importance"]
    # eigen_importance = ortho_importance["Explained Variance"]

    # tau, p_value = weighted_kendalls_tau(ml_importance, eigen_importance)
    # print("5. Weighted Kendall's Tau between ML Importance and PCA Eigenvalues:")
    # print(f"Correlation: {tau:.4f} (p-value: {p_value:.4f})")


if __name__ == "__main__":
    main()
