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
from src.data_analysis.data_analysis import fetch_historical_data
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
def get_weights_ffd(d, thres):
    """
    Get weights for fractional differentiation.
    """
    w, k = [1.0], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def _frac_diff_ffd_batch(df_chunk, w, width):
    """
    Helper to apply fractional differentiation on a batch of columns.
    """
    df_batch = {}
    for name in df_chunk.columns:
        series_f = df_chunk[name].ffill().dropna()
        df_ = pd.Series(0.0, index=series_f.index, name=name)

        for i in range(width, series_f.shape[0]):
            loc0, loc1 = series_f.index[i - width], series_f.index[i]
            if not np.isfinite(df_chunk.loc[loc1, name]):
                continue
            df_[loc1] = np.dot(w.T, series_f.loc[loc0:loc1])[0]
        df_batch[name] = df_.copy(deep=True)

    if not df_batch:
        return pd.DataFrame()

    return pd.concat(df_batch, axis=1)


@timer
def frac_diff_ffd(series, d, thres=1e-5):
    """
    Fractional differentiation with fixed-width window.
    Parallelized with column batching.
    """
    w = get_weights_ffd(d, thres)
    width = len(w) - 1

    n_jobs = cpu_count()
    columns = series.columns
    n_batches = min(n_jobs, len(columns))

    if n_batches < 1:
        return pd.DataFrame(index=series.index)

    col_chunks = np.array_split(columns, n_batches)
    # Use backend='multiprocessing' to avoid ResourceTracker/loky cleanup errors
    batch_results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_frac_diff_ffd_batch)(series[chunk], w, width)
        for chunk in col_chunks
    )
    df = pd.concat(batch_results, axis=1)
    return df


@timer
def _check_stationarity_batch(df_chunk):
    """
    Helper to check stationarity for a batch of columns.
    Returns a list of booleans (True if stationary).
    """
    results = []
    for col in df_chunk.columns:
        series = df_chunk[col]
        try:
            if series.nunique() <= 1:
                results.append(False)
                continue
            p_val = adfuller(series, maxlag=1, regression="c", autolag=None)[1]
            results.append(p_val < 0.05)
        except Exception:
            results.append(False)
    return results


@timer
def find_minimum_d(series):
    """
    Find minimum d for stationarity using ADF test.
    Returns optimal d and the differentiated series.
    Uses batched parallel execution for efficiency.
    """
    n_jobs = cpu_count()

    for d in np.linspace(0, 1, 11):
        d_series_df = frac_diff_ffd(series, d, thres=1e-5).dropna()
        if d_series_df.empty:
            continue

        # Split columns into batches for efficient parallel processing
        columns = d_series_df.columns
        # Handle case where n_jobs > n_columns
        n_batches = min(n_jobs, len(columns))
        col_chunks = np.array_split(columns, n_batches)

        # Execute batches in parallel
        # We pass the full dataframe subset to avoid slicing/pickling overhead repeatedly
        # Use backend='multiprocessing' to avoid ResourceTracker/loky cleanup errors
        batch_results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            delayed(_check_stationarity_batch)(d_series_df[chunk])
            for chunk in col_chunks
        )
        # Flatten results
        is_stationary = [item for sublist in batch_results for item in sublist]

        if all(is_stationary):
            return d, d_series_df  # Found a d that makes all columns stationary

    # Fallback to d=1.0
    d_series_df = frac_diff_ffd(series, 1.0, thres=1e-5).dropna()
    return 1.0, d_series_df


@timer
def orthogonalize_pca(stationary_features):
    # Orthogonalize features using PCA
    pca = PCA()
    orthogonal_features_data = pca.fit_transform(stationary_features)
    orthogonal_features = pd.DataFrame(
        orthogonal_features_data,
        index=stationary_features.index,
        columns=[f"PC{i + 1}" for i in range(orthogonal_features_data.shape[1])],
    )
    return orthogonal_features, pca


@timer
def step_2_feature_engineering(bars):
    """
    Create features, make them stationary, and orthogonalize them.
    """
    features = create_features(bars)
    features = features.dropna()

    # Fractional differentiation to reach stationarity
    # find_minimum_d now returns the d value AND the differentiated series
    d_star, stationary_features = find_minimum_d(features)
    
    # Standardize features before PCA
    # PCA is sensitive to scale; without standardization, features with larger variances (magnitudes)
    # will dominate the principal components.
    scaler = StandardScaler()
    stationary_features_scaled = pd.DataFrame(
        scaler.fit_transform(stationary_features),
        index=stationary_features.index,
        columns=stationary_features.columns
    )

    # Orthogonalize features using PCA
    pca = PCA()
    orthogonal_features_data = pca.fit_transform(stationary_features_scaled)
    orthogonal_features = pd.DataFrame(
        orthogonal_features_data,
        index=stationary_features.index,
        columns=[f"PC{i+1}" for i in range(orthogonal_features_data.shape[1])],
    )

    return features, orthogonal_features, pca


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
    # count = pd.Series(0, index=close_idx[t1.index[0] : t1.max()])

    # Use searchsorted to find integer locations for slicing the DatetimeIndex
    idx_start = close_idx.searchsorted(t1.index[0])
    idx_end = close_idx.searchsorted(t1.max())

    # Create the series using the sliced index
    # We add 1 to idx_end to include the end timestamp in the slice, mimicking inclusive slicing if needed,
    # or adjust based on exact requirements. searchsorted returns the insertion point.
    # If t1.max() is in close_idx, searchsorted returns its index (if side='left' which is default).
    # We want to include the range up to t1.max().

    # If the exact timestamp t1.max() exists, we want to include it.
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
    Execute the full machine learning pipeline.
    """
    # Part I: Data Analysis
    # Step 1: Data Structuring
    bars = step_1_data_structuring(raw_tick_data, config["dollar_threshold"])

    # Step 2: Feature Engineering
    features, orthogonal_features, pca = step_2_feature_engineering(bars)

    # Step 3: Labeling and Weighting
    labels, sample_weights = step_3_labeling_and_weighting(bars, config)

    # Align data
    combined = pd.concat([labels, orthogonal_features, sample_weights], axis=1).dropna()
    X = combined[orthogonal_features.columns]
    y = combined["bin"]
    sample_weights_series = combined["sample_weight"]
    t1_series = combined["t1"]

    # Part II: Modeling
    # Manual Cross-validation with PurgedKFold
    cv = PurgedKFold(
        n_splits=config["n_splits"], t1=t1_series, pct_embargo=config["pct_embargo"]
    )

    scores = []
    for train_idx, test_idx in cv.split(X, y, groups=t1_series):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        sample_weight_train = sample_weights_series.iloc[train_idx]

        # Ensure model is a fresh instance for each fold if it's stateful
        # For RandomForestClassifier, it's generally fine to reuse if not stateful, but good practice to clone
        # For simplicity here, we assume it's stateless or reinitialized in some way outside this loop.
        # If the model state needs to be reset, use `sklearn.base.clone(model)`.

        model.fit(X_train, y_train, sample_weight=sample_weight_train.values)
        y_pred = model.predict(X_test)

        # Calculate F1 score for the current fold
        from sklearn.metrics import f1_score

        scores.append(f1_score(y_test, y_pred, average="weighted"))

    # Fit the model on the entire dataset for feature importance analysis
    model.fit(X, y, sample_weight=sample_weights_series.values)

    return model, scores, X, y, sample_weights_series, t1_series, features, pca


@timer
def feature_importance_mdi(model, X, y):
    """
    Mean Decrease Impurity (MDI) feature importance.
    Trains a new RandomForestClassifier with max_features=1 to avoid masking effects.
    """
    # Use a new model with the specified max_features
    mdi_model = RandomForestClassifier(
        n_estimators=model.n_estimators,
        max_features=1,  # Key change for López de Prado's method
        random_state=model.random_state,
        n_jobs=model.n_jobs,
    )
    mdi_model.fit(X, y)

    importances = pd.Series(mdi_model.feature_importances_, index=X.columns, name="MDI")
    return importances.sort_values(ascending=False)


def _get_mda_score_for_feature(
    model, X_test, y_test, col, base_score, scorer, sample_weights, labels=None
):
    """Helper for MDA: Permutes a single column and returns the score difference."""
    X_test_permuted = X_test.copy()
    np.random.shuffle(X_test_permuted[col].values)

    y_pred_permuted = (
        model.predict_proba(X_test_permuted)
        if scorer == log_loss
        else model.predict(X_test_permuted)
    )

    if scorer == log_loss and labels is not None:
        permuted_score = scorer(
            y_test, y_pred_permuted, sample_weight=sample_weights, labels=labels
        )
    else:
        permuted_score = scorer(y_test, y_pred_permuted, sample_weight=sample_weights)

    return base_score - permuted_score


@timer
def feature_importance_mda(model, X, y, cv, sample_weights, t1, scoring="neg_log_loss"):
    """
    Mean Decrease Accuracy (MDA) feature importance using PurgedKFold.
    The permutation of features is parallelized.
    """
    if scoring == "neg_log_loss":
        scorer = log_loss
    else:
        # Assuming f1_score with "weighted" average
        scorer = lambda y_true, y_pred, sample_weight, labels=None: f1_score(
            y_true, y_pred, average="weighted", sample_weight=sample_weight
        )

    fold_importances = []
    for train_idx, test_idx in cv.split(X, y, groups=t1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        sample_weight_train = sample_weights.iloc[train_idx]
        sample_weight_test = sample_weights.iloc[test_idx]

        model.fit(X_train, y_train, sample_weight=sample_weight_train.values)

        # Capture classes from the trained model
        labels = model.classes_

        y_pred = (
            model.predict_proba(X_test) if scorer == log_loss else model.predict(X_test)
        )

        if scorer == log_loss:
            base_score = scorer(
                y_test, y_pred, sample_weight=sample_weight_test.values, labels=labels
            )
        else:
            base_score = scorer(y_test, y_pred, sample_weight=sample_weight_test.values)

        # Sequential evaluation of each feature to avoid multiprocessing issues
        feature_scores = []
        for col in X.columns:
            score = _get_mda_score_for_feature(
                model,
                X_test,
                y_test,
                col,
                base_score,
                scorer,
                sample_weight_test.values,
                labels=labels if scorer == log_loss else None,
            )
            feature_scores.append(score)

        fold_importances.append(pd.Series(feature_scores, index=X.columns))

    # Average importance across all folds
    importances = pd.concat(fold_importances, axis=1).mean(axis=1)

    # Add the full model score for reference (using the last fold's base score)
    importances.loc["full_model"] = base_score

    return importances.sort_values(ascending=False)


def _run_sfi_for_feature(model, X, y, col, cv, sample_weights, t1):
    """Helper for SFI: Runs a full CV for a single feature."""
    scores = []
    for train_idx, test_idx in cv.split(X, y, groups=t1):
        X_train, X_test = X[[col]].iloc[train_idx], X[[col]].iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        sample_weight_train = sample_weights.iloc[train_idx]

        model.fit(X_train, y_train, sample_weight=sample_weight_train.values)
        y_pred = model.predict(X_test)

        score = f1_score(y_test, y_pred, average="weighted")
        scores.append(score)
    return np.mean(scores)


@timer
def feature_importance_sfi(model, X, y, cv, sample_weights, t1, scoring="f1_weighted"):
    """
    Single Feature Importance (SFI) using PurgedKFold.
    """
    # Use a clean, unfitted model for each feature to avoid pickling overhead and state leakage
    base_model = clone(model)
    
    # Sequential evaluation of each feature
    scores = []
    for col in X.columns:
        score = _run_sfi_for_feature(base_model, X, y, col, cv, sample_weights, t1)
        scores.append(score)

    importances = pd.Series(scores, index=X.columns)
    return importances.sort_values(ascending=False)


@timer
def feature_importance_orthogonal(model, X_ortho, y, sample_weights, pca):
    """
    Feature importance on orthogonal features (PCA components).
    """
    model.fit(X_ortho, y, sample_weight=sample_weights.values)
    importances_ortho = pd.Series(
        model.feature_importances_, index=X_ortho.columns, name="Orthogonal Importance"
    )

    # Combine with PCA explained variance
    explained_variance = pd.Series(
        pca.explained_variance_ratio_, index=X_ortho.columns, name="Explained Variance"
    )

    results = pd.concat([importances_ortho, explained_variance], axis=1)
    results = results.sort_values(by="Orthogonal Importance", ascending=False)

    return results


from scipy.stats import weightedtau


def weighted_kendalls_tau(series1, series2):
    """
    Computes the weighted Kendall's tau rank correlation between two series.
    """
    # Align the two series by their index
    aligned_s1, aligned_s2 = series1.align(series2, join="inner")

    if aligned_s1.empty or aligned_s2.empty:
        return np.nan, np.nan

        # Scipy's weightedtau computes the weighted correlation.
        # By default, it uses a linear weighting scheme where disagreements
        # at the top of the rank are penalized more heavily.
    correlation, p_value = weightedtau(aligned_s1, aligned_s2)

    return correlation, p_value


def main():
    """
    Main function to run the ML pipeline.
    """
    raw_tick_data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        start_date="2025-06-01T00:00:00Z",
        end_date="2025-08-01T00:00:00Z",
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv",
    )

    # raw_tick_data.rename(
    #     columns={
    #         "Open": "open",
    #         "High": "high",
    #         "Low": "low",
    #         "Close": "close",
    #         "Volume": "volume",
    #     },
    #     inplace=True,
    # )
    print(raw_tick_data.head())
    raw_tick_data.index = pd.to_datetime(raw_tick_data.index)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    config = {
        "dollar_threshold": 5_000_000,
        "horizon": 5,
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
    print("\n--- Feature Importance Analysis ---")

    # Get PurgedKFold for MDA and SFI
    cv = PurgedKFold(
        n_splits=config["n_splits"], t1=t1, pct_embargo=config["pct_embargo"]
    )

    # 1. Mean Decrease Impurity (MDI) - on original features for interpretability
    # Use features.loc[X.index] to ensure we only use rows that survived alignment
    # Note: MDI is strictly for the *trained* model, which uses *orthogonal* features (X).
    # If we want MDI on *original* features, we'd need to retrain a model on them.
    # The previous code implied we could just pass features, but MDI comes from the tree structure
    # built on X (orthogonal). So MDI here is technically on Principal Components.
    # If we want MDI on original features, we must fit a new model on them.
    print("\n1. Mean Decrease Impurity (MDI) on orthogonal features (PCs):")
    mdi_importance = feature_importance_mdi(trained_model, X, y)
    print(mdi_importance)

    # 2. Mean Decrease Accuracy (MDA) - on orthogonal features
    print("\n2. Mean Decrease Accuracy (MDA) on orthogonal features:")
    mda_importance = feature_importance_mda(trained_model, X, y, cv, sample_weights, t1)
    print(mda_importance)

    # 3. Single Feature Importance (SFI) - on original features for interpretability
    # SFI is great because we can check each original feature independently.
    print("\n3. Single Feature Importance (SFI) on original features:")
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
