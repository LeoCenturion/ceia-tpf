import os
import sys

# Make the script runnable from anywhere
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from statsmodels.tsa.stattools import adfuller

from src.data_analysis.bar_aggregation import create_dollar_bars
from src.data_analysis.data_analysis import fetch_historical_data
from src.data_analysis.indicators import create_features
from src.modeling import PurgedKFold


# Part I: Data Analysis
# Step 1: Data Structuring
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


def frac_diff_ffd(series, d, thres=1e-5):
    """
    Fractional differentiation with fixed-width window.
    """
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    df = {}
    for name in series.columns:
        series_f = series[[name]].fillna(method="ffill").dropna()
        df_ = pd.Series(0, index=series_f.index, name=name)

        for i in range(width, series_f.shape[0]):
            loc0, loc1 = series_f.index[i - width], series_f.index[i]
            if not np.isfinite(series.loc[loc1, name]):
                continue
            df_[loc1] = np.dot(w.T, series_f.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def find_minimum_d(series):
    """
    Find minimum d for stationarity using ADF test.
    """
    for d in np.linspace(0, 1, 11):
        d_series = frac_diff_ffd(series, d, thres=1e-5).dropna()
        if not d_series.empty:
            p_val = adfuller(d_series, maxlag=1, regression="c", autolag=None)[1]
            if p_val < 0.05:
                return d
    return 1.0


def step_2_feature_engineering(bars):
    """
    Create features, make them stationary, and orthogonalize them.
    """
    features = create_features(bars)
    features = features.dropna()

    # Fractional differentiation to reach stationarity
    d_star = find_minimum_d(features)
    stationary_features = frac_diff_ffd(features, d_star)
    stationary_features = stationary_features.dropna()

    # Orthogonalize features using PCA
    pca = PCA()
    orthogonal_features = pca.fit_transform(stationary_features)
    orthogonal_features = pd.DataFrame(
        orthogonal_features, index=stationary_features.index
    )

    return orthogonal_features


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
    iloc = close_idx.searchsorted(np.array([t1.index[0], t1.iloc[0]]))[0]
    count = pd.Series(0, index=close_idx[iloc : close_idx.searchsorted(t1.max()) + 1])
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


def step_3_labeling_and_weighting(bars, config):
    """
    Apply Triple-Barrier method, compute uniqueness, and sample weights.
    """
    close = bars["close"]
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
    sample_weights = get_sample_weights(
        events_for_weights["t1"], num_co_events, close
    )

    return labels, sample_weights


# Part II: Modeling
def machine_learning_cycle(raw_tick_data, model, config):
    """
    Execute the full machine learning pipeline.
    """
    # Part I: Data Analysis
    # Step 1: Data Structuring
    bars = step_1_data_structuring(raw_tick_data, config["dollar_threshold"])

    # Step 2: Feature Engineering
    orthogonal_features = step_2_feature_engineering(bars)

    # Step 3: Labeling and Weighting
    labels, sample_weights = step_3_labeling_and_weighting(bars, config)

    # Align data
    combined = pd.concat(
        [labels, orthogonal_features, sample_weights], axis=1
    ).dropna()
    X = combined[orthogonal_features.columns]
    y = combined["bin"]
    sample_weights_series = combined["sample_weight"]
    t1_series = combined["t1"]

    # Part II: Modeling
    # Cross-validation with PurgedKFold
    cv = PurgedKFold(
        n_splits=config["n_splits"], t1=t1_series, pct_embargo=config["pct_embargo"]
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1_weighted",
        fit_params={"sample_weight": sample_weights_series.values},
    )

    # Train final model on all data
    model.fit(X, y, sample_weight=sample_weights_series.values)

    return model, scores


def main():
    """
    Main function to run the ML pipeline.
    """
    raw_tick_data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        # start_date="2025-09-01T00:00:00Z",
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

    trained_model, scores = machine_learning_cycle(raw_tick_data, model, config)

    print(f"Model: {trained_model}")
    print(f"Cross-validation F1 scores: {scores}")
    print(f"Average F1 score: {np.mean(scores)}")


if __name__ == "__main__":
    main()
