import cupy as cp
import numpy as np
from numba import njit
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.constants import (
    CLOSE_COL,
    HIGH_COL,
    LOW_COL,
    OPEN_COL,
    VOLUME_COL,
)
from src.data_analysis.data_analysis import ewm, fetch_historical_data, sma, std
from src.data_analysis.indicators import aroon, cci, rsi_indicator

# --- Part 1: Data Simulation and Volume Bar Creation ---
# The paper uses high-frequency data to construct volume bars.
# We'll simulate 1-minute data to demonstrate the process.


def awesome_oscillator(
    high: pd.Series, low: pd.Series, fast_period: int = 5, slow_period: int = 34
) -> pd.Series:
    """Calculates the Awesome Oscillator."""
    median_price = (high + low) / 2
    ao = sma(median_price, fast_period) - sma(median_price, slow_period)
    return ao


def macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """Calculates MACD, Signal Line, and Histogram."""
    ema_fast = ewm(close, span=fast_period)
    ema_slow = ewm(close, span=slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = ewm(macd_line, span=signal_period)
    histogram = macd_line - signal_line
    return pd.DataFrame({"MACD": macd_line, "Signal": signal_line, "Hist": histogram})


def mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 14
) -> pd.Series:
    """Calculates the Money Flow Index."""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_flow = pd.Series(
        np.where(typical_price > typical_price.shift(1), money_flow, 0),
        index=typical_price.index,
    )
    negative_flow = pd.Series(
        np.where(typical_price < typical_price.shift(1), money_flow, 0),
        index=typical_price.index,
    )

    positive_mf = positive_flow.rolling(window=n, min_periods=0).sum()
    negative_mf = negative_flow.rolling(window=n, min_periods=0).sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        money_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_ratio))
    mfi.replace([np.inf, -np.inf], 100, inplace=True)
    return mfi


def stochastic_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, d_n: int = 3
) -> pd.DataFrame:
    """Calculates the Stochastic Oscillator (%K and %D)."""
    low_n = low.rolling(window=n).min()
    high_n = high.rolling(window=n).max()
    k_percent = 100 * ((close - low_n) / (high_n - low_n).replace(0, 1e-9))
    d_percent = sma(k_percent, d_n)
    return pd.DataFrame({"%K": k_percent, "%D": d_percent})


# Custom implementations to replace pandas_ta
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    h_minus_l = high - low
    h_minus_pc = abs(high - close.shift(1))
    l_minus_pc = abs(low - close.shift(1))
    tr = pd.concat([h_minus_l, h_minus_pc, l_minus_pc], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    # Wilder's smoothing is equivalent to an EMA with alpha = 1/n. Span for ewm is approx 2*n - 1
    return ewm(tr, span=2 * n - 1)


def willr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    high_n = high.rolling(n).max()
    low_n = low.rolling(n).min()
    wr = -100 * (high_n - close) / (high_n - low_n).replace(0, 1e-9)
    return wr


def roc(close: pd.Series, n: int = 10) -> pd.Series:
    return (close.diff(n) / close.shift(n)).replace([np.inf, -np.inf], 0) * 100


def ultimate_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    fast: int = 7,
    medium: int = 14,
    slow: int = 28,
):
    close_prev = close.shift(1).bfill()
    bp = close - pd.concat([low, close_prev], axis=1).min(axis=1)
    tr = true_range(high, low, close)

    bp_sum_fast = bp.rolling(fast).sum()
    tr_sum_fast = tr.rolling(fast).sum()

    bp_sum_medium = bp.rolling(medium).sum()
    tr_sum_medium = tr.rolling(medium).sum()

    bp_sum_slow = bp.rolling(slow).sum()
    tr_sum_slow = tr.rolling(slow).sum()

    avg_fast = bp_sum_fast / tr_sum_fast.replace(0, 1e-9)
    avg_medium = bp_sum_medium / tr_sum_medium.replace(0, 1e-9)
    avg_slow = bp_sum_slow / tr_sum_slow.replace(0, 1e-9)

    uo = 100 * (4 * avg_fast + 2 * avg_medium + avg_slow) / (4 + 2 + 1)
    return pd.DataFrame({f"UO_{fast}_{medium}_{slow}": uo})


def true_strength_index(
    close: pd.Series, fast: int = 13, slow: int = 25, signal: int = 13
):
    pc = close.diff(1)
    pc_ema_fast = ewm(pc, span=fast)
    pc_ema_slow = ewm(pc_ema_fast, span=slow)

    apc = abs(pc)
    apc_ema_fast = ewm(apc, span=fast)
    apc_ema_slow = ewm(apc_ema_fast, span=slow)

    tsi = 100 * pc_ema_slow / apc_ema_slow.replace(0, 1e-9)
    signal_line = ewm(tsi, span=signal)
    return pd.DataFrame({f"TSI_{slow}_{fast}": tsi, f"TSIs_{slow}_{fast}": signal_line})


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0)
    minus_dm = down.where((down > up) & (down > 0), 0)

    _atr = atr(high, low, close, n)

    plus_di = 100 * ewm(plus_dm, span=2 * n - 1) / _atr.replace(0, 1e-9)
    minus_di = 100 * ewm(minus_dm, span=2 * n - 1) / _atr.replace(0, 1e-9)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9)
    adx_series = ewm(dx, span=2 * n - 1)

    return pd.DataFrame(
        {f"ADX_{n}": adx_series, f"DMP_{n}": plus_di, f"DMN_{n}": minus_di}
    )



def _stochastic_series(series: pd.Series, n: int) -> pd.Series:
    low_n = series.rolling(window=n).min()
    high_n = series.rolling(window=n).max()
    stoch = 100 * ((series - low_n) / (high_n - low_n).replace(0, 1e-9))
    return stoch


def stc(
    close: pd.Series, fast_span=23, slow_span=50, stoch_n=10, signal_span=3
) -> pd.DataFrame:
    ema_fast = ewm(close, span=fast_span)
    ema_slow = ewm(close, span=slow_span)
    macd_line = ema_fast - ema_slow

    stoch_k = _stochastic_series(macd_line, n=stoch_n)
    stoch_d = ewm(stoch_k, span=signal_span)

    stoch_k2 = _stochastic_series(stoch_d, n=stoch_n)
    stoch_d2 = ewm(stoch_k2, span=signal_span)
    return pd.DataFrame(
        {
            f"STC_{stoch_n}_{fast_span}_{slow_span}": stoch_d2,
            f"STCst_{stoch_n}_{fast_span}_{slow_span}": stoch_k2,
        }
    )


def vortex(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    tr = true_range(high, low, close)

    vmp = abs(high - low.shift(1))
    vmm = abs(low - high.shift(1))

    vmp_sum = vmp.rolling(n).sum()
    vmm_sum = vmm.rolling(n).sum()
    tr_sum = tr.rolling(n).sum()

    vip = vmp_sum / tr_sum.replace(0, 1e-9)
    vim = vmm_sum / tr_sum.replace(0, 1e-9)

    return pd.DataFrame({f"VTXP_{n}": vip, f"VTXM_{n}": vim})


def bollinger_bands(close: pd.Series, n: int = 20, std_dev: float = 2.0):
    sma_val = sma(close, n)
    std_val = std(close, n)
    upper = sma_val + std_dev * std_val
    lower = sma_val - std_dev * std_val
    bbp = (close - lower) / (upper - lower).replace(0, 1e-9)
    return pd.DataFrame(
        {
            f"BBP_{n}_{std_dev}": bbp,
            f"BBU_{n}_{std_dev}": upper,
            f"BBL_{n}_{std_dev}": lower,
        }
    )


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n_ema: int = 20,
    n_atr: int = 20,
    multiplier: float = 2.0,
):
    ema_val = ewm(close, span=n_ema)
    atr_val = atr(high, low, close, n=n_atr)
    upper = ema_val + multiplier * atr_val
    lower = ema_val - multiplier * atr_val
    return pd.DataFrame(
        {f"KCU_{n_ema}_{multiplier:.1f}": upper, f"KCL_{n_ema}_{multiplier:.1f}": lower}
    )


def donchian_channels(high: pd.Series, low: pd.Series, n: int = 20):
    upper = high.rolling(n).max()
    lower = low.rolling(n).min()
    return pd.DataFrame({f"DCU_{n}_{n}": upper, f"DCL_{n}_{n}": lower})


@njit(cache=True)
def _assign_bar_ids(volumes: np.ndarray, threshold: float) -> np.ndarray:
    """Assign a bar ID to each row using a resetting volume accumulator."""
    bar_ids = np.empty(len(volumes), dtype=np.int64)
    current_bar = 0
    cum_vol = 0.0
    for i in range(len(volumes)):
        bar_ids[i] = current_bar
        cum_vol += volumes[i]
        if cum_vol >= threshold:
            current_bar += 1
            cum_vol = 0.0
    return bar_ids


def aggregate_to_volume_bars(df: pd.DataFrame, volume_threshold: float = 50000) -> pd.DataFrame:
    """
    Aggregates time-series data into volume bars based on a volume threshold.
    This follows the core concept of the dissertation (Section 3.3).
    """
    bar_ids = _assign_bar_ids(df[VOLUME_COL].values.astype(float), float(volume_threshold))

    # All OHLCV aggregations in a single vectorized groupby pass
    grp = df.groupby(bar_ids)
    idx = df.index.to_series()
    volume_bars_df = pd.DataFrame(
        {
            "open_time": idx.groupby(bar_ids).first(),
            "close_time": idx.groupby(bar_ids).last(),
            "open_price": grp[OPEN_COL].first(),
            HIGH_COL: grp[HIGH_COL].max(),
            LOW_COL: grp[LOW_COL].min(),
            "close_price": grp[CLOSE_COL].last(),
            "total_volume": grp[VOLUME_COL].sum(),
        }
    )

    # intra-bar log-return std — null out the first row of each bar so
    # we don't bleed the previous bar's close into the current bar's std
    log_ret = np.log(df[CLOSE_COL] / df[CLOSE_COL].shift(1))
    bar_id_series = pd.Series(bar_ids, index=df.index)
    bar_starts = bar_id_series != bar_id_series.shift(1)
    log_ret[bar_starts] = np.nan
    intra_std = log_ret.groupby(bar_ids).std().fillna(0)
    # fillna(0) matches the original `if len(bar_log_returns) > 1 else 0` guard

    volume_bars_df["intra_bar_std"] = intra_std.values
    volume_bars_df.reset_index(drop=True, inplace=True)

    # Drop trailing incomplete bar (rows that never accumulated enough volume)
    if not volume_bars_df.empty:
        volume_bars_df = volume_bars_df[
            volume_bars_df["total_volume"] >= volume_threshold
        ]

    if volume_bars_df.empty:
        return volume_bars_df

    volume_bars_df["bar_return"] = (
        volume_bars_df["close_price"] / volume_bars_df["open_price"]
    ) - 1

    return volume_bars_df


# --- Part 2: Target Labeling and Feature Engineering ---


def create_labels(df, tau=0.35) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates target labels based on the triple-barrier method variation
    described in Section 3.3.2, Equation 3.1.
    Also returns the event end times (t1) for use in purging.
    """
    # print("Step 3: Creating target labels...")

    # The event horizon for this labeling is the next bar.
    t1 = pd.Series(df.index, index=df.index).shift(-1)

    df["label"] = 0
    # Shift returns and volatility to use future information for labeling ONLY
    df["next_bar_return"] = df["bar_return"].shift(-1)

    # Conditions for label = 1 ('top')
    # Condition 1: Next bar's return must be positive.
    cond1 = df["next_bar_return"] >= 0

    # Condition 2: Next bar's return must exceed current return + volatility threshold.
    cond2 = df["next_bar_return"] >= (df["bar_return"] + df["intra_bar_std"] * tau)

    df.loc[cond1 & cond2, "label"] = 1

    # Clean up columns used only for labeling. This drops the last row.
    df.dropna(subset=["next_bar_return"], inplace=True)
    df.drop(columns=["next_bar_return"], inplace=True)

    # Align t1 with the final dataframe index
    t1 = t1.reindex(df.index)

    # print(len(df))
    # print(
    #     f"Labeling complete. Class distribution:\n{df['label'].value_counts(normalize=True)}\n"
    # )
    return df, t1


def _create_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a set of technical analysis features based on percentage changes and raw price data.
    """
    features = pd.DataFrame(index=df.index)

    # --- Original features based on percentage changes ---
    open_pct = df[OPEN_COL].pct_change().fillna(0)
    high_pct = df[HIGH_COL].pct_change().fillna(0)
    low_pct = df[LOW_COL].pct_change().fillna(0)
    close_pct = df[CLOSE_COL].pct_change().fillna(0)
    features["pct_change"] = close_pct
    features["RSI_pct"] = rsi_indicator(close_pct, n=14)
    stoch_pct = stochastic_oscillator(high_pct, low_pct, close_pct)
    features["Stoch_K_pct"] = stoch_pct["%K"]
    features["Stoch_D_pct"] = stoch_pct["%D"]
    macd_pct_df = macd(close_pct)
    features["MACD_pct"] = macd_pct_df["MACD"]
    features["MACD_Signal_pct"] = macd_pct_df["Signal"]
    features["MACD_Hist_pct"] = macd_pct_df["Hist"]
    if VOLUME_COL in df.columns:
        features["MFI_pct"] = mfi(high_pct, low_pct, close_pct, df[VOLUME_COL], n=14)
    sma20_pct = sma(close_pct, 20)
    std20_pct = std(close_pct, 20)
    features["BB_Upper_pct"] = sma20_pct + (std20_pct * 2)
    features["BB_Lower_pct"] = sma20_pct - (std20_pct * 2)
    features["BB_Width_pct"] = (
        features["BB_Upper_pct"] - features["BB_Lower_pct"]
    ) / sma20_pct

    # Lagged pct_change features
    for lag in range(1, 6):
        features[f"open_pct_lag_{lag}"] = open_pct.shift(lag)
        features[f"high_pct_lag_{lag}"] = high_pct.shift(lag)
        features[f"low_pct_lag_{lag}"] = low_pct.shift(lag)
        features[f"close_pct_lag_{lag}"] = close_pct.shift(lag)

    # --- New features based on raw price data ---

    # Volume
    if VOLUME_COL in df.columns:
        features[VOLUME_COL] = df[VOLUME_COL]
        features["avg_volume_20"] = sma(df[VOLUME_COL], 20)

    # Momentum Indicators
    features["RSI"] = rsi_indicator(df[CLOSE_COL], n=14)
    features["AO"] = awesome_oscillator(df[HIGH_COL], df[LOW_COL])
    features["WR"] = willr(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL])
    features["ROC"] = roc(df[CLOSE_COL])
    features = pd.concat(
        [features, ultimate_oscillator(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL])],
        axis=1,
    )
    features = pd.concat([features, true_strength_index(df[CLOSE_COL])], axis=1)
    stoch_price = stochastic_oscillator(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL])
    features["Stoch_K"] = stoch_price["%K"]
    features["Stoch_D"] = stoch_price["%D"]

    # Trend Indicators
    macd_price_df = macd(df[CLOSE_COL])
    features["MACD"] = macd_price_df["MACD"]
    features["MACD_Signal"] = macd_price_df["Signal"]
    features["MACD_Hist"] = macd_price_df["Hist"]
    features = pd.concat(
        [features, adx(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL])], axis=1
    )
    features = pd.concat([features, aroon(df[HIGH_COL], df[LOW_COL])], axis=1)
    features["CCI"] = cci(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL])
    features = pd.concat([features, stc(df[CLOSE_COL])], axis=1)
    vortex_df = vortex(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL])
    features = pd.concat([features, vortex_df], axis=1)
    if "VTXP_14" in features.columns and "VTXM_14" in features.columns:
        features["VORTEX_diff"] = features["VTXP_14"] - features["VTXM_14"]

    # Fluctuation Indicators
    bbands = bollinger_bands(df[CLOSE_COL])
    if bbands is not None and not bbands.empty:
        features["BBP"] = bbands.get("BBP_20_2.0")

    keltner = keltner_channels(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL])
    if keltner is not None and not keltner.empty:
        kcu = keltner.get("KCU_20_2.0")
        kcl = keltner.get("KCL_20_2.0")
        if kcu is not None and kcl is not None:
            kc_range = kcu - kcl
            features["KCP"] = (df[CLOSE_COL] - kcl) / kc_range.replace(0, np.nan)

    donchian = donchian_channels(df[HIGH_COL], df[LOW_COL])
    if donchian is not None and not donchian.empty:
        dcu = donchian.get("DCU_20_20")
        dcl = donchian.get("DCL_20_20")
        if dcu is not None and dcl is not None:
            dc_range = dcu - dcl
            features["DCP"] = (df[CLOSE_COL] - dcl) / dc_range.replace(0, np.nan)

    # EMA features
    emas = [10, 15, 20, 30, 40, 50, 60]
    for e in emas:
        features[f"above_ema_{e}"] = (
            df[CLOSE_COL] > ewm(df[CLOSE_COL], span=e)
        ).astype(int)

    # Consecutive run feature
    signs = np.sign(close_pct)
    signs = signs.replace(0, np.nan).ffill().fillna(0).astype(int)
    blocks = signs.diff().ne(0).cumsum()
    features["run"] = signs.groupby(blocks).cumsum()

    # Fill NaN values that might have been generated
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.bfill(inplace=True)
    features.ffill(inplace=True)

    return features


def create_features(df):
    """
    Creates simple features for the model. The paper uses 796 features;
    we will simulate a few for demonstration (Section 3.3.3).
    """
    # print("Step 4: Creating features...")
    df["feature_return_lag_1"] = df["bar_return"].shift(1)
    df["feature_volatility_lag_1"] = df["intra_bar_std"].shift(1)
    df["feature_rolling_mean_return_5"] = (
        df["bar_return"].shift(1).rolling(window=5).mean()
    )
    df["feature_rolling_std_return_5"] = (
        df["bar_return"].shift(1).rolling(window=5).std()
    )

    # Add features from xgboost_price_reversal.py
    # Create a temporary df with standard column names for feature functions
    temp_df_for_features = pd.DataFrame(index=df.index)
    temp_df_for_features[OPEN_COL] = df["open_price"]
    temp_df_for_features[HIGH_COL] = df["High"]
    temp_df_for_features[LOW_COL] = df["Low"]
    temp_df_for_features[CLOSE_COL] = df["close_price"]
    temp_df_for_features[VOLUME_COL] = df["total_volume"]

    reversal_features = _create_reversal_features(temp_df_for_features)

    # Prefix and add new features to the main dataframe
    for col in reversal_features.columns:
        df[f"feature_{col}"] = reversal_features[col]

    # Drop rows with NaNs created by lagging/rolling features
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # print("Features created.\n")
    return df


def select_features(
    X: pd.DataFrame, y: pd.Series, corr_threshold=0.7, p_value_threshold=0.1
) -> list:
    """
    Selects features based on Pearson correlation and p-value.
    """
    selected_features = []
    for col in X.columns:
        # Drop rows with NaN in either column for correlation calculation
        temp_df = pd.concat([X[col], y], axis=1).dropna()
        if len(temp_df) < 2:
            continue

        corr, p_value = pearsonr(temp_df.iloc[:, 0], temp_df.iloc[:, 1])
        if abs(corr) >= corr_threshold and p_value < p_value_threshold:
            selected_features.append(col)

    # print(
    #     f"Selected {len(selected_features)} features out of {len(X.columns)} based on correlation criteria."
    # )
    return selected_features


# --- Part 3: Model Training and Evaluation ---


def manual_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    test_size: float = 0.3,
    refit_every: int = 1,
    train_window_size: int = 0,
):
    """
    Performs a manual walk-forward backtest with a sliding window.
    The model is refit every `refit_every` steps.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        model: A scikit-learn compatible classifier.
        test_size (float): The proportion of the dataset to be used for testing.
        refit_every (int): How often (in steps) to refit the model. Default is 1 (every step).
        train_window_size (int, optional): The size of the sliding training window.
                                           If None, defaults to the initial training set size.
    """
    split_index = int(len(X) * (1 - test_size))
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]
    if train_window_size == 0:
        train_window_size = len(X)  # expanding window

    y_pred = []
    scaler = StandardScaler()

    # print("Starting walk-forward backtest...")
    # Walk forward
    for i in range(len(X_test)):
        current_split_index = split_index + i

        # Periodically refit the model on a sliding window
        if i % refit_every == 0:
            print(f"Refitting model at step {i + 1}/{len(X_test)}...")

            # Define the sliding window for training
            start_index = max(0, current_split_index - train_window_size)
            X_train_current = X.iloc[start_index:current_split_index]
            y_train_current = y.iloc[start_index:current_split_index]

            # Scale training data
            X_train_current_scaled = scaler.fit_transform(X_train_current)

            # Calculate sample weights for balanced classes
            classes = np.unique(y_train_current)
            sample_weights_gpu = None
            if len(classes) > 1:
                weights = compute_class_weight(
                    class_weight="balanced", classes=classes, y=y_train_current
                )
                class_weight_dict = dict(zip(classes, weights))
                sample_weights = y_train_current.map(class_weight_dict).to_numpy()
                sample_weights_gpu = cp.asarray(sample_weights)

            # Move data to GPU and retrain model
            X_train_gpu = cp.asarray(X_train_current_scaled)
            y_train_gpu = cp.asarray(y_train_current)

            model.fit(X_train_gpu, y_train_gpu, sample_weight=sample_weights_gpu)

        # Get current test sample and scale it using the latest scaler
        X_test_current = X.iloc[current_split_index : current_split_index + 1]
        X_test_current_scaled = scaler.transform(X_test_current)

        # Move data to GPU for prediction
        X_test_gpu = cp.asarray(X_test_current_scaled)

        # Predict
        prediction = model.predict(X_test_gpu)
        y_pred.append(int(cp.asnumpy(prediction)[0]))

        if (i + 1) % 50 == 0 or (i + 1) == len(X_test):
            print(f"Processed {i + 1}/{len(X_test)} steps...")

    # Evaluate
    print("\n--- Backtest Classification Report ---")
    target_names = ["non-top (0)", "top (1)"]
    labels = [0, 1]
    report_str = classification_report(
        y_test, y_pred, labels=labels, target_names=target_names, zero_division=0
    )
    print(report_str)
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Backtest Accuracy: {accuracy:.2f}")

    return y_pred, y_test, report_dict


def main():
    from src.modeling.machine_learning.xgboost_pipeline_palazzo import (
        PalazzoXGBoostPipeline,
        XGBClassifierPalazzo,
    )
    from src.modeling.pipeline_runner import run_optuna_optimization

    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"
    raw_data = fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        data_path=data_path,
    )

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
        experiment_name="Palazzo_XGBoost_PR_Optimization",
        n_trials=30,
        data_path=data_path,
    )


if __name__ == "__main__":
    main()
