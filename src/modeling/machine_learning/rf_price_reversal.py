import argparse
from typing import cast

import mplfinance as mpf
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

from src.constants import (
    CLOSE_COL,
    HIGH_COL,
    LOW_COL,
    OPEN_COL,
    VOLUME_COL,
)
from src.data_analysis.data_analysis import ewm, fetch_historical_data, sma, std
from src.data_analysis.indicators import rsi_indicator
from src.modeling.machine_learning.xgboost_price_reversal_palazzo import (
    _create_reversal_features,
    aggregate_to_volume_bars,
)
from src.modeling.pipeline import AbstractMLPipeline
from src.modeling.pipeline_runner import run_optuna_optimization, run_pipeline


def awesome_oscillator(
    high: pd.Series, low: pd.Series, fast_period: int = 5, slow_period: int = 34
) -> pd.Series:
    median_price = (high + low) / 2
    ao = sma(median_price, fast_period) - sma(median_price, slow_period)
    return ao


def macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    ema_fast = ewm(close, span=fast_period)
    ema_slow = ewm(close, span=slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = ewm(macd_line, span=signal_period)
    histogram = macd_line - signal_line
    return pd.DataFrame({"MACD": macd_line, "Signal": signal_line, "Hist": histogram})


def mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 14
) -> pd.Series:
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
        mfi_series = 100 - (100 / (1 + money_ratio))
    mfi_series.replace([np.inf, -np.inf], 100, inplace=True)
    return mfi_series


def stochastic_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, d_n: int = 3
) -> pd.DataFrame:
    low_n = low.rolling(window=n).min()
    high_n = high.rolling(window=n).max()
    k_percent = 100 * ((close - low_n) / (high_n - low_n).replace(0, 1e-9))
    d_percent = sma(k_percent, d_n)
    return pd.DataFrame({"%K": k_percent, "%D": d_percent})


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    h_minus_l = high - low
    h_minus_pc = abs(high - close.shift(1))
    l_minus_pc = abs(low - close.shift(1))
    return pd.concat([h_minus_l, h_minus_pc, l_minus_pc], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return ewm(tr, span=2 * n - 1)


def willr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    high_n = high.rolling(n).max()
    low_n = low.rolling(n).min()
    return -100 * (high_n - close) / (high_n - low_n).replace(0, 1e-9)


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
    close_prev = close.shift(1).fillna(method="bfill")
    bp = close - pd.concat([low, close_prev], axis=1).min(axis=1)
    tr = true_range(high, low, close)

    avg_fast = bp.rolling(fast).sum() / tr.rolling(fast).sum().replace(0, 1e-9)
    avg_medium = bp.rolling(medium).sum() / tr.rolling(medium).sum().replace(0, 1e-9)
    avg_slow = bp.rolling(slow).sum() / tr.rolling(slow).sum().replace(0, 1e-9)

    uo = 100 * (4 * avg_fast + 2 * avg_medium + avg_slow) / (4 + 2 + 1)
    return pd.DataFrame({f"UO_{fast}_{medium}_{slow}": uo})


def true_strength_index(
    close: pd.Series, fast: int = 13, slow: int = 25, signal: int = 13
):
    pc = close.diff(1)
    pc_ema_slow = ewm(ewm(pc, span=fast), span=slow)
    apc_ema_slow = ewm(ewm(abs(pc), span=fast), span=slow)

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
    return pd.DataFrame(
        {f"ADX_{n}": ewm(dx, span=2 * n - 1), f"DMP_{n}": plus_di, f"DMN_{n}": minus_di}
    )


def aroon(high: pd.Series, low: pd.Series, n: int = 14) -> pd.DataFrame:
    periods_since_high = high.rolling(n).apply(lambda x: n - 1 - np.argmax(x), raw=True)
    periods_since_low = low.rolling(n).apply(lambda x: n - 1 - np.argmin(x), raw=True)
    aroon_up = 100 * (n - periods_since_high) / n
    aroon_down = 100 * (n - periods_since_low) / n
    return pd.DataFrame({f"AROONU_{n}": aroon_up, f"AROOND_{n}": aroon_down})


def cci(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20, c: float = 0.015
) -> pd.Series:
    tp = (high + low + close) / 3
    tp_sma = sma(tp, n)
    mad = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - tp_sma) / (c * mad).replace(0, 1e-9)


def _stochastic_series(series: pd.Series, n: int) -> pd.Series:
    low_n = series.rolling(window=n).min()
    high_n = series.rolling(window=n).max()
    return 100 * ((series - low_n) / (high_n - low_n).replace(0, 1e-9))


def stc(
    close: pd.Series, fast_span=23, slow_span=50, stoch_n=10, signal_span=3
) -> pd.DataFrame:
    macd_line = ewm(close, span=fast_span) - ewm(close, span=slow_span)
    stoch_k2 = _stochastic_series(ewm(_stochastic_series(macd_line, n=stoch_n), span=signal_span), n=stoch_n)
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
    tr_sum = tr.rolling(n).sum().replace(0, 1e-9)
    return pd.DataFrame(
        {f"VTXP_{n}": vmp.rolling(n).sum() / tr_sum, f"VTXM_{n}": vmm.rolling(n).sum() / tr_sum}
    )


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


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    open_pct = df[OPEN_COL].pct_change().fillna(0)
    high_pct = df[HIGH_COL].pct_change().fillna(0)
    low_pct = df[LOW_COL].pct_change().fillna(0)
    close_pct = df["Close"].pct_change().fillna(0)
    features["pct_change"] = close_pct
    features["RSI_pct"] = rsi_indicator(close_pct, n=14)
    stoch_pct = stochastic_oscillator(high_pct, low_pct, close_pct)
    features["Stoch_K_pct"] = stoch_pct["%K"]
    features["Stoch_D_pct"] = stoch_pct["%D"]
    macd_pct_df = macd(close_pct)
    features["MACD_pct"] = macd_pct_df["MACD"]
    features["MACD_Signal_pct"] = macd_pct_df["Signal"]
    features["MACD_Hist_pct"] = macd_pct_df["Hist"]
    if "Volume" in df.columns:
        features["MFI_pct"] = mfi(high_pct, low_pct, close_pct, df["Volume"], n=14)
    sma20_pct = sma(close_pct, 20)
    std20_pct = std(close_pct, 20)
    features["BB_Upper_pct"] = sma20_pct + (std20_pct * 2)
    features["BB_Lower_pct"] = sma20_pct - (std20_pct * 2)
    features["BB_Width_pct"] = (
        features["BB_Upper_pct"] - features["BB_Lower_pct"]
    ) / sma20_pct

    for lag in range(1, 6):
        features[f"open_pct_lag_{lag}"] = open_pct.shift(lag)
        features[f"high_pct_lag_{lag}"] = high_pct.shift(lag)
        features[f"low_pct_lag_{lag}"] = low_pct.shift(lag)
        features[f"close_pct_lag_{lag}"] = close_pct.shift(lag)

    if "Volume" in df.columns:
        features["Volume"] = df["Volume"]
        features["avg_volume_20"] = sma(df["Volume"], 20)

    features["RSI"] = rsi_indicator(df["Close"], n=14)
    features["AO"] = awesome_oscillator(df[HIGH_COL], df[LOW_COL])
    features["WR"] = willr(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL])
    features["ROC"] = roc(df["Close"])
    features = pd.concat(
        [features, ultimate_oscillator(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL])],
        axis=1,
    )
    features = pd.concat([features, true_strength_index(df["Close"])], axis=1)
    stoch_price = stochastic_oscillator(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL])
    features["Stoch_K"] = stoch_price["%K"]
    features["Stoch_D"] = stoch_price["%D"]

    macd_price_df = macd(df["Close"])
    features["MACD"] = macd_price_df["MACD"]
    features["MACD_Signal"] = macd_price_df["Signal"]
    features["MACD_Hist"] = macd_price_df["Hist"]
    features = pd.concat(
        [features, adx(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL])], axis=1
    )
    features = pd.concat([features, aroon(df[HIGH_COL], df[LOW_COL])], axis=1)
    features["CCI"] = cci(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL])
    features = pd.concat([features, stc(df["Close"])], axis=1)
    vortex_df = vortex(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL])
    features = pd.concat([features, vortex_df], axis=1)
    if "VTXP_14" in features.columns and "VTXM_14" in features.columns:
        features["VORTEX_diff"] = features["VTXP_14"] - features["VTXM_14"]

    bbands = bollinger_bands(df["Close"])
    if bbands is not None and not bbands.empty:
        features["BBP"] = bbands.get("BBP_20_2.0")

    keltner = keltner_channels(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL])
    if keltner is not None and not keltner.empty:
        kcu = keltner.get("KCU_20_2.0")
        kcl = keltner.get("KCL_20_2.0")
        if kcu is not None and kcl is not None:
            kc_range = kcu - kcl
            features["KCP"] = (df["Close"] - kcl) / kc_range.replace(0, np.nan)

    donchian = donchian_channels(df[HIGH_COL], df[LOW_COL])
    if donchian is not None and not donchian.empty:
        dcu = donchian.get("DCU_20_20")
        dcl = donchian.get("DCL_20_20")
        if dcu is not None and dcl is not None:
            dc_range = dcu - dcl
            features["DCP"] = (df["Close"] - dcl) / dc_range.replace(0, np.nan)

    for e in [10, 15, 20, 30, 40, 50, 60]:
        features[f"above_ema_{e}"] = (df["Close"] > ewm(df["Close"], span=e)).astype(int)

    signs = np.sign(close_pct)
    signs = signs.replace(0, np.nan).ffill().fillna(0).astype(int)
    blocks = signs.diff().ne(0).cumsum()
    features["run"] = signs.groupby(blocks).cumsum()

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.bfill(inplace=True)
    features.ffill(inplace=True)

    return features


def create_target_variable(
    df: pd.DataFrame,
    method: str = "ao_on_pct_change",
    peak_distance: int = 1,
    peak_threshold: float = 0,
    std_fraction: float = 1.0,
) -> pd.DataFrame:
    if method == "pct_change_std":
        window = 24 * 7
        close_pct_change = df[CLOSE_COL].pct_change()
        future_pct_change = close_pct_change.shift(-1)
        rolling_std = close_pct_change.rolling(window=window).std()

        df["target"] = 0
        df.loc[future_pct_change >= (rolling_std * std_fraction), "target"] = 1
        df.loc[future_pct_change <= -(rolling_std * std_fraction), "target"] = -1
        return df

    if method == "ao_on_pct_change":
        ao = awesome_oscillator(
            df[HIGH_COL].pct_change().fillna(0), df[LOW_COL].pct_change().fillna(0)
        )
    elif method == "ao_on_price":
        ao = awesome_oscillator(df[HIGH_COL], df[LOW_COL])
    elif method == "pct_change_on_ao":
        ao = awesome_oscillator(df[HIGH_COL], df[LOW_COL]).pct_change().fillna(0).replace(
            [np.inf, -np.inf], 0
        )
    else:
        raise ValueError(f"Invalid method '{method}' for create_target_variable")

    if ao is None or ao.isnull().all():
        df["target"] = 0
        return df

    peaks, _ = find_peaks(ao, distance=peak_distance, threshold=peak_threshold)
    troughs, _ = find_peaks(-ao, distance=peak_distance, threshold=peak_threshold)

    df["target"] = 0
    df.loc[df.index[peaks], "target"] = 1
    df.loc[df.index[troughs], "target"] = -1

    return df


def select_features(
    X: pd.DataFrame, y: pd.Series, corr_threshold=0.3, p_value_threshold=0.05
) -> list:
    selected_features = []
    for col in X.columns:
        temp_df = pd.concat([X[col], y], axis=1).dropna()
        if len(temp_df) < 2:
            continue
        corr, p_value = pearsonr(temp_df.iloc[:, 0], temp_df.iloc[:, 1])
        if abs(corr) >= corr_threshold and p_value < p_value_threshold:
            selected_features.append(col)
    print(
        f"Selected {len(selected_features)} features out of {len(X.columns)} based on correlation criteria."
    )
    return selected_features


def plot_reversals_on_candlestick(
    data: pd.DataFrame, reversal_points: pd.DataFrame, sample_size: int = None
):
    plot_data = data.copy()
    plot_reversal_points = reversal_points

    if sample_size:
        plot_data = plot_data.tail(sample_size)
        plot_reversal_points = reversal_points[
            reversal_points.index >= plot_data.index[0]
        ]

    tops = plot_reversal_points[plot_reversal_points["target"] == 1]
    bottoms = plot_reversal_points[plot_reversal_points["target"] == -1]

    top_markers = pd.Series(np.nan, index=plot_data.index)
    bottom_markers = pd.Series(np.nan, index=plot_data.index)

    top_indices = tops.index.intersection(plot_data.index)
    bottom_indices = bottoms.index.intersection(plot_data.index)

    if not top_indices.empty:
        top_markers.loc[top_indices] = plot_data.loc[top_indices, HIGH_COL] * 1.01
    if not bottom_indices.empty:
        bottom_markers.loc[bottom_indices] = plot_data.loc[bottom_indices, LOW_COL] * 0.99

    addplots = [
        mpf.make_addplot(top_markers, type="scatter", marker="^", color="green", markersize=100),
        mpf.make_addplot(bottom_markers, type="scatter", marker="v", color="red", markersize=100),
    ]

    title = "Candlestick Chart with Tops and Bottoms"
    if sample_size:
        title += f" (Last {sample_size} hours)"

    mpf.plot(
        plot_data,
        type="candle",
        style="yahoo",
        title=title,
        ylabel="Price ($)",
        addplot=addplots,
        figsize=(15, 7),
        volume=True,
        panel_ratios=(3, 1),
    )


def create_labels_triple_barrier(
    df: pd.DataFrame,
    tau: float = 0.7,
    use_ao: bool = False,
    vol_window: int = 14,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Binary triple-barrier labeling (label=1 / label=0).

    At each bar t, computes a "movement" series and its rolling volatility:
      - use_ao=False: movement = close pct_change  (price momentum, like Palazzo)
      - use_ao=True:  movement = AO.diff()          (rate of change of AO momentum)

    label=1 when the next bar's movement is:
      (a) positive, AND
      (b) >= current movement + rolling_vol * tau   (clears the volatility hurdle)
    label=0 otherwise.

    For use_ao=True the signal captures momentum acceleration: the AO gaining
    ground faster than its own recent volatility — analogous to Palazzo's
    "next return exceeds current return + intra-bar vol * tau".

    Returns (labeled_df, t1) where t1 is the next bar's timestamp (1-bar horizon).
    """
    t1 = pd.Series(df.index, index=df.index).shift(-1)

    if use_ao:
        ao = awesome_oscillator(df[HIGH_COL], df[LOW_COL])
        series = ao.diff()
    else:
        series = df[CLOSE_COL].pct_change()

    rolling_vol = series.rolling(window=vol_window).std()

    df = df.copy()
    df["_next_val"] = series.shift(-1)

    df["label"] = 0
    df.loc[(df["_next_val"] >= 0) & (df["_next_val"] >= series + rolling_vol * tau), "label"] = 1

    df.dropna(subset=["_next_val"], inplace=True)
    df.drop(columns=["_next_val"], inplace=True)

    t1 = t1.reindex(df.index)
    return df, t1


# --- Pipeline ---


class RFPriceReversalPipeline(AbstractMLPipeline):
    def __init__(self, config):
        super().__init__(config)

    def step_1_data_structuring(self, raw_tick_data) -> pd.DataFrame:
        return raw_tick_data

    def step_2_feature_engineering(self, bars) -> pd.DataFrame:
        return create_features(bars).dropna()

    def step_3_labeling_and_weighting(
        self, bars: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
        peak_method = self.config.get("peak_method", "ao_on_pct_change")
        peak_distance = self.config.get("peak_distance", 1)

        df = create_target_variable(
            bars.copy(),
            method=peak_method,
            peak_distance=peak_distance,
            peak_threshold=self.config.get("peak_threshold", 0.0),
            std_fraction=self.config.get("std_fraction", 1.0),
        )
        y = cast(pd.Series, df["target"])

        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        sample_weights = y.map(dict(zip(classes, weights)))

        # For find_peaks-based methods the label at t depends on AO values up to
        # t + peak_distance bars, so t1 must reflect that full horizon so that
        # PurgedKFold purges the contaminated samples near fold boundaries.
        # pct_change_std only looks 1 bar ahead, so 1h is correct there.
        if peak_method == "pct_change_std":
            horizon = pd.Timedelta(hours=1)
        else:
            horizon = pd.Timedelta(hours=peak_distance)

        t1 = pd.Series(bars.index + horizon, index=bars.index, name="t1")
        return y, sample_weights, t1

    @classmethod
    def get_optuna_params(cls, trial) -> dict:
        peak_method = trial.suggest_categorical(
            "peak_method", ["ao_on_price", "ao_on_pct_change", "pct_change_on_ao"]
        )
        peak_distance = trial.suggest_int("peak_distance", 1, 24 * 7)
        if peak_method == "ao_on_price":
            peak_threshold = trial.suggest_float("peak_threshold", 0.0, 100.0)
        elif peak_method == "pct_change_on_ao":
            peak_threshold = trial.suggest_float("peak_threshold", 0.0, 5.0)
        else:
            peak_threshold = trial.suggest_float("peak_threshold", 0.0, 0.005)
        corr_threshold = trial.suggest_float("corr_threshold", 0.1, 0.7)
        return {
            "peak_method": peak_method,
            "peak_distance": peak_distance,
            "peak_threshold": peak_threshold,
            "corr_threshold": corr_threshold,
        }

    def cross_validation_feature_engineering(self, train, test, y_train, y_test):
        from imblearn.over_sampling import SMOTE

        corr_threshold = self.config.get("corr_threshold", 0.3)
        selected = select_features(train, y_train, corr_threshold=corr_threshold)
        if selected:
            train, test = train[selected], test[selected]

        # SMOTE on training fold only — k_neighbors capped to minority class size
        # so it works even when minority samples are very few.
        min_class_count = int(y_train.value_counts().min())
        k = min(5, min_class_count - 1)
        if k >= 1:
            sm = SMOTE(random_state=42, k_neighbors=k)
            X_res, y_res = sm.fit_resample(train.values, y_train.values)
            train = pd.DataFrame(X_res, columns=train.columns)
            y_train = pd.Series(y_res, name=y_train.name)

        return train, test, y_train, y_test


class RFTripleBarrierPipeline(RFPriceReversalPipeline):
    """
    RF pipeline with binary triple-barrier labeling on hourly OHLCV bars.

    Replaces the peak-detection labeling with a causal 1-bar-ahead barrier:
      use_ao=False  →  barrier applied to close pct_change
      use_ao=True   →  barrier applied to AO.diff() (momentum acceleration)

    t1 is exactly 1 bar ahead, so PurgedKFold purging is precise with no
    approximation needed.
    """

    @classmethod
    def get_optuna_params(cls, trial) -> dict:
        return {
            "tau": trial.suggest_float("tau", 0.3, 2.0),
            "use_ao": trial.suggest_categorical("use_ao", [True, False]),
            "vol_window": trial.suggest_int("vol_window", 5, 30),
            "corr_threshold": trial.suggest_float("corr_threshold", 0.1, 0.7),
        }

    def step_3_labeling_and_weighting(
        self, bars: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
        df, t1 = create_labels_triple_barrier(
            bars,
            tau=self.config.get("tau", 0.7),
            use_ao=self.config.get("use_ao", False),
            vol_window=self.config.get("vol_window", 14),
        )
        y = cast(pd.Series, df["label"])

        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        sample_weights = y.map(dict(zip(classes, weights)))

        return y, sample_weights, t1


class RFTripleBarrierVolumePipeline(RFTripleBarrierPipeline):
    """
    Extends RFTripleBarrierPipeline with volume bar aggregation.

    step_1 aggregates 1-minute ticks into volume bars (volume_threshold BTC units),
    step_2 adds 4 bar-intrinsic features (bar_return, intra_bar_std and their
    rolling stats) on top of the shared indicator set — identical to
    PalazzoXGBoostPipeline's feature engineering.
    step_3 delegates to RFTripleBarrierPipeline after remapping close_price → CLOSE_COL.
    """

    def step_1_data_structuring(self, raw_tick_data) -> pd.DataFrame:
        df = aggregate_to_volume_bars(raw_tick_data, self.config["volume_threshold"])
        if not df.empty and "close_time" in df.columns:
            df.set_index("close_time", inplace=True)
        return df

    def step_2_feature_engineering(self, bars) -> pd.DataFrame:
        temp_df = pd.DataFrame(index=bars.index)
        temp_df[OPEN_COL] = bars["open_price"]
        temp_df[HIGH_COL] = bars["High"]
        temp_df[LOW_COL] = bars["Low"]
        temp_df[CLOSE_COL] = bars["close_price"]
        if "total_volume" in bars.columns:
            temp_df[VOLUME_COL] = bars["total_volume"]

        features = _create_reversal_features(temp_df).add_prefix("feature_")
        features["feature_return_lag_1"] = bars["bar_return"].shift(1)
        features["feature_volatility_lag_1"] = bars["intra_bar_std"].shift(1)
        features["feature_rolling_mean_return_5"] = bars["bar_return"].shift(1).rolling(5).mean()
        features["feature_rolling_std_return_5"] = bars["bar_return"].shift(1).rolling(5).std()
        return features.dropna()

    @classmethod
    def get_optuna_params(cls, trial) -> dict:
        params = super().get_optuna_params(trial)
        params["volume_threshold"] = trial.suggest_int("volume_threshold", 25000, 100000)
        return params

    def step_3_labeling_and_weighting(self, bars):
        # Volume bars use close_price; remap to CLOSE_COL so the parent can find it.
        # HIGH_COL / LOW_COL are already the correct column names from aggregate_to_volume_bars.
        std_bars = bars.copy()
        std_bars[CLOSE_COL] = bars["close_price"]
        return super().step_3_labeling_and_weighting(std_bars)


class RFClassifier(RandomForestClassifier):
    """RandomForestClassifier with Optuna search space declared as a classmethod."""

    @classmethod
    def get_optuna_params(cls, trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 50, log=True),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "random_state": 42,
            "n_jobs": 3,
            "class_weight": "balanced",
        }


# --- Optimization ---


def run_optimization(config: dict, raw_data: pd.DataFrame, n_trials: int = 20):
    run_optuna_optimization(
        pipeline_cls=RFPriceReversalPipeline,
        model_cls=RFClassifier,
        raw_data=raw_data,
        pipeline_config=config,
        experiment_name="rf_price_reversal_optimization",
        n_trials=n_trials,
        run_name_prefix="rf_reversal",
    )


def run_optimization_triple_barrier(config: dict, raw_data: pd.DataFrame, n_trials: int = 20):
    run_optuna_optimization(
        pipeline_cls=RFTripleBarrierPipeline,
        model_cls=RFClassifier,
        raw_data=raw_data,
        pipeline_config=config,
        experiment_name="rf_triple_barrier_optimization",
        n_trials=n_trials,
        run_name_prefix="rf_triple_barrier",
    )


def run_optimization_volume_bars(config: dict, raw_data: pd.DataFrame, n_trials: int = 20):
    run_optuna_optimization(
        pipeline_cls=RFTripleBarrierVolumePipeline,
        model_cls=RFClassifier,
        raw_data=raw_data,
        pipeline_config=config,
        experiment_name="rf_volume_bars_optimization",
        n_trials=n_trials,
        run_name_prefix="rf_volume_bars",
    )


def run_single_pipeline():
    data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv"
    raw_data = fetch_historical_data(
        data_path=data_path,
        start_date="2022-01-01T00:00:00Z",
    )

    config = {
        "peak_method": "pct_change_on_ao",
        "peak_distance": 24,
        "peak_threshold": 0.19867630413848203,
        "std_fraction": 1.0,
        "corr_threshold": 0.3,
        "n_splits": 3,
        "pct_embargo": 0.01,
        "use_pca": False,
    }

    model_params = {
        "n_estimators": 91,
        "max_depth": 30,
        "min_samples_split": 12,
        "min_samples_leaf": 16,
        "random_state": 42,
        "n_jobs": 3,
        "class_weight": "balanced",
    }

    pipeline = RFPriceReversalPipeline(config)

    run_pipeline(
        pipeline=pipeline,
        model_cls=RandomForestClassifier,
        raw_data=raw_data,
        model_params=model_params,
        experiment_name="RF_PriceReversal_Pipeline",
        data_path=data_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Run RF Price Reversal Pipeline or Optuna study.")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Optuna hyperparameter optimization study.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["reversal", "triple_barrier", "volume_bars"],
        default="reversal",
        help=(
            "reversal: peak-detection 3-class labeling (default); "
            "triple_barrier: binary triple-barrier on hourly bars; "
            "volume_bars: binary triple-barrier on volume bars."
        ),
    )
    parser.add_argument(
        "--use-ao",
        action="store_true",
        help="For triple_barrier / volume_bars: apply barrier to AO.diff() instead of close pct_change.",
    )
    args = parser.parse_args()

    hourly_data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv"
    minute_data_path = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT_consolidated_klines.csv"

    if args.pipeline == "volume_bars":
        raw_data = fetch_historical_data(
            symbol="BTC/USDT", timeframe="1m", data_path=minute_data_path
        )

        config = {
            "volume_threshold": 50000,
            "tau": 0.7,
            "use_ao": args.use_ao,
            "vol_window": 14,
            "n_splits": 3,
            "pct_embargo": 0.01,
            "use_pca": False,
            "corr_threshold": 0.3,
        }
        if args.optimize:
            run_optimization_volume_bars(config, raw_data, n_trials=20)
        else:
            model_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": 3,
                "class_weight": "balanced",
            }
            run_pipeline(
                pipeline=RFTripleBarrierVolumePipeline(config),
                model_cls=RandomForestClassifier,
                raw_data=raw_data,
                model_params=model_params,
                experiment_name="RF_TripleBarrier_VolumeBars_Pipeline",
                data_path=minute_data_path,
            )

    elif args.pipeline == "triple_barrier":
        raw_data = fetch_historical_data(
            data_path=hourly_data_path, start_date="2022-01-01T00:00:00Z"
        )
        config = {
            "tau": 0.7,
            "use_ao": args.use_ao,
            "vol_window": 14,
            "n_splits": 3,
            "pct_embargo": 0.01,
            "use_pca": False,
            "corr_threshold": 0.3,
        }
        if args.optimize:
            run_optimization_triple_barrier(config, raw_data, n_trials=20)
        else:
            model_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": 3,
                "class_weight": "balanced",
            }
            run_pipeline(
                pipeline=RFTripleBarrierPipeline(config),
                model_cls=RandomForestClassifier,
                raw_data=raw_data,
                model_params=model_params,
                experiment_name="RF_TripleBarrier_Pipeline",
                data_path=hourly_data_path,
            )

    else:  # reversal
        raw_data = fetch_historical_data(
            data_path=hourly_data_path, start_date="2022-01-01T00:00:00Z"
        )
        config = {
            "n_splits": 3,
            "pct_embargo": 0.01,
            "use_pca": False,
        }
        if args.optimize:
            run_optimization(config, raw_data, n_trials=20)
        else:
            run_single_pipeline()


if __name__ == "__main__":
    main()
