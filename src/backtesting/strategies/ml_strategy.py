import pandas as pd
import numpy as np
from backtesting import Strategy
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from sklearn.utils.class_weight import compute_class_weight


from src.backtesting.backtesting import run_optimizations
from src.data_analysis.data_analysis import ewm, sma, std
from src.data_analysis.indicators import momentum_indicator, rsi_indicator


def _create_features(data):
    """Creates a feature matrix from the price data."""
    close = pd.Series(data.Close)

    # Technical Indicators
    rsi = rsi_indicator(close, n=14)
    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    momentum = momentum_indicator(close, window=10)

    # Bollinger Bands
    std20 = std(close, 20)
    upper_band = sma20 + (std20 * 2)
    lower_band = sma20 - (std20 * 2)

    # Price-derived features
    volatility = std(close, 20)
    lagged_return = close.pct_change(1)

    # Combine into a DataFrame for easier manipulation
    df = pd.DataFrame(
        {
            "RSI": rsi,
            "SMA20": sma20,
            "SMA50": sma50,
            "Momentum": momentum,
            "Upper_Band": upper_band,
            "Lower_Band": lower_band,
            "Volatility": volatility,
            "Lagged_Return": lagged_return,
            "Close": close,
        }
    )

    # Create relative features
    df["SMA_Ratio"] = df["SMA20"] / df["SMA50"]
    df["BB_Width"] = (df["Upper_Band"] - df["Lower_Band"]) / df["SMA20"]

    # Select and clean final features
    final_features = df[
        [
            "RSI",
            "Momentum",
            "Volatility",
            "Lagged_Return",
            "SMA_Ratio",
            "BB_Width",
            "Close",
        ]
    ]
    return final_features.bfill().ffill().values


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
        mfi_series = 100 - (100 / (1 + money_ratio))
    mfi_series.replace([np.inf, -np.inf], 100, inplace=True)
    return mfi_series


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
    close_prev = close.shift(1).fillna(method="bfill")
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
    cci_series = (tp - tp_sma) / (c * mad).replace(0, 1e-9)
    return cci_series


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


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a set of technical analysis features based on percentage changes and raw price data.
    """
    features = pd.DataFrame(index=df.index)

    # --- Original features based on percentage changes ---
    open_pct = df["Open"].pct_change().fillna(0)
    high_pct = df["High"].pct_change().fillna(0)
    low_pct = df["Low"].pct_change().fillna(0)
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

    # Lagged pct_change features
    for lag in range(1, 6):
        features[f"open_pct_lag_{lag}"] = open_pct.shift(lag)
        features[f"high_pct_lag_{lag}"] = high_pct.shift(lag)
        features[f"low_pct_lag_{lag}"] = low_pct.shift(lag)
        features[f"close_pct_lag_{lag}"] = close_pct.shift(lag)

    # --- New features based on raw price data ---

    # Volume
    if "Volume" in df.columns:
        features["Volume"] = df["Volume"]
        features["avg_volume_20"] = sma(df["Volume"], 20)

    # Momentum Indicators
    features["RSI"] = rsi_indicator(df["Close"], n=14)
    features["AO"] = awesome_oscillator(df["High"], df["Low"])
    features["WR"] = willr(df["High"], df["Low"], df["Close"])
    features["ROC"] = roc(df["Close"])
    features = pd.concat(
        [features, ultimate_oscillator(df["High"], df["Low"], df["Close"])], axis=1
    )
    features = pd.concat([features, true_strength_index(df["Close"])], axis=1)
    stoch_price = stochastic_oscillator(df["High"], df["Low"], df["Close"])
    features["Stoch_K"] = stoch_price["%K"]
    features["Stoch_D"] = stoch_price["%D"]

    # Trend Indicators
    macd_price_df = macd(df["Close"])
    features["MACD"] = macd_price_df["MACD"]
    features["MACD_Signal"] = macd_price_df["Signal"]
    features["MACD_Hist"] = macd_price_df["Hist"]
    features = pd.concat([features, adx(df["High"], df["Low"], df["Close"])], axis=1)
    features = pd.concat([features, aroon(df["High"], df["Low"])], axis=1)
    features["CCI"] = cci(df["High"], df["Low"], df["Close"])
    features = pd.concat([features, stc(df["Close"])], axis=1)
    vortex_df = vortex(df["High"], df["Low"], df["Close"])
    features = pd.concat([features, vortex_df], axis=1)
    if "VTXP_14" in features.columns and "VTXM_14" in features.columns:
        features["VORTEX_diff"] = features["VTXP_14"] - features["VTXM_14"]

    # Fluctuation Indicators
    bbands = bollinger_bands(df["Close"])
    if bbands is not None and not bbands.empty:
        features["BBP"] = bbands.get("BBP_20_2.0")

    keltner = keltner_channels(df["High"], df["Low"], df["Close"])
    if keltner is not None and not keltner.empty:
        kcu = keltner.get("KCU_20_2.0")
        kcl = keltner.get("KCL_20_2.0")
        if kcu is not None and kcl is not None:
            kc_range = kcu - kcl
            features["KCP"] = (df["Close"] - kcl) / kc_range.replace(0, np.nan)

    donchian = donchian_channels(df["High"], df["Low"])
    if donchian is not None and not donchian.empty:
        dcu = donchian.get("DCU_20_20")
        dcl = donchian.get("DCL_20_20")
        if dcu is not None and dcl is not None:
            dc_range = dcu - dcl
            features["DCP"] = (df["Close"] - dcl) / dc_range.replace(0, np.nan)

    # EMA features
    emas = [10, 15, 20, 30, 40, 50, 60]
    for e in emas:
        features[f"above_ema_{e}"] = (df["Close"] > ewm(df["Close"], span=e)).astype(
            int
        )

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


def create_target_variable(
    df: pd.DataFrame,
    method: str = "ao_on_pct_change",
    peak_distance: int = 1,
    peak_threshold: float = 0,
    std_fraction: float = 1.0,
) -> pd.DataFrame:
    """
    Identifies local tops (1), bottoms (-1), and non-reversal points (0)
    using different methods, and returns the DataFrame with a 'target' column.

    Methods:
    - 'ao_on_price': Awesome Oscillator on actual prices.
    - 'ao_on_pct_change': Awesome Oscillator on price percentage changes.
    - 'pct_change_on_ao': Percentage change of Awesome Oscillator on actual prices.
    - 'pct_change_std': Target based on closing price pct_change exceeding 1 std dev.
    """
    if method == "pct_change_std":
        window = 24 * 7
        close_pct_change = df["Close"].pct_change()
        # The target is based on the NEXT period's price change.
        future_pct_change = close_pct_change.shift(-1)
        rolling_std = close_pct_change.rolling(window=window).std()

        df["target"] = 0
        df.loc[future_pct_change >= (rolling_std * std_fraction), "target"] = 1
        df.loc[future_pct_change <= -(rolling_std * std_fraction), "target"] = -1
        return df

    if method == "ao_on_pct_change":
        # computing the peaks from the awesome oscillator from the pct_change of the values
        high_pct = df["High"].pct_change().fillna(0)
        low_pct = df["Low"].pct_change().fillna(0)
        ao = awesome_oscillator(high_pct, low_pct)
    elif method == "ao_on_price":
        # computing the peaks from the awesome oscillator from the actual price values
        ao = awesome_oscillator(df["High"], df["Low"])
    elif method == "pct_change_on_ao":
        # computing the peaks from the pct_change of the awesome oscillator from the actual price values
        ao_price = awesome_oscillator(df["High"], df["Low"])
        ao = ao_price.pct_change().fillna(0).replace([np.inf, -np.inf], 0)
    else:
        raise ValueError(
            f"Invalid method '{method}' specified for create_target_variable"
        )

    if ao is None or ao.isnull().all():
        # If AO can't be calculated, label all points as neutral
        df["target"] = 0
        return df

    # Find peaks (tops) and troughs (bottoms) in the AO
    peaks, _ = find_peaks(ao, distance=peak_distance, threshold=peak_threshold)
    troughs, _ = find_peaks(-ao, distance=peak_distance, threshold=peak_threshold)

    # Create the target column, default to 0 (neutral)
    df["target"] = 0
    df.loc[df.index[peaks], "target"] = 1  # Tops
    df.loc[df.index[troughs], "target"] = -1  # Bottoms

    return df


def select_features(
    X: pd.DataFrame, y: pd.Series, corr_threshold=0.7, p_value_threshold=0.1
) -> list:
    """
    Selects features based on Pearson correlation and p-value.
    Note: The correlation threshold of 0.7 is extremely high and might result in
    very few or no features being selected. The paper's methodology is followed here.
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

    print(
        f"Selected {len(selected_features)} features out of {len(X.columns)} based on correlation criteria."
    )
    return selected_features


class SVCStrategy(Strategy):  # pylint: disable=attribute-defined-outside-init
    # SVC Hyperparameters
    kernel = "rbf"
    C = 1.0
    gamma = "scale"

    # Strategy Parameters
    refit_period = 24 * 7  # Refit weekly
    lookback_length = 24 * 30 * 3  # 3 months of hourly data

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.scaler = None
        self.features = None
        self.target = None
        self.y_true = []
        self.y_pred = []

    def init(self):
        self.scaler = StandardScaler()

        # Feature and target creation
        self.features = self.I(_create_features, self.data)
        self.target = (
            (self.data.Close.to_series().pct_change(1).shift(-1) > 0)
            .astype(int)
            .bfill()
            .values
        )

    def next(self):
        # Retrain the model periodically
        if (
            len(self.data) > self.lookback_length
            and len(self.data) % self.refit_period == 0
        ):
            X_train_raw = self.features[-self.lookback_length : -1]
            y_train = self.target[-self.lookback_length : -1]

            X_train = self.scaler.fit_transform(X_train_raw)
            self.model = SVC(
                kernel=self.kernel, C=self.C, gamma=self.gamma, probability=True
            )
            self.model.fit(X_train, y_train)

        # Make prediction and trade if the model is trained
        if self.model:
            current_features = self.features[-1].reshape(1, -1)
            scaled_features = self.scaler.transform(current_features)
            prediction = self.model.predict(scaled_features)[0]

            true_label = self.target[len(self.data.Close) - 1]
            self.y_true.append(true_label)
            self.y_pred.append(prediction)

            if prediction == 1 and not self.position.is_long:
                self.buy()
            elif prediction == 0 and self.position:
                self.position.close()

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }


class RandomForestClassifierStrategy(
    Strategy
):  # pylint: disable=attribute-defined-outside-init
    # Random Forest Hyperparameters
    n_estimators = 100
    max_depth = 10
    min_samples_split = 10
    max_features = 0.5

    # Strategy Parameters
    refit_period = 24 * 7
    lookback_length = 24 * 30 * 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.scaler = None
        self.features = None
        self.target = None
        self.y_true = []
        self.y_pred = []

    def init(self):
        self.scaler = StandardScaler()

        self.features = self.I(_create_features, self.data)
        self.target = (
            (self.data.Close.to_series().pct_change(1).shift(-1) > 0)
            .astype(int)
            .bfill()
            .values
        )

    def next(self):
        if (
            len(self.data) > self.lookback_length
            and len(self.data) % self.refit_period == 0
        ):
            X_train_raw = self.features[-self.lookback_length : -1]
            y_train = self.target[-self.lookback_length : -1]

            X_train = self.scaler.fit_transform(X_train_raw)

            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=42,
                n_jobs=-1,
            )
            self.model.fit(X_train, y_train)

        if self.model:
            current_features = self.features[-1].reshape(1, -1)
            scaled_features = self.scaler.transform(current_features)
            prediction = self.model.predict(scaled_features)[0]

            true_label = self.target[len(self.data.Close) - 1]
            self.y_true.append(true_label)
            self.y_pred.append(prediction)

            if prediction == 1 and not self.position.is_long:
                self.buy()
            elif prediction == 0 and self.position:
                self.position.close()

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 30, step=5),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        }


class XGBoostPriceReversalStrategy(
    Strategy
):  # pylint: disable=attribute-defined-outside-init
    # Parameters from Optuna study
    peak_method = "pct_change_on_ao"
    peak_distance = 4
    peak_threshold = 0.3088073570333899
    corr_threshold = 0.24234536219482808
    n_estimators = 154
    learning_rate = 0.1187900234174778
    max_depth = 10
    subsample = 0.5591095634844386
    colsample_bytree = 0.8383126666102035
    gamma = 0.11692201750072992
    min_child_weight = 6

    # Strategy Parameters
    refit_period = 24 * 7
    lookback_length = 24 * 30 * 6  # 6 months of hourly data

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.scaler = None
        self.selected_cols = None
        self.features_df = None
        self.target_series = None
        self.y_true = []
        self.y_pred = []

    def init(self):
        self.scaler = StandardScaler()

        df = self.data.df
        self.features_df = create_features(df.copy())

        reversal_data = create_target_variable(
            df.copy(),
            method=self.peak_method,
            peak_distance=self.peak_distance,
            peak_threshold=self.peak_threshold,
        )
        y = reversal_data["target"]
        self.target_series = y.map({-1: 0, 0: 1, 1: 2}).bfill().ffill()

    def next(self):
        # Retrain the model periodically
        if (
            len(self.data) > self.lookback_length
            and len(self.data) % self.refit_period == 0
        ):
            end_idx = len(self.data) - 1
            start_idx = end_idx - self.lookback_length

            features_train_df = self.features_df.iloc[start_idx:end_idx]
            y_train_series = self.target_series.iloc[start_idx:end_idx]

            X_train_raw = features_train_df.values
            y_train = y_train_series.values

            # Scale and fit
            X_train = self.scaler.fit_transform(X_train_raw)

            # Class weights
            classes = np.unique(y_train)
            weights = compute_class_weight(
                class_weight="balanced", classes=classes, y=y_train
            )
            class_weight_dict = dict(zip(classes, weights))
            sample_weights = y_train_series.map(class_weight_dict).to_numpy()

            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                gamma=self.gamma,
                min_child_weight=self.min_child_weight,
                objective="multi:softmax",
                num_class=3,
                eval_metric="mlogloss",
                tree_method="hist",
                # device='cuda', # Uncomment if you have a CUDA-enabled GPU and XGBoost with GPU support
                random_state=42,
                n_jobs=-1,
            )
            self.model.fit(X_train, y_train, sample_weight=sample_weights)

        # Make prediction and trade if the model is trained
        if self.model:
            current_features_all = self.features_df.iloc[len(self.data) - 1]
            current_features_selected = current_features_all.values.reshape(1, -1)

            scaled_features = self.scaler.transform(current_features_selected)
            prediction = self.model.predict(scaled_features)[0]

            true_label = self.target_series.iloc[len(self.data) - 1]
            self.y_true.append(true_label)
            self.y_pred.append(prediction)

            # Trading logic: Buy on bottom, sell/close on top
            if prediction == 0 and not self.position.is_long:  # Bottom signal
                self.buy()
            elif prediction == 2 and self.position.is_long:  # Top signal
                self.position.close()

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            "refit_period": trial.suggest_int("refit_period", 24 * 3, 24 * 7, step=24),
            "peak_method": trial.suggest_categorical(
                "peak_method", ["pct_change_on_ao"]
            ),
            "peak_distance": trial.suggest_int("peak_distance", 4, 4),
            "peak_threshold": trial.suggest_float(
                "peak_threshold", 0.3088073570333899, 0.3088073570333899
            ),
            "corr_threshold": trial.suggest_float(
                "corr_threshold", 0.24234536219482808, 0.24234536219482808
            ),
            "n_estimators": trial.suggest_int("n_estimators", 154, 154),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.1187900234174778, 0.1187900234174778
            ),
            "max_depth": trial.suggest_int("max_depth", 10, 10),
            "subsample": trial.suggest_float(
                "subsample", 0.5591095634844386, 0.5591095634844386
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.8383126666102035, 0.8383126666102035
            ),
            "gamma": trial.suggest_float(
                "gamma", 0.11692201750072992, 0.11692201750072992
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 6, 6),
        }


def main():
    """Main function to run optimization for ML classification strategies."""
    strategies = {
        "XGBoostPriceReversalStrategy": XGBoostPriceReversalStrategy,
    }
    run_optimizations(
        strategies=strategies,
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2022-01-01T00:00:00Z",
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="ML Classification Strategies",
        n_trials_per_strategy=1,
        n_jobs=4,
    )


if __name__ == "__main__":
    main()
