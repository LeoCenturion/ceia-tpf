import os
import sys
import time
from functools import wraps, partial
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import clone
import cupy as cp

# Make the script runnable from anywhere
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_analysis.data_analysis import fetch_historical_data, sma, ewm, std
from src.data_analysis.indicators import rsi_indicator
from src.modeling import PurgedKFold
from src.constants import (
    OPEN_COL,
    HIGH_COL,
    LOW_COL,
    CLOSE_COL,
    VOLUME_COL,
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

# --- Palazzo Helper Functions (extracted from xgboost_price_reversal_palazzo.py) ---

def awesome_oscillator(high: pd.Series, low: pd.Series, fast_period: int = 5, slow_period: int = 34) -> pd.Series:
    median_price = (high + low) / 2
    ao = sma(median_price, fast_period) - sma(median_price, slow_period)
    return ao

def macd(close: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    ema_fast = ewm(close, span=fast_period)
    ema_slow = ewm(close, span=slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = ewm(macd_line, span=signal_period)
    histogram = macd_line - signal_line
    return pd.DataFrame({"MACD": macd_line, "Signal": signal_line, "Hist": histogram})

def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 14) -> pd.Series:
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    positive_flow = pd.Series(np.where(typical_price > typical_price.shift(1), money_flow, 0), index=typical_price.index)
    negative_flow = pd.Series(np.where(typical_price < typical_price.shift(1), money_flow, 0), index=typical_price.index)
    positive_mf = positive_flow.rolling(window=n, min_periods=0).sum()
    negative_mf = negative_flow.rolling(window=n, min_periods=0).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        money_ratio = positive_mf / negative_mf
        mfi_res = 100 - (100 / (1 + money_ratio))
    mfi_res.replace([np.inf, -np.inf], 100, inplace=True)
    return mfi_res

def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, d_n: int = 3) -> pd.DataFrame:
    low_n = low.rolling(window=n).min()
    high_n = high.rolling(window=n).max()
    k_percent = 100 * ((close - low_n) / (high_n - low_n).replace(0, 1e-9))
    d_percent = sma(k_percent, d_n)
    return pd.DataFrame({"%K": k_percent, "%D": d_percent})

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    h_minus_l = high - low
    h_minus_pc = abs(high - close.shift(1))
    l_minus_pc = abs(low - close.shift(1))
    tr = pd.concat([h_minus_l, h_minus_pc, l_minus_pc], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return ewm(tr, span=2 * n - 1)

def willr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    high_n = high.rolling(n).max()
    low_n = low.rolling(n).min()
    wr = -100 * (high_n - close) / (high_n - low_n).replace(0, 1e-9)
    return wr

def roc(close: pd.Series, n: int = 10) -> pd.Series:
    return (close.diff(n) / close.shift(n)).replace([np.inf, -np.inf], 0) * 100

def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, fast: int = 7, medium: int = 14, slow: int = 28):
    close_prev = close.shift(1).fillna(method="bfill")
    bp = close - pd.concat([low, close_prev], axis=1).min(axis=1)
    tr = true_range(high, low, close)
    avg_fast = bp.rolling(fast).sum() / tr.rolling(fast).sum().replace(0, 1e-9)
    avg_medium = bp.rolling(medium).sum() / tr.rolling(medium).sum().replace(0, 1e-9)
    avg_slow = bp.rolling(slow).sum() / tr.rolling(slow).sum().replace(0, 1e-9)
    uo = 100 * (4 * avg_fast + 2 * avg_medium + avg_slow) / (4 + 2 + 1)
    return pd.DataFrame({f"UO_{fast}_{medium}_{slow}": uo})

def true_strength_index(close: pd.Series, fast: int = 13, slow: int = 25, signal: int = 13):
    pc = close.diff(1)
    tsi = 100 * ewm(ewm(pc, span=fast), span=slow) / ewm(ewm(abs(pc), span=fast), span=slow).replace(0, 1e-9)
    signal_line = ewm(tsi, span=signal)
    return pd.DataFrame({f"TSI_{slow}_{fast}": tsi, f"TSIs_{slow}_{fast}": signal_line})

def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    up = high.diff(); down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0)
    minus_dm = down.where((down > up) & (down > 0), 0)
    _atr = atr(high, low, close, n)
    plus_di = 100 * ewm(plus_dm, span=2 * n - 1) / _atr.replace(0, 1e-9)
    minus_di = 100 * ewm(minus_dm, span=2 * n - 1) / _atr.replace(0, 1e-9)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9)
    return pd.DataFrame({f"ADX_{n}": ewm(dx, span=2 * n - 1), f"DMP_{n}": plus_di, f"DMN_{n}": minus_di})

def aroon(high: pd.Series, low: pd.Series, n: int = 14) -> pd.DataFrame:
    up = high.rolling(n).apply(lambda x: 100 * (n - (n - 1 - np.argmax(x))) / n, raw=True)
    down = low.rolling(n).apply(lambda x: 100 * (n - (n - 1 - np.argmin(x))) / n, raw=True)
    return pd.DataFrame({f"AROONU_{n}": up, f"AROOND_{n}": down})

def cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20, c: float = 0.015) -> pd.Series:
    tp = (high + low + close) / 3
    mad = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma(tp, n)) / (c * mad).replace(0, 1e-9)

def stc(close: pd.Series, fast_span=23, slow_span=50, stoch_n=10, signal_span=3) -> pd.DataFrame:
    macd_line = ewm(close, span=fast_span) - ewm(close, span=slow_span)
    def _stoch(s, n):
        return 100 * (s - s.rolling(n).min()) / (s.rolling(n).max() - s.rolling(n).min()).replace(0, 1e-9)
    stoch_d = ewm(_stoch(macd_line, stoch_n), span=signal_span)
    stoch_k2 = _stoch(stoch_d, stoch_n)
    return pd.DataFrame({f"STC_{stoch_n}_{fast_span}_{slow_span}": ewm(stoch_k2, span=signal_span), f"STCst_{stoch_n}_{fast_span}_{slow_span}": stoch_k2})

def vortex(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    tr = true_range(high, low, close)
    vip = abs(high - low.shift(1)).rolling(n).sum() / tr.rolling(n).sum().replace(0, 1e-9)
    vim = abs(low - high.shift(1)).rolling(n).sum() / tr.rolling(n).sum().replace(0, 1e-9)
    return pd.DataFrame({f"VTXP_{n}": vip, f"VTXM_{n}": vim})

def bollinger_bands(close: pd.Series, n: int = 20, std_dev: float = 2.0):
    sma_val = sma(close, n); std_val = std(close, n)
    upper = sma_val + std_dev * std_val; lower = sma_val - std_dev * std_val
    return pd.DataFrame({f"BBP_{n}_{std_dev}": (close - lower) / (upper - lower).replace(0, 1e-9)})

def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, n_ema: int = 20, n_atr: int = 20, multiplier: float = 2.0):
    ema_val = ewm(close, span=n_ema); atr_val = atr(high, low, close, n=n_atr)
    u = ema_val + multiplier * atr_val; l = ema_val - multiplier * atr_val
    return pd.DataFrame({f"KCU_{n_ema}_{multiplier:.1f}": u, f"KCL_{n_ema}_{multiplier:.1f}": l})

def donchian_channels(high: pd.Series, low: pd.Series, n: int = 20):
    return pd.DataFrame({f"DCU_{n}_{n}": high.rolling(n).max(), f"DCL_{n}_{n}": low.rolling(n).min()})

def aggregate_to_volume_bars(df, volume_threshold=50000):
    print(f"Step 1: Aggregating into volume bars of {volume_threshold} units...")
    bars = []; current_bar_data = []; cumulative_volume = 0
    for index, row in df.iterrows():
        current_bar_data.append(row); cumulative_volume += row["volume"]
        if cumulative_volume >= volume_threshold:
            bar_df = pd.DataFrame(current_bar_data)
            log_ret = np.log(bar_df["close"] / bar_df["close"].shift(1)).dropna()
            bars.append({
                "open_time": bar_df.index[0], "close_time": bar_df.index[-1],
                "open_price": bar_df[OPEN_COL].iloc[0], "High": bar_df[HIGH_COL].max(),
                "Low": bar_df[LOW_COL].min(), "close_price": bar_df["close"].iloc[-1],
                "total_volume": cumulative_volume, "intra_bar_std": log_ret.std() if len(log_ret) > 1 else 0
            })
            current_bar_data = []; cumulative_volume = 0
    vdf = pd.DataFrame(bars)
    if vdf.empty: return vdf
    vdf["bar_return"] = (vdf["close_price"] / vdf["open_price"]) - 1
    vdf.set_index("close_time", inplace=True)
    return vdf

def _create_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    o_p, h_p, l_p, c_p = df[OPEN_COL].pct_change().fillna(0), df[HIGH_COL].pct_change().fillna(0), df[LOW_COL].pct_change().fillna(0), df[CLOSE_COL].pct_change().fillna(0)
    features["pct_change"] = c_p
    features["RSI_pct"] = rsi_indicator(c_p, n=14)
    st_pct = stochastic_oscillator(h_p, l_p, c_p)
    features["Stoch_K_pct"], features["Stoch_D_pct"] = st_pct["%K"], st_pct["%D"]
    ma_pct = macd(c_p)
    features["MACD_pct"], features["MACD_Signal_pct"], features["MACD_Hist_pct"] = ma_pct["MACD"], ma_pct["Signal"], ma_pct["Hist"]
    for l in range(1, 6):
        features[f"open_pct_lag_{l}"] = o_p.shift(l); features[f"high_pct_lag_{l}"] = h_p.shift(l)
        features[f"low_pct_lag_{l}"] = l_p.shift(l); features[f"close_pct_lag_{l}"] = c_p.shift(l)
    features["RSI"] = rsi_indicator(df[CLOSE_COL], n=14)
    features["AO"] = awesome_oscillator(df[HIGH_COL], df[LOW_COL])
    features["WR"] = willr(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL])
    features["ROC"] = roc(df[CLOSE_COL])
    features = pd.concat([features, ultimate_oscillator(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL]), true_strength_index(df[CLOSE_COL])], axis=1)
    st_pr = stochastic_oscillator(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL])
    features["Stoch_K"], features["Stoch_D"] = st_pr["%K"], st_pr["%D"]
    ma_pr = macd(df[CLOSE_COL])
    features["MACD"], features["MACD_Signal"], features["MACD_Hist"] = ma_pr["MACD"], ma_pr["Signal"], ma_pr["Hist"]
    features = pd.concat([features, adx(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL]), aroon(df[HIGH_COL], df[LOW_COL])], axis=1)
    features["CCI"] = cci(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL])
    features = pd.concat([features, stc(df[CLOSE_COL]), vortex(df[HIGH_COL], df[LOW_COL], df[CLOSE_COL])], axis=1)
    features["BBP"] = bollinger_bands(df[CLOSE_COL])["BBP_20_2.0"]
    for e in [10, 20, 50]: features[f"above_ema_{e}"] = (df[CLOSE_COL] > ewm(df[CLOSE_COL], span=e)).astype(int)
    features.replace([np.inf, -np.inf], np.nan, inplace=True); features.ffill(inplace=True); features.bfill(inplace=True)
    return features

# --- Pipeline Class ---

class PalazzoXGBoostPipeline:
    def __init__(self, config):
        self.config = config

    @timer
    def step_1_data_structuring(self, raw_tick_data):
        return aggregate_to_volume_bars(raw_tick_data, self.config["volume_threshold"])

    @timer
    def step_2_feature_engineering(self, bars):
        print("Step 2: Creating features...")
        temp_df = pd.DataFrame(index=bars.index)
        temp_df[OPEN_COL], temp_df[HIGH_COL], temp_df[LOW_COL], temp_df[CLOSE_COL] = bars["open_price"], bars["High"], bars["Low"], bars["close_price"]
        features = _create_reversal_features(temp_df)
        final_features = pd.DataFrame(index=bars.index)
        for col in features.columns: final_features[f"feature_{col}"] = features[col]
        final_features["feature_return_lag_1"] = bars["bar_return"].shift(1)
        final_features["feature_volatility_lag_1"] = bars["intra_bar_std"].shift(1)
        return final_features.dropna()

    @timer
    def step_3_labeling_and_weighting(self, bars):
        print("Step 3: Creating target labels and sample weights...")
        df = bars.copy()
        df["label"] = 0
        df["next_bar_return"] = df["bar_return"].shift(-1)
        cond1 = df["next_bar_return"] >= 0
        cond2 = df["next_bar_return"] >= (df["bar_return"] + df["intra_bar_std"] * self.config["tau"])
        df.loc[cond1 & cond2, "label"] = 1
        
        # t1 for PurgedKFold (end of the period that defines the label)
        # For Palazzo, the label is defined by the next bar. So t1 is the index of the bar after 'next_bar'.
        # Actually, if we are at bar i, label i depends on bar i+1. So the window ends at i+1.
        t1 = pd.Series(df.index, index=df.index).shift(-1)
        
        # Balanced weights
        y = df["label"].dropna()
        weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
        class_weight_dict = dict(zip(np.unique(y), weights))
        sample_weights = y.map(class_weight_dict)
        
        return df[["label"]].dropna(), sample_weights, t1.dropna()

    @timer
    def run(self, raw_tick_data, model_params):
        bars = self.step_1_data_structuring(raw_tick_data)
        features = self.step_2_feature_engineering(bars)
        labels, weights, t1 = self.step_3_labeling_and_weighting(bars)
        
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
            
            # Using cupy for XGBoost if device is cuda
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
    raw_data.rename(columns={CLOSE_COL: "close", VOLUME_COL: "volume"}, inplace=True)
    
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
