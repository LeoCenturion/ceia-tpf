"""
Purged K-Fold Cross-Validation for rule-based trading strategies.

Each strategy is wrapped in a scikit-learn compatible interface and can be
optimised with Optuna.  Results are tracked in MLflow.  The best parameters
from each study are then re-evaluated via run_pipeline for a final,
fully-logged run.

Financial metrics (Sharpe, Calmar, PSR) are replicated here so src.modeling
stays import-independent of src.backtesting.
"""

from __future__ import annotations

import logging
from typing import Type

import mlflow
import numpy as np
import optuna
import pandas as pd
from numba import njit
from scipy import stats
from scipy.special import ndtr
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import classification_report, f1_score

from src.data_analysis.data_analysis import adjust_data_to_ubtc, fetch_historical_data
from src.modeling import PurgedKFold
from src.modeling.pipeline import AbstractMLPipeline
from src.modeling.pipeline_runner import run_optuna_optimization, run_pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Financial metrics – replicated from src.backtesting.ratios
# ---------------------------------------------------------------------------

_EULER_GAMMA = 0.5772156649015328


def _fold_to_returns(
    close: pd.Series, signals: pd.Series, commission: float = 0.001
) -> pd.Series:
    sig = signals.reindex(close.index).shift(1)
    trade_costs = sig.diff().abs() * commission
    returns = close.pct_change() * sig - trade_costs
    return returns.dropna()


def _sharpe_ratio(returns, periods_per_year: int = 365 * 24) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return np.nan
    std = r.std(ddof=1)
    return np.nan if std == 0.0 else float(r.mean() / std * np.sqrt(periods_per_year))


def _calmar_ratio(returns, periods_per_year: int = 365 * 24) -> float:
    r = pd.Series(np.asarray(returns, dtype=float))
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return np.nan
    ann = float(r.mean() * periods_per_year)
    cum = (1.0 + r).cumprod()
    max_dd = float(abs(((cum - cum.cummax()) / cum.cummax()).min()))
    return np.nan if max_dd == 0.0 else ann / max_dd


def _sr_std(sr_hat: float, skew: float, kurtosis: float, n_obs: int) -> float:
    variance = (1.0 - skew * sr_hat + (kurtosis - 1.0) / 4.0 * sr_hat**2) / (n_obs - 1)
    return float(np.sqrt(max(variance, 0.0)))


def _probabilistic_sharpe_ratio(returns, benchmark_sr: float = 0.0) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 4 or np.ptp(r) == 0.0:
        return np.nan
    sr_hat = r.mean() / r.std(ddof=1)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    sigma = _sr_std(sr_hat, skew, kurt, len(r))
    if sigma == 0.0:
        return 1.0 if sr_hat > benchmark_sr else 0.0
    return float(ndtr((sr_hat - benchmark_sr) / sigma))


# ---------------------------------------------------------------------------
# Signal generators – pure pandas, one per strategy
# ---------------------------------------------------------------------------


def _crossover_latch(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Latching signal: 1 when fast crosses above slow, 0 on the reverse."""
    cross_up   = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    sig = pd.Series(np.nan, index=fast.index)
    sig[cross_up]   = 1
    sig[cross_down] = 0
    return sig.ffill().fillna(0).astype(int)


def smacross_signals(data: pd.DataFrame, n1: int, n2: int) -> pd.Series:
    """signal=1 whenever fast SMA > slow SMA (level comparison)."""
    close = data["Close"]
    return (close.rolling(n1).mean() > close.rolling(n2).mean()).astype(int).fillna(0)


def ma_crossover_signals(
    data: pd.DataFrame, short_window: int, long_window: int
) -> pd.Series:
    """Latching crossover on SMA of pct-change."""
    pct = data["Close"].pct_change()
    return _crossover_latch(
        pct.rolling(short_window).mean(), pct.rolling(long_window).mean()
    )


def bollinger_bands_signals(
    data: pd.DataFrame, bb_window: int, bb_std: float
) -> pd.Series:
    """1 when price dips below lower band; 0 when it rises above upper band."""
    close = data["Close"]
    ma    = close.rolling(bb_window).mean()
    std_dev = close.rolling(bb_window).std()
    upper = ma + bb_std * std_dev
    lower = ma - bb_std * std_dev
    sig = pd.Series(np.nan, index=data.index)
    sig[close < lower] = 1
    sig[close > upper] = 0
    return sig.ffill().fillna(0).astype(int)


def macd_signals(
    data: pd.DataFrame, fast_span: int, slow_span: int, signal_span: int
) -> pd.Series:
    """Latching crossover between MACD line and signal line."""
    close = data["Close"]
    macd = (
        close.ewm(span=fast_span, adjust=False).mean()
        - close.ewm(span=slow_span, adjust=False).mean()
    )
    return _crossover_latch(macd, macd.ewm(span=signal_span, adjust=False).mean())


@njit
def _rsi_divergence_core(
    low: np.ndarray, high: np.ndarray, rsi: np.ndarray, divergence_period: int
) -> np.ndarray:
    n = len(low)
    signals = np.zeros(n, dtype=np.int64)
    sig = 0
    for i in range(divergence_period + 1, n):
        if np.isnan(rsi[i]):
            signals[i] = sig
            continue
        w_low  = low[i - divergence_period:i]
        w_high = high[i - divergence_period:i]
        w_rsi  = rsi[i - divergence_period:i]
        p_low  = np.argmin(w_low)
        p_high = np.argmax(w_high)
        if low[i] < w_low[p_low] and rsi[i] > w_rsi[p_low]:
            sig = 1
        elif high[i] > w_high[p_high] and rsi[i] < w_rsi[p_high]:
            sig = 0
        signals[i] = sig
    return signals


def rsi_divergence_signals(
    data: pd.DataFrame, rsi_window: int, divergence_period: int
) -> pd.Series:
    """Bullish RSI divergence → 1; bearish → 0 (latching)."""
    close = data["Close"]
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(rsi_window).mean()
    loss  = (-delta.clip(upper=0)).rolling(rsi_window).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    signals = _rsi_divergence_core(
        data["Low"].values, data["High"].values, rsi.values, divergence_period
    )
    return pd.Series(signals, index=data.index)


def multi_indicator_signals(
    data: pd.DataFrame,
    bb_window: int,
    bb_std: float,
    fast_sma_window: int,
    slow_sma_window: int,
) -> pd.Series:
    """Bollinger Band breakout confirmed by dual-SMA trend filter."""
    close    = data["Close"]
    ma       = close.rolling(bb_window).mean()
    std_dev  = close.rolling(bb_window).std()
    upper    = ma + bb_std * std_dev
    lower    = ma - bb_std * std_dev
    sma_fast = close.rolling(fast_sma_window).mean()
    sma_slow = close.rolling(slow_sma_window).mean()
    sig = pd.Series(np.nan, index=data.index)
    sig[(close > upper) & (sma_fast > sma_slow)] = 1
    sig[(close < lower) & (sma_fast < sma_slow)] = 0
    return sig.ffill().fillna(0).astype(int)


# ---------------------------------------------------------------------------
# sklearn-compatible strategy wrappers
# ---------------------------------------------------------------------------


class SmaCrossClassifier(BaseEstimator, ClassifierMixin):
    """SMA crossover: long when fast SMA > slow SMA."""

    def __init__(self, n1: int = 10, n2: int = 20):
        self.n1 = n1
        self.n2 = n2

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None):
        self.X_train_ = X.copy()
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        full = pd.concat([self.X_train_, X])
        return smacross_signals(full, self.n1, self.n2).loc[X.index].values

    @classmethod
    def get_optuna_params(cls, trial: optuna.Trial) -> dict:
        n1 = trial.suggest_int("n1", 5, 50)
        n2 = trial.suggest_int("n2", n1 + 1, 200)
        return {"n1": n1, "n2": n2}


class MaCrossoverClassifier(BaseEstimator, ClassifierMixin):
    """Latching crossover on SMA of percent-change series."""

    def __init__(self, short_window: int = 32, long_window: int = 129):
        self.short_window = short_window
        self.long_window = long_window

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None):
        self.X_train_ = X.copy()
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        full = pd.concat([self.X_train_, X])
        return (
            ma_crossover_signals(full, self.short_window, self.long_window)
            .loc[X.index]
            .values
        )

    @classmethod
    def get_optuna_params(cls, trial: optuna.Trial) -> dict:
        short_window = trial.suggest_int("short_window", 5, 100)
        long_window = trial.suggest_int("long_window", short_window + 1, 300)
        return {"short_window": short_window, "long_window": long_window}


class BollingerBandsClassifier(BaseEstimator, ClassifierMixin):
    """Mean-reversion: long below lower band, flat above upper band."""

    def __init__(self, bb_window: int = 20, bb_std: float = 2.0):
        self.bb_window = bb_window
        self.bb_std = bb_std

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None):
        self.X_train_ = X.copy()
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        full = pd.concat([self.X_train_, X])
        return (
            bollinger_bands_signals(full, self.bb_window, self.bb_std)
            .loc[X.index]
            .values
        )

    @classmethod
    def get_optuna_params(cls, trial: optuna.Trial) -> dict:
        return {
            "bb_window": trial.suggest_int("bb_window", 10, 60),
            "bb_std": trial.suggest_float("bb_std", 1.0, 3.5),
        }


class MACDClassifier(BaseEstimator, ClassifierMixin):
    """MACD line vs. signal line latching crossover."""

    def __init__(self, fast_span: int = 12, slow_span: int = 26, signal_span: int = 9):
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.signal_span = signal_span

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None):
        self.X_train_ = X.copy()
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        full = pd.concat([self.X_train_, X])
        return (
            macd_signals(full, self.fast_span, self.slow_span, self.signal_span)
            .loc[X.index]
            .values
        )

    @classmethod
    def get_optuna_params(cls, trial: optuna.Trial) -> dict:
        fast_span = trial.suggest_int("fast_span", 5, 30)
        slow_span = trial.suggest_int("slow_span", fast_span + 1, 60)
        signal_span = trial.suggest_int("signal_span", 3, 20)
        return {
            "fast_span": fast_span,
            "slow_span": slow_span,
            "signal_span": signal_span,
        }


class RSIDivergenceClassifier(BaseEstimator, ClassifierMixin):
    """Bullish/bearish RSI divergence signal."""

    def __init__(self, rsi_window: int = 14, divergence_period: int = 30):
        self.rsi_window = rsi_window
        self.divergence_period = divergence_period

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None):
        self.X_train_ = X.copy()
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        full = pd.concat([self.X_train_, X])
        return (
            rsi_divergence_signals(full, self.rsi_window, self.divergence_period)
            .loc[X.index]
            .values
        )

    @classmethod
    def get_optuna_params(cls, trial: optuna.Trial) -> dict:
        return {
            "rsi_window": trial.suggest_int("rsi_window", 7, 28),
            "divergence_period": trial.suggest_int("divergence_period", 10, 60),
        }


class MultiIndicatorClassifier(BaseEstimator, ClassifierMixin):
    """Bollinger Band breakout confirmed by dual-SMA trend filter."""

    def __init__(
        self,
        bb_window: int = 20,
        bb_std: float = 2.0,
        fast_sma_window: int = 50,
        slow_sma_window: int = 100,
    ):
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.fast_sma_window = fast_sma_window
        self.slow_sma_window = slow_sma_window

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None):
        self.X_train_ = X.copy()
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        full = pd.concat([self.X_train_, X])
        return (
            multi_indicator_signals(
                full,
                self.bb_window,
                self.bb_std,
                self.fast_sma_window,
                self.slow_sma_window,
            )
            .loc[X.index]
            .values
        )

    @classmethod
    def get_optuna_params(cls, trial: optuna.Trial) -> dict:
        fast_sma = trial.suggest_int("fast_sma_window", 10, 100)
        slow_sma = trial.suggest_int("slow_sma_window", fast_sma + 1, 300)
        return {
            "bb_window": trial.suggest_int("bb_window", 10, 60),
            "bb_std": trial.suggest_float("bb_std", 1.0, 3.5),
            "fast_sma_window": fast_sma,
            "slow_sma_window": slow_sma,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TradingStrategyPipeline(AbstractMLPipeline):
    """
    Pipeline for rule-based trading strategy classifiers.

    Features (step_2) are the raw OHLCV bars — each classifier decides how to
    use them internally.  Labels (step_3) are the close price series, which
    run_cv uses to compute fold returns and Sharpe ratios.

    Config keys:
        n_splits        – PurgedKFold splits (default 5)
        pct_embargo     – embargo fraction (default 0.01)
        commission      – round-trip cost per trade (default 0.001)
        periods_per_year – override annualisation factor (auto-detected if absent)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.problem_type = "trading"

    def step_1_data_structuring(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        return adjust_data_to_ubtc(raw_data)

    def step_2_feature_engineering(self, bars: pd.DataFrame) -> pd.DataFrame:
        # Drop last bar so every event has a well-defined end time (next bar)
        return bars.iloc[:-1]

    def step_3_labeling_and_weighting(
        self, bars: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        close = bars["Close"].iloc[:-1]
        t1 = pd.Series(bars.index[1:], index=bars.index[:-1])
        weights = pd.Series(1.0, index=close.index)
        return close, weights, t1

    def run_cv(self, raw_data: pd.DataFrame, model) -> tuple:
        """Purged K-Fold CV without scaling or PCA; evaluates with Sharpe ratio."""
        bars = self.step_1_data_structuring(raw_data)
        X = self.step_2_feature_engineering(bars)
        close, sw, t1 = self.step_3_labeling_and_weighting(bars)

        common_idx = (
            X.index.intersection(close.index)
            .intersection(sw.index)
            .intersection(t1.index)
        )
        X = X.loc[common_idx]
        close = close.loc[common_idx]
        sw = sw.loc[common_idx]
        t1 = t1.loc[common_idx]

        commission = self.config.get("commission", 0.001)
        periods_per_year = self.config.get("periods_per_year", None)
        if periods_per_year is None:
            idx = pd.to_datetime(close.index)
            years = (idx[-1] - idx[0]).total_seconds() / (365.25 * 24 * 3600)
            periods_per_year = int(len(close) / years) if years > 0 else 365 * 24

        cv = PurgedKFold(
            n_splits=self.config.get("n_splits", 5),
            t1=t1,
            pct_embargo=self.config.get("pct_embargo", 0.01),
        )

        scores = []
        all_true_labels: list = []
        all_signals: list = []
        print(f"Starting trading CV ({self.config.get('n_splits', 5)} folds)…")
        for i, (train_idx, test_idx) in enumerate(cv.split(X, close)):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            close_test = close.iloc[test_idx]
            sw_train = sw.iloc[train_idx]

            if X_train.empty or X_test.empty:
                print(f"  Skipping fold {i + 1} (empty split).")
                continue

            fold_model = clone(model)
            fold_model.fit(X_train, None, sample_weight=sw_train.values)
            signals = pd.Series(fold_model.predict(X_test), index=X_test.index)

            rets = _fold_to_returns(close_test, signals, commission)
            sr = _sharpe_ratio(rets, periods_per_year)
            safe_sr = float(sr) if np.isfinite(sr) else 0.0
            print(f"  Fold {i + 1} Sharpe: {safe_sr:.4f}")

            # Derive true direction: 1 if next bar's close was higher, 0 otherwise
            next_ret = close_test.pct_change().shift(-1)
            mask = next_ret.notna()
            true_labels = (next_ret[mask] > 0).astype(int).values
            fold_signals = signals[mask].values

            fold_f1 = f1_score(
                true_labels, fold_signals, average="weighted", zero_division=0
            )
            scores.append(fold_f1)
            print(f"  Fold {i + 1} Weighted F1: {fold_f1:.4f}")

            all_true_labels.append(true_labels)
            all_signals.append(fold_signals)

        if all_true_labels:
            report = classification_report(
                np.concatenate(all_true_labels),
                np.concatenate(all_signals),
                target_names=["flat", "long"],
                zero_division=0,
            )
            print("\nOverall CV Signal vs Direction Report:\n", report)
            if mlflow.active_run():
                mlflow.log_text(report, "cv_signal_classification_report.txt")

        trained_model = clone(model)
        trained_model.fit(X, None, sample_weight=sw.values)

        return trained_model, scores, X, close, sw, t1, None


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def run_trading_optimization(
    model_class: Type[BaseEstimator],
    raw_data: pd.DataFrame,
    pipeline_config: dict,
    n_trials: int = 30,
    experiment_name: str = "Trading_Strategy_Optimization",
    n_jobs: int = 1,
    optuna_storage: str = "sqlite:///optuna-study.db",
) -> tuple[dict, float]:
    tracking_uri = pipeline_config.get("tracking_uri", "sqlite:///mlflow.db?timeout=60")
    _, best_model_params, best_value = run_optuna_optimization(
        pipeline_cls=TradingStrategyPipeline,
        model_cls=model_class,
        raw_data=raw_data,
        pipeline_config=pipeline_config,
        experiment_name=experiment_name,
        n_trials=n_trials,
        n_jobs=n_jobs,
        run_name_prefix=model_class.__name__,
        optuna_storage=optuna_storage,
        tracking_uri=tracking_uri,
        best_metric_name="best_avg_cv_weighted_f1",
        data_path=pipeline_config.get("data_path"),
    )
    return best_model_params, best_value


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    data_path = (
        "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/"
        "data/binance/python/data/spot/daily/klines/BTCUSDT/1m/"
        "BTCUSDT_consolidated_klines.csv"
    )
    raw_data = fetch_historical_data(
        symbol="BTC/USDT", timeframe="1m", data_path=data_path
    )

    N_TRIALS = 30
    pipeline_config = {
        "n_splits": 5,
        "pct_embargo": 0.01,
        "commission": 0.001,
        "tracking_uri": "sqlite:///mlflow.db?timeout=60",
        "data_path": data_path,
    }

    strategies: dict[str, Type[BaseEstimator]] = {
        # "SmaCrossClassifier": SmaCrossClassifier,
        # "MaCrossoverClassifier": MaCrossoverClassifier,
        # "BollingerBandsClassifier": BollingerBandsClassifier,
        # "MACDClassifier": MACDClassifier,
        "RSIDivergenceClassifier": RSIDivergenceClassifier,
        "MultiIndicatorClassifier": MultiIndicatorClassifier,
    }

    results: dict[str, dict] = {}
    for name, model_cls in strategies.items():
        print(f"\n{'=' * 60}")
        print(f"Optimizing {name} …")
        print("=" * 60)
        try:
            best_params, best_sharpe = run_trading_optimization(
                model_class=model_cls,
                raw_data=raw_data,
                pipeline_config=pipeline_config,
                n_trials=N_TRIALS,
                n_jobs=8,
                experiment_name="Trading_Strategy_Optimization",
            )
            results[name] = {"best_params": best_params, "best_f1": best_sharpe}
        except Exception as exc:
            logger.error("Failed to optimize %s: %s", name, exc)
            results[name] = {"error": str(exc)}

    print("\n\n── Summary ──────────────────────────────────────────────")
    for name, result in results.items():
        if "error" in result:
            print(f"  {name}: FAILED — {result['error']}")
        else:
            print(
                f"  {name}: best Weighted F1={result['best_f1']:.4f}  "
                f"params={result['best_params']}"
            )


if __name__ == "__main__":
    main()
