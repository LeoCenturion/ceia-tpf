"""
Optuna hyperparameter optimization for statistical time series models.

Each model (SMA, ARIMA, SARIMA, Kalman+ARIMA, ARIMAX+GARCH) is wrapped in a
scikit-learn compatible interface and optimized with Optuna.  Results are tracked
in MLflow.  The best parameters from each study are then re-evaluated via
run_pipeline for a final, fully-logged run.
"""

import logging
from typing import Optional, Type

import mlflow
import numpy as np
import optuna
import pandas as pd
import statsmodels.api as sm
from pykalman import KalmanFilter
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import classification_report, f1_score

from src.data_analysis.data_analysis import fetch_historical_data, timer
from src.modeling import PurgedKFold
from src.modeling.pipeline import AbstractMLPipeline
from src.modeling.pipeline_runner import run_optuna_optimization, run_pipeline

try:
    from arch import arch_model
except ImportError:
    arch_model = None  # type: ignore[assignment]

try:
    from neuralprophet import NeuralProphet as _NeuralProphet
except ImportError:
    _NeuralProphet = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ── helpers (ported from statistical_strategies) ──────────────────────────────


def price_difference(series: np.ndarray) -> np.ndarray:
    """Computes (price[i] - price[i-1]) / price[i-1]."""
    return pd.Series(series).pct_change().fillna(0).values


def kalman_filter_indicator(series: np.ndarray) -> np.ndarray:
    """Applies a Kalman filter to smooth a time series."""
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    filtered_means, _ = kf.filter(series)
    return filtered_means.flatten()


# ── sklearn wrappers ──────────────────────────────────────────────────────────


class SMAClassifier(BaseEstimator, ClassifierMixin):
    """SMA crossover: +1 when fast SMA crosses above slow SMA, -1 below."""

    def __init__(self, n1: int = 10, n2: int = 20):
        self.n1 = n1
        self.n2 = n2

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None):
        self.X_train_ = X["close"].values
        self.classes_ = np.array([-1, 0, 1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        close_test = X["close"].values
        full = np.concatenate([self.X_train_, close_test])
        offset = len(self.X_train_)
        predictions = np.zeros(len(close_test), dtype=int)

        for i in range(len(close_test)):
            idx = offset + i
            if idx < self.n2:
                continue
            sma1 = np.mean(full[idx - self.n1 + 1 : idx + 1])
            sma2 = np.mean(full[idx - self.n2 + 1 : idx + 1])
            if sma1 > sma2:
                predictions[i] = 1
            elif sma1 < sma2:
                predictions[i] = -1

        return predictions

    @classmethod
    def get_optuna_params(cls, trial: optuna.Trial) -> dict:
        n1 = trial.suggest_int("n1", 5, 50)
        n2 = trial.suggest_int("n2", n1 + 1, 200)
        return {"n1": n1, "n2": n2}


class ARIMAClassifier(BaseEstimator, ClassifierMixin):
    """
    ARIMA-based directional classifier.

    Fits an ARIMA model on the return series and predicts the sign of the
    one-step-ahead forecast.  During predict, the model is updated online
    (via statsmodels append) and fully refitted every refit_period steps.
    """

    def __init__(
        self,
        p: int = 12,
        d: int = 1,
        q: int = 12,
        lookback_length: int = 24 * 30,
        threshold: float = 1e-5,
        refit_period: int = 24 * 7,
    ):
        self.p = p
        self.d = d
        self.q = q
        self.lookback_length = lookback_length
        self.threshold = threshold
        self.refit_period = refit_period

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None):
        returns = X["returns"].values
        self.X_train_ = returns
        self.classes_ = np.array([-1, 0, 1])
        try:
            data = returns[-self.lookback_length :]
            self.model_fit_ = sm.tsa.ARIMA(data, order=(self.p, self.d, self.q)).fit()
        except Exception:
            self.model_fit_ = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_test = X["returns"].values
        current_fit = self.model_fit_
        predictions = []

        for i, val in enumerate(X_test):
            # Update: full refit or incremental append
            if i > 0 and i % self.refit_period == 0:
                combined = np.concatenate(
                    [self.X_train_[-self.lookback_length :], X_test[: i + 1]]
                )
                try:
                    current_fit = sm.tsa.ARIMA(
                        combined[-self.lookback_length :],
                        order=(self.p, self.d, self.q),
                    ).fit()
                except Exception:
                    current_fit = None
            elif current_fit is not None:
                try:
                    current_fit = current_fit.append([val])
                except Exception:
                    current_fit = None

            # Forecast next-step direction
            pred = 0
            if current_fit is not None:
                try:
                    fc = current_fit.forecast(steps=1)[0]
                    pred = (
                        1
                        if fc > self.threshold
                        else (-1 if fc < -self.threshold else 0)
                    )
                except Exception:
                    pass
            predictions.append(pred)

        return np.array(predictions)

    @classmethod
    def get_optuna_params(cls, trial: optuna.Trial) -> dict:
        return {
            "p": trial.suggest_int("p", 1, 10),
            "d": trial.suggest_int("d", 0, 2),
            "q": trial.suggest_int("q", 0, 5),
            "threshold": trial.suggest_float("threshold", 1e-6, 1e-3, log=True),
            "refit_period": trial.suggest_categorical(
                "refit_period", [24 * 7, 24 * 30]
            ),
            "lookback_length": trial.suggest_categorical(
                "lookback_length", [24 * 30, 24 * 60]
            ),
        }


class SARIMAClassifier(BaseEstimator, ClassifierMixin):
    """SARIMA-based directional classifier with seasonal component."""

    def __init__(
        self,
        p: int = 5,
        d: int = 1,
        q: int = 0,
        P: int = 1,
        D: int = 1,
        Q: int = 0,
        s: int = 24,
        lookback_length: int = 24 * 30,
        threshold: float = 1e-5,
        refit_period: int = 24 * 7,
    ):
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.s = s
        self.lookback_length = lookback_length
        self.threshold = threshold
        self.refit_period = refit_period

    def _fit_sarima(self, data: np.ndarray):
        return sm.tsa.SARIMAX(
            data,
            order=(self.p, self.d, self.q),
            seasonal_order=(self.P, self.D, self.Q, self.s),
        ).fit(disp=False)

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None):
        returns = X["returns"].values
        self.X_train_ = returns
        self.classes_ = np.array([-1, 0, 1])
        try:
            self.model_fit_ = self._fit_sarima(returns[-self.lookback_length :])
        except Exception:
            self.model_fit_ = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_test = X["returns"].values
        current_fit = self.model_fit_
        predictions = []

        for i, val in enumerate(X_test):
            if i > 0 and i % self.refit_period == 0:
                combined = np.concatenate(
                    [self.X_train_[-self.lookback_length :], X_test[: i + 1]]
                )
                try:
                    current_fit = self._fit_sarima(combined[-self.lookback_length :])
                except Exception:
                    current_fit = None
            elif current_fit is not None:
                try:
                    current_fit = current_fit.append([val])
                except Exception:
                    current_fit = None

            pred = 0
            if current_fit is not None:
                try:
                    fc = current_fit.forecast(steps=1)[0]
                    pred = (
                        1
                        if fc > self.threshold
                        else (-1 if fc < -self.threshold else 0)
                    )
                except Exception:
                    pass
            predictions.append(pred)

        return np.array(predictions)

    @classmethod
    def get_optuna_params(cls, trial: optuna.Trial) -> dict:
        return {
            "p": trial.suggest_int("p", 1, 5),
            "d": trial.suggest_int("d", 0, 2),
            "q": trial.suggest_int("q", 0, 5),
            "P": trial.suggest_int("P", 0, 2),
            "D": trial.suggest_int("D", 0, 2),
            "Q": trial.suggest_int("Q", 0, 2),
            "s": trial.suggest_categorical("s", [12, 24, 48]),
            "threshold": trial.suggest_float("threshold", 1e-6, 1e-3, log=True),
            "refit_period": trial.suggest_categorical(
                "refit_period", [24 * 7, 24 * 30]
            ),
        }


class KalmanARIMAClassifier(BaseEstimator, ClassifierMixin):
    """
    Kalman filter pre-processing followed by an ARIMA classifier.

    The Kalman filter smooths the return series before fitting ARIMA, which
    reduces noise and can improve forecast stability.
    """

    def __init__(
        self,
        p: int = 12,
        d: int = 0,
        q: int = 12,
        lookback_length: int = 24 * 30,
        threshold: float = 0.001,
        refit_period: int = 24 * 7,
    ):
        self.p = p
        self.d = d
        self.q = q
        self.lookback_length = lookback_length
        self.threshold = threshold
        self.refit_period = refit_period

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None):
        returns = X["returns"].values
        self.X_train_ = returns
        filtered = kalman_filter_indicator(returns)
        self.filtered_train_ = filtered
        self.classes_ = np.array([-1, 0, 1])
        try:
            data = filtered[-self.lookback_length :]
            self.model_fit_ = sm.tsa.ARIMA(data, order=(self.p, self.d, self.q)).fit()
        except Exception:
            self.model_fit_ = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_test = X["returns"].values
        filtered_test = kalman_filter_indicator(X_test)
        current_fit = self.model_fit_
        predictions = []

        for i, val in enumerate(filtered_test):
            if i > 0 and i % self.refit_period == 0:
                combined_filtered = np.concatenate(
                    [
                        self.filtered_train_[-self.lookback_length :],
                        filtered_test[: i + 1],
                    ]
                )
                try:
                    current_fit = sm.tsa.ARIMA(
                        combined_filtered[-self.lookback_length :],
                        order=(self.p, self.d, self.q),
                    ).fit()
                except Exception:
                    current_fit = None
            elif current_fit is not None:
                try:
                    current_fit = current_fit.append([val])
                except Exception:
                    current_fit = None

            pred = 0
            if current_fit is not None:
                try:
                    fc = current_fit.forecast(steps=1)[0]
                    pred = (
                        1
                        if fc > self.threshold
                        else (-1 if fc < -self.threshold else 0)
                    )
                except Exception:
                    pass
            predictions.append(pred)

        return np.array(predictions)

    @classmethod
    def get_optuna_params(cls, trial: optuna.Trial) -> dict:
        return {
            "p": trial.suggest_int("p", 5, 10),
            "d": trial.suggest_int("d", 0, 2),
            "q": trial.suggest_int("q", 3, 7),
            "threshold": trial.suggest_float("threshold", 1e-4, 1e-2, log=True),
            "refit_period": trial.suggest_categorical(
                "refit_period", [24 * 7, 24 * 30]
            ),
            "lookback_length": trial.suggest_categorical(
                "lookback_length", [24 * 30, 24 * 60]
            ),
        }


class ARIMAXGARCHClassifier(BaseEstimator, ClassifierMixin):
    """
    ARIMAX + GARCH classifier.

    GARCH is fitted on the scaled return series to estimate conditional
    volatility, which is fed as an exogenous variable to ARIMAX.
    Requires the `arch` package.
    """

    # Scale factor mirrors the original strategy (avoids numerical issues)
    _SCALE = 10_000.0

    def __init__(
        self,
        p: int = 5,
        d: int = 1,
        q: int = 0,
        g_p: int = 1,
        g_q: int = 1,
        lookback_length: int = 24 * 30,
        threshold: float = 0.5,
        refit_period: int = 24 * 7,
    ):
        self.p = p
        self.d = d
        self.q = q
        self.g_p = g_p
        self.g_q = g_q
        self.lookback_length = lookback_length
        self.threshold = threshold
        self.refit_period = refit_period

    def _fit_models(self, returns: np.ndarray):
        """Returns (garch_fit, arimax_fit) or (None, None) on failure."""
        if arch_model is None:
            raise ImportError("Install the 'arch' package for ARIMAXGARCHClassifier.")
        scaled = returns * self._SCALE
        try:
            garch_fit = arch_model(scaled, p=self.g_p, q=self.g_q).fit(disp="off")
            cond_vol = garch_fit.conditional_volatility
            arimax_fit = sm.tsa.ARIMA(
                scaled, exog=cond_vol, order=(self.p, self.d, self.q)
            ).fit()
            return garch_fit, arimax_fit
        except Exception:
            return None, None

    def _garch_next_vol(self, garch_fit) -> Optional[float]:
        try:
            fc = garch_fit.forecast(horizon=1)
            return float(np.sqrt(fc.variance.iloc[-1, 0]))
        except Exception:
            return None

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None):
        if arch_model is None:
            raise ImportError("Install the 'arch' package for ARIMAXGARCHClassifier.")
        returns = X["returns"].values
        self.X_train_ = returns
        self.classes_ = np.array([-1, 0, 1])
        self.garch_fit_, self.arimax_fit_ = self._fit_models(
            returns[-self.lookback_length :]
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_test = X["returns"].values
        current_garch = self.garch_fit_
        current_arimax = self.arimax_fit_
        predictions = []

        for i, val in enumerate(X_test):
            if i > 0 and i % self.refit_period == 0:
                combined = np.concatenate(
                    [self.X_train_[-self.lookback_length :], X_test[: i + 1]]
                )
                current_garch, current_arimax = self._fit_models(
                    combined[-self.lookback_length :]
                )
            elif current_garch is not None and current_arimax is not None:
                next_vol = self._garch_next_vol(current_garch)
                if next_vol is not None:
                    try:
                        current_arimax = current_arimax.append(
                            [val * self._SCALE],
                            exog=np.array([[next_vol]]),
                        )
                    except Exception:
                        current_arimax = None

            pred = 0
            if current_garch is not None and current_arimax is not None:
                next_vol = self._garch_next_vol(current_garch)
                if next_vol is not None:
                    try:
                        fc = current_arimax.forecast(steps=1, exog=[next_vol])[0]
                        pred = (
                            1
                            if fc > self.threshold
                            else (-1 if fc < -self.threshold else 0)
                        )
                    except Exception:
                        pass
            predictions.append(pred)

        return np.array(predictions)

    @classmethod
    def get_optuna_params(cls, trial: optuna.Trial) -> dict:
        return {
            "p": trial.suggest_int("p", 1, 5),
            "d": trial.suggest_int("d", 0, 1),
            "q": trial.suggest_int("q", 0, 5),
            "g_p": trial.suggest_int("g_p", 1, 3),
            "g_q": trial.suggest_int("g_q", 1, 3),
            "threshold": trial.suggest_float("threshold", 0.01, 10.0, log=True),
            "refit_period": trial.suggest_categorical(
                "refit_period", [24 * 7, 24 * 30]
            ),
            "lookback_length": trial.suggest_categorical(
                "lookback_length", [24 * 30, 24 * 60]
            ),
        }


class NeuralProphetClassifier(BaseEstimator, ClassifierMixin):
    """
    NeuralProphet autoregressive directional classifier.

    Fits NeuralProphet in AR mode (n_lags > 0) on close prices. The sign of
    (predicted_close[t+1] - actual_close[t]) is used as the directional signal.
    Because yhat1 at row t+1 is computed from close[t-n_lags+1 : t], it does
    not leak future data.
    """

    def __init__(
        self,
        n_lags: int = 24,
        num_hidden_layers: int = 0,
        d_hidden: int = 64,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        threshold: float = 1e-4,
    ):
        self.n_lags = n_lags
        self.num_hidden_layers = num_hidden_layers
        self.d_hidden = d_hidden
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold

    def _build_prophet_df(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"ds": X.index, "y": X["close"].values})

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None):
        if _NeuralProphet is None:
            raise ImportError("Install neuralprophet to use NeuralProphetClassifier.")
        self.X_train_ = X.copy()
        self.classes_ = np.array([-1, 0, 1])

        model_kwargs: dict = {
            "n_lags": self.n_lags,
            "n_forecasts": 1,
            "num_hidden_layers": self.num_hidden_layers,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "trainer_config": {
                "enable_progress_bar": False,
                "enable_model_summary": False,
            },
        }
        if self.num_hidden_layers > 0:
            model_kwargs["d_hidden"] = self.d_hidden

        try:
            self.model_ = _NeuralProphet(**model_kwargs)
            self.model_.fit(self._build_prophet_df(X), freq="h")
        except Exception:
            self.model_ = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            return np.zeros(len(X), dtype=int)

        combined_close = pd.concat([self.X_train_["close"], X["close"]])
        full_df = pd.DataFrame({"ds": combined_close.index, "y": combined_close.values})

        try:
            future = self.model_.make_future_dataframe(
                full_df, n_historic_predictions=True
            )
            forecast = self.model_.predict(future)
            forecast = (
                forecast[["ds", "yhat1"]].set_index("ds").reindex(combined_close.index)
            )

            # yhat1[t] = predicted close[t]; shift(-1) places predicted_close[t+1] at row t
            # direction[t] = sign(predicted_close[t+1] - actual_close[t]) ≡ label[t]
            yhat_next = forecast["yhat1"].shift(-1)
            pred_return = (yhat_next - combined_close) / combined_close.clip(min=1e-10)

            direction = np.where(
                pred_return > self.threshold,
                1,
                np.where(pred_return < -self.threshold, -1, 0),
            )
            return (
                pd.Series(direction, index=combined_close.index)
                .reindex(X.index)
                .fillna(0)
                .astype(int)
                .values
            )
        except Exception:
            return np.zeros(len(X), dtype=int)

    @classmethod
    def get_optuna_params(cls, trial: optuna.Trial) -> dict:
        return {
            "n_lags": trial.suggest_int("n_lags", 12, 168),
            "num_hidden_layers": trial.suggest_int("num_hidden_layers", 0, 2),
            "d_hidden": trial.suggest_categorical("d_hidden", [32, 64, 128]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "epochs": trial.suggest_categorical("epochs", [50, 100, 200]),
            "threshold": trial.suggest_float("threshold", 1e-5, 1e-3, log=True),
        }


# ── pipeline ──────────────────────────────────────────────────────────────────


class StatisticalModelPipeline(AbstractMLPipeline):
    """
    Pipeline for statistical time series classifiers.

    Skips StandardScaler and PCA — statistical models operate directly on the
    raw return series.  The feature matrix has two columns: 'close' (price
    level, for SMA) and 'returns' (pct change, for ARIMA-family models).
    Labels are the sign of the next bar's return.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.problem_type = "classification"

    def step_1_data_structuring(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        df = raw_data.copy()
        start = self.config.get("start_date")
        end = self.config.get("end_date")
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        return df

    def step_2_feature_engineering(self, bars: pd.DataFrame) -> pd.DataFrame:
        close = bars["Close"]
        returns = close.pct_change().fillna(0)
        return pd.DataFrame(
            {"close": close, "returns": returns}, index=bars.index
        ).dropna()

    def step_3_labeling_and_weighting(
        self, bars: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        close = bars["Close"]
        next_return = close.pct_change().shift(-1)
        labels = np.sign(next_return).fillna(0).astype(int)
        t1 = pd.Series(bars.index, index=bars.index)
        sample_weights = pd.Series(1.0, index=bars.index)
        return labels, sample_weights, t1

    @timer
    def run_cv(self, raw_tick_data: pd.DataFrame, model) -> tuple:
        """Purged CV without scaling or PCA."""
        bars = self.step_1_data_structuring(raw_tick_data)
        features = self.step_2_feature_engineering(bars)
        labels, sample_weights, t1 = self.step_3_labeling_and_weighting(bars)

        common_idx = (
            features.index.intersection(labels.index)
            .intersection(sample_weights.index)
            .intersection(t1.index)
        )
        X_raw = features.loc[common_idx]
        y = labels.loc[common_idx]
        sw = sample_weights.loc[common_idx]
        t1_series = t1.loc[common_idx]

        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        cv = PurgedKFold(
            n_splits=self.config["n_splits"],
            t1=t1_series,
            pct_embargo=self.config["pct_embargo"],
        )

        scores = []
        all_y_test: list = []
        all_y_pred: list = []
        print(f"Starting statistical CV ({self.config['n_splits']} folds)…")
        for i, (train_idx, test_idx) in enumerate(cv.split(X_raw, y)):
            X_train = X_raw.iloc[train_idx]
            X_test = X_raw.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            if X_train.empty or X_test.empty:
                print(f"  Skipping fold {i + 1} (empty split).")
                continue

            fold_model = clone(model)
            fold_model.fit(X_train, y_train, sample_weight=sw.loc[X_train.index].values)
            y_pred = fold_model.predict(X_test)
            score = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            scores.append(score)
            all_y_test.append(y_test.values)
            all_y_pred.append(y_pred)
            print(f"  Fold {i + 1} F1: {score:.4f}")

        trained_model = clone(model)
        trained_model.fit(X_raw, y, sample_weight=sw.values)

        return trained_model, scores, X_raw, y, sw, t1_series, None


# ── optimization ──────────────────────────────────────────────────────────────


def run_statistical_optimization(
    model_class: Type[BaseEstimator],
    raw_data: pd.DataFrame,
    pipeline_config: dict,
    n_trials: int = 30,
    experiment_name: str = "Statistical_Models_Optimization",
    n_jobs: int = 1,
    optuna_storage: str = "sqlite:///optuna-study.db",
) -> tuple[dict, float]:
    tracking_uri = pipeline_config.get("tracking_uri", "sqlite:///mlflow.db")
    _, best_model_params, best_value = run_optuna_optimization(
        pipeline_cls=StatisticalModelPipeline,
        model_cls=model_class,
        raw_data=raw_data,
        pipeline_config=pipeline_config,
        experiment_name=experiment_name,
        n_trials=n_trials,
        n_jobs=n_jobs,
        run_name_prefix=model_class.__name__,
        optuna_storage=optuna_storage,
        tracking_uri=tracking_uri,
        best_metric_name="best_avg_cv_f1",
        data_path=pipeline_config.get("data_path"),
    )
    return best_model_params, best_value


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    data_path = (
        "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/"
        "data/BTCUSDT_1h.csv"
    )
    raw_data = fetch_historical_data(
        data_path=data_path,
        start_date="2020-01-01T00:00:00Z",
    )

    N_TRIALS = 10
    pipeline_config = {
        "n_splits": 3,
        "pct_embargo": 0.01,
        "start_date": "2022-01-01",
        "tracking_uri": "sqlite:///mlflow.db",
        "data_path": data_path,
    }

    models_to_optimize: dict[str, Type[BaseEstimator]] = {
        # "SMAClassifier": SMAClassifier,
        # "ARIMAClassifier": ARIMAClassifier,
        # SARIMA is slow — use a smaller n_trials or comment out for quick runs
        # "SARIMAClassifier": SARIMAClassifier,
        "KalmanARIMAClassifier": KalmanARIMAClassifier,
    }
    if arch_model is not None:
        models_to_optimize["ARIMAXGARCHClassifier"] = ARIMAXGARCHClassifier
    if _NeuralProphet is not None:
        models_to_optimize["NeuralProphetClassifier"] = NeuralProphetClassifier

    results: dict[str, dict] = {}
    for name, model_cls in models_to_optimize.items():
        print(f"\n{'=' * 60}")
        print(f"Optimizing {name} …")
        print("=" * 60)
        try:
            best_params, best_f1 = run_statistical_optimization(
                model_class=model_cls,
                raw_data=raw_data,
                pipeline_config=pipeline_config,
                n_trials=N_TRIALS,
                experiment_name="Statistical_Models_Optimization",
            )
            results[name] = {"best_params": best_params, "best_f1": best_f1}
        except Exception as exc:
            logger.error("Failed to optimize %s: %s", name, exc)
            results[name] = {"error": str(exc)}

    print("\n\n── Summary ──────────────────────────────────────────────")
    for name, result in results.items():
        if "error" in result:
            print(f"  {name}: FAILED — {result['error']}")
        else:
            print(
                f"  {name}: best F1={result['best_f1']:.4f}  "
                f"params={result['best_params']}"
            )


if __name__ == "__main__":
    main()
