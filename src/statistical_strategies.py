import pandas as pd
import numpy as np
from backtesting import Strategy

# Import stats libraries, handle potential import errors
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

try:
    import statsmodels.api as sm
except ImportError:
    sm = None


# Custom indicator for preprocessing
def normalized_price_change(series: pd.Series, window: int = 1000) -> np.ndarray:
    """Calculates (price[i] - price[i-1]) / std(price[-window:])."""
    diff = series.diff()
    std_dev = series.rolling(window).std()
    with np.errstate(divide='ignore', invalid='ignore'):
        result = diff / std_dev
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result.fillna(0).values


class ProphetStrategy(Strategy):
    refit_period = 100  # Refit the model every N bars
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        if Prophet is None:
            raise ImportError("Prophet is not installed. Please run 'poetry add prophet'.")
        self.model = None
        self.forecast = None

    def next(self):
        # Refit model periodically
        if len(self.data.Close) > 1 and len(self.data.Close) % self.refit_period == 0:
            prophet_data = pd.DataFrame({'ds': self.data.index, 'y': self.data.Close})
            self.model = Prophet()
            self.model.fit(prophet_data, progress=False)

        # Generate forecast if model is fitted
        if self.model:
            future = self.model.make_future_dataframe(periods=1, freq='H')  # Assuming hourly data
            self.forecast = self.model.predict(future)

        price = self.data.Close[-1]
        if self.forecast is not None:
            forecast_price = self.forecast['yhat'].iloc[-1]
            if forecast_price > price and not self.position.is_long:
                self.buy(sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit))
            elif forecast_price < price and not self.position.is_short:
                self.sell(sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit))

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            'refit_period': trial.suggest_int('refit_period', 50, 200),
        }


class ARIMAStrategy(Strategy):
    p = 5
    d = 1
    q = 0
    refit_period = 100
    std_window = 1000
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        if sm is None:
            raise ImportError("statsmodels is not installed. Please run 'poetry add statsmodels'.")
        self.model_fit = None
        self.processed_data = self.I(normalized_price_change, self.data.Close, self.std_window)

    def next(self):
        price = self.data.Close[-1]

        # Refit model periodically and if we have enough data
        if len(self.data) % self.refit_period == 0 and len(self.data) > self.std_window:
            try:
                model = sm.tsa.ARIMA(self.processed_data, order=(self.p, self.d, self.q))
                self.model_fit = model.fit()
            except Exception:  # Catches convergence errors etc.
                self.model_fit = None

        if self.model_fit:
            try:
                forecast_processed = self.model_fit.forecast(steps=1)[0]
                if forecast_processed > 0 and not self.position.is_long:
                    self.buy(sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit))
                elif forecast_processed < 0 and not self.position.is_short:
                    self.sell(sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit))
            except Exception:
                pass  # Ignore forecast errors

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            'p': trial.suggest_int('p', 1, 10),
            'd': trial.suggest_int('d', 0, 2),
            'q': trial.suggest_int('q', 0, 5),
            'refit_period': trial.suggest_int('refit_period', 50, 200),
            'std_window': trial.suggest_int('std_window', 500, 1500),
        }


class SARIMAStrategy(Strategy):
    p, d, q = 5, 1, 0
    P, D, Q, s = 1, 1, 0, 24  # Seasonal order, s=24 for daily seasonality on hourly data
    refit_period = 100
    std_window = 1000
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        if sm is None:
            raise ImportError("statsmodels is not installed. Please run 'poetry add statsmodels'.")
        self.model_fit = None
        self.processed_data = self.I(normalized_price_change, self.data.Close, self.std_window)

    def next(self):
        price = self.data.Close[-1]

        if len(self.data) % self.refit_period == 0 and len(self.data) > self.std_window:
            try:
                model = sm.tsa.SARIMAX(
                    self.processed_data,
                    order=(self.p, self.d, self.q),
                    seasonal_order=(self.P, self.D, self.Q, self.s)
                )
                self.model_fit = model.fit(disp=False)
            except Exception:
                self.model_fit = None

        if self.model_fit:
            try:
                forecast_processed = self.model_fit.forecast(steps=1)[0]
                if forecast_processed > 0 and not self.position.is_long:
                    self.buy(sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit))
                elif forecast_processed < 0 and not self.position.is_short:
                    self.sell(sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit))
            except Exception:
                pass

    @classmethod
    def get_optuna_params(cls, trial):
        return {
            'p': trial.suggest_int('p', 1, 5),
            'd': trial.suggest_int('d', 0, 2),
            'q': trial.suggest_int('q', 0, 5),
            'P': trial.suggest_int('P', 0, 2),
            'D': trial.suggest_int('D', 0, 2),
            'Q': trial.suggest_int('Q', 0, 2),
            's': trial.suggest_categorical('s', [12, 24, 48]),
            'refit_period': trial.suggest_int('refit_period', 50, 200),
            'std_window': trial.suggest_int('std_window', 500, 1500),
        }
