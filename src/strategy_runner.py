import mlflow
from backtesting import Strategy
from backtesting.lib import crossover

from backtest_utils import (
    pct_change,
    sma,
    ewm,
    std,
    rsi_indicator,
    fetch_historical_data,
    adjust_data_to_ubtc,
    optimize_strategy,
    run_optimizations,
)


class MaCrossover(Strategy):
    short_window = 50
    long_window = 200
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        price_change = self.I(pct_change, self.data.Close)
        self.ma_short = self.I(sma, price_change, self.short_window)
        self.ma_long = self.I(sma, price_change, self.long_window)

    def next(self):
        price = self.data.Close[-1]
        if crossover(self.ma_short, self.ma_long):
            self.buy(sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit))
        elif crossover(self.ma_long, self.ma_short):
            self.sell(
                sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit)
            )

    @classmethod
    def get_optuna_params(cls, trial):
        """Suggest hyperparameters for Optuna optimization."""
        return {
            'short_window': trial.suggest_int('short_window', 10, 100),
            'long_window': trial.suggest_int('long_window', 50, 250),
        }


class BollingerBands(Strategy):
    bb_window = 20
    bb_std = 2
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        self.ma = self.I(sma, self.data.Close, self.bb_window)
        self.std = self.I(std, self.data.Close, self.bb_window)
        self.upper_band = self.ma + self.bb_std * self.std
        self.lower_band = self.ma - self.bb_std * self.std

    def next(self):
        price = self.data.Close[-1]
        if price < self.lower_band:
            self.buy(sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit))
        elif price > self.upper_band:
            self.sell(
                sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit)
            )

    @classmethod
    def get_optuna_params(cls, trial):
        """Suggest hyperparameters for Optuna optimization."""
        return {
            'bb_window': trial.suggest_int('bb_window', 10, 100),
            'bb_std': trial.suggest_float('bb_std', 1.5, 3.5),
        }


class MACD(Strategy):
    fast_span = 12
    slow_span = 26
    signal_span = 9
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        self.macd = self.I(ewm, self.data.Close, self.fast_span) - self.I(ewm, self.data.Close, self.slow_span)
        self.signal = self.I(ewm, self.macd, self.signal_span)

    def next(self):
        price = self.data.Close[-1]
        if crossover(self.macd, self.signal):
            self.buy(sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit))
        elif crossover(self.signal, self.macd):
            self.sell(
                sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit)
            )

    @classmethod
    def get_optuna_params(cls, trial):
        """Suggest hyperparameters for Optuna optimization."""
        return {
            'fast_span': trial.suggest_int('fast_span', 5, 50),
            'slow_span': trial.suggest_int('slow_span', 20, 100),
            'signal_span': trial.suggest_int('signal_span', 5, 50),
        }


class RSIDivergence(Strategy):
    rsi_window = 14
    divergence_period = 30
    stop_loss = 0.05
    take_profit = 0.10

    def init(self):
        self.rsi = self.I(rsi_indicator, self.data.Close, self.rsi_window)

    def next(self):
        if len(self.data.Close) < self.divergence_period + 1:
            return

        price = self.data.Close[-1]

        # Bullish divergence: price makes a lower low, RSI makes a higher low
        price_low_lookback = self.data.Low[-self.divergence_period:-1]
        prev_price_low_idx_in_slice = price_low_lookback.argmin()
        prev_low_idx = -(self.divergence_period - prev_price_low_idx_in_slice)

        # Bearish divergence: price makes a higher high, RSI makes a lower high
        price_high_lookback = self.data.High[-self.divergence_period:-1]
        prev_price_high_idx_in_slice = price_high_lookback.argmax()
        prev_high_idx = -(self.divergence_period - prev_price_high_idx_in_slice)

        if self.data.Low[-1] < self.data.Low[prev_low_idx] and self.rsi[-1] > self.rsi[prev_low_idx]:
            self.buy(sl=price * (1 - self.stop_loss), tp=price * (1 + self.take_profit))
        elif self.data.High[-1] > self.data.High[prev_high_idx] and self.rsi[-1] < self.rsi[prev_high_idx]:
            self.sell(sl=price * (1 + self.stop_loss), tp=price * (1 - self.take_profit))

    @classmethod
    def get_optuna_params(cls, trial):
        """Suggest hyperparameters for Optuna optimization."""
        return {
            'rsi_window': trial.suggest_int('rsi_window', 5, 30),
            'divergence_period': trial.suggest_int('divergence_period', 10, 60),
        }

#AI given the following description...
#  Multi-Indicator Algorithmic Strategy
# Advanced research has explored combining multiple indicators into a single algorithmic system to improve signal quality and filter out noise.
#     Strategy: One notable strategy from an ETH Zurich Master's Thesis developed a long-short model using a combination of indicators.
#         Long Entry: Price closes above the upper Bollinger Band AND the 50-period SMA is above the 100-period SMA (confirming an uptrend).
#         Short Entry: Price closes below the lower Bollinger Band AND the 50-period SMA is below the 100-period SMA (confirming a downtrend).
#         Exit: A trailing stop-loss was used for exits.
#     Reported Performance: This specific strategy, when backtested on hourly BTC data from 2017 to 2019, produced a Sharpe Ratio of 3.2 with a maximum drawdown of 25%. In comparison, a buy-and-hold strategy over the same period had a Sharpe Ratio of 1.13 and a maximum drawdown of 85%.
# Add a new strategy AI!

def main():
    """Main function to run the optimization with default parameters."""
    strategies = {
        "MaCrossover": MaCrossover,
        "BollingerBands": BollingerBands,
        "MACD": MACD,
        "RSIDivergence": RSIDivergence
    }
    run_optimizations(
        strategies=strategies,
        data_path="/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/BTCUSDT_1h.csv",
        start_date="2022-01-01T00:00:00Z",
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="Trading Strategies",
        n_trials_per_strategy=10
    )


if __name__ == "__main__":
    main()
