"""
CPCV-adapted strategy subclasses.

The original trading strategies (MaCrossover, BollingerBands, MACD,
RSIDivergence, MultiIndicatorStrategy) call self.buy() / self.sell() in
next(), which requires a live broker object.  TrialStrategy.predict() runs
with broker=None, so those calls would raise AttributeError.

Each subclass here overrides next() to set self.signal (int) instead,
converting the same entry logic into a binary long/flat signal that
predict() can collect without a broker.

  signal = 1  → long
  signal = 0  → flat / exit
"""


import numpy as np
from backtesting.lib import crossover

from src.backtesting.strategies.trading_strategies import (
    BollingerBands,
    MACD,
    MaCrossover,
    MultiIndicatorStrategy,
    RSIDivergence,
)
from src.data_analysis.data_analysis import ewm


class MaCrossoverCPCV(MaCrossover):
    """MaCrossover for CPCV: sets self.signal instead of placing orders."""

    def next(self):
        if self.ma_short is None or self.ma_long is None:
            return
        if crossover(self.ma_short, self.ma_long):
            self.signal = 1
        elif crossover(self.ma_long, self.ma_short):
            self.signal = 0


class BollingerBandsCPCV(BollingerBands):
    """BollingerBands for CPCV: sets self.signal instead of placing orders."""

    def next(self):
        price = self.data.Close[-1]
        if self.lower_band is None or self.upper_band is None:
            return
        if price < self.lower_band[-1]:
            self.signal = 1
        elif price > self.upper_band[-1]:
            self.signal = 0


class MACDCPCV(MACD):
    """MACD for CPCV.

    MACD.__init__ stores the MACD signal line in self.signal (shadowing
    TrialStrategy.signal which is an int).  This subclass renames the
    indicator to self.macd_signal_line and resets self.signal to 0 so
    predict() can read it as a trading signal.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Parent set self.signal = None (the MACD signal line array).
        # Reset to int so predict() can read it correctly.
        self.signal = 0
        self.macd_signal_line: np.ndarray = np.ndarray([])

    def init(self):
        self.macd = self.I(ewm, self.data.Close, self.fast_span) - self.I(
            ewm, self.data.Close, self.slow_span
        )
        if self.macd is not None:
            self.macd_signal_line = self.I(ewm, self.macd, self.signal_span)

    def next(self):
        if self.macd is None or self.macd_signal_line is None:
            return
        if crossover(self.macd, self.macd_signal_line):
            self.signal = 1
        elif crossover(self.macd_signal_line, self.macd):
            self.signal = 0


class RSIDivergenceCPCV(RSIDivergence):
    """RSIDivergence for CPCV: sets self.signal instead of placing orders."""

    def next(self):
        if self.rsi is None or len(self.data.Close) < self.divergence_period + 1:
            return

        price_low_lookback = self.data.Low[-self.divergence_period : -1]
        prev_low_slice_idx = price_low_lookback.argmin()
        prev_low_idx = -(self.divergence_period - prev_low_slice_idx)

        price_high_lookback = self.data.High[-self.divergence_period : -1]
        prev_high_slice_idx = price_high_lookback.argmax()
        prev_high_idx = -(self.divergence_period - prev_high_slice_idx)

        if (
            self.data.Low[-1] < self.data.Low[prev_low_idx]
            and self.rsi[-1] > self.rsi[prev_low_idx]
        ):
            self.signal = 1
        elif (
            self.data.High[-1] > self.data.High[prev_high_idx]
            and self.rsi[-1] < self.rsi[prev_high_idx]
        ):
            self.signal = 0


class MultiIndicatorStrategyCPCV(MultiIndicatorStrategy):
    """MultiIndicatorStrategy for CPCV.

    The original strategy references self.position and self.trades (broker
    state) and uses trailing stop-loss orders.  The CPCV version uses only
    price and indicator comparisons, producing a stateless long/flat signal.
    """

    def next(self):
        price = self.data.Close[-1]
        if (
            self.upper_band is not None
            and self.sma_fast is not None
            and self.sma_slow is not None
            and price > self.upper_band[-1]
            and self.sma_fast[-1] > self.sma_slow[-1]
        ):
            self.signal = 1
        elif (
            self.lower_band is not None
            and self.sma_fast is not None
            and self.sma_slow is not None
            and price < self.lower_band[-1]
            and self.sma_fast[-1] < self.sma_slow[-1]
        ):
            self.signal = 0
