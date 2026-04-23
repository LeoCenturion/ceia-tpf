from abc import ABC, abstractmethod
from typing import Literal, Any
import pandas as pd


class Strategy(ABC):
    @abstractmethod
    def get_signal(self, data: pd.DataFrame) -> Literal["BUY", "SELL", "HOLD"]:
        pass

    @abstractmethod
    def get_order_size(self) -> float:
        pass


class MACDStrategy(Strategy):
    def __init__(self, **params: Any):
        self.fast_period: int = params.get("fast_period", 12)
        self.slow_period: int = params.get("slow_period", 26)
        self.signal_period: int = params.get("signal_period", 9)

    def get_signal(self, data: pd.DataFrame) -> Literal["BUY", "SELL", "HOLD"]:
        # Calculate Fast and Slow EMAs
        fast_ema: pd.Series = pd.Series(
            data["close"].ewm(span=self.fast_period, adjust=False).mean()
        )
        slow_ema: pd.Series = pd.Series(
            data["close"].ewm(span=self.slow_period, adjust=False).mean()
        )

        # Calculate MACD
        data["macd"] = fast_ema - slow_ema

        # Calculate Signal Line
        data["macds"] = pd.Series(
            data["macd"].ewm(span=self.signal_period, adjust=False).mean()
        )

        # Calculate MACD Histogram
        data["macdh"] = data["macd"] - data["macds"]

        # Generate Signal
        if data["macdh"].iloc[-1] > 0 and data["macdh"].iloc[-2] < 0:
            return "BUY"
        elif data["macdh"].iloc[-1] < 0 and data["macdh"].iloc[-2] > 0:
            return "SELL"
        else:
            return "HOLD"

    def get_order_size(self) -> float:
        # For simplicity, we'll return a fixed size.
        # In a real bot, this would be based on risk management.
        return 0.01
