from abc import ABC, abstractmethod
from typing import Literal, Any
import pandas as pd

from src.app.chronos_strategy import ChronosPalazzoStrategy
from src.app.strategy import Strategy, MACDStrategy


class StrategyFactory:
    @staticmethod
    def create_strategy(name: str, **params: Any) -> Strategy:
        if name == "macd":
            return MACDStrategy(**params)
        elif name == "ChronosPalazzo":
            return ChronosPalazzoStrategy(params)
        else:
            raise ValueError(f"Strategy '{name}' not found.")
