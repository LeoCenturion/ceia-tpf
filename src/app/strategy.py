from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    @abstractmethod
    def get_signal(self, data):
        pass

    @abstractmethod
    def get_order_size(self):
        pass

class MACDStrategy(Strategy):
    def __init__(self, **params):
        self.fast_period = params.get('fast_period', 12)
        self.slow_period = params.get('slow_period', 26)
        self.signal_period = params.get('signal_period', 9)

    def get_signal(self, data):
        # Calculate Fast and Slow EMAs
        fast_ema = data['close'].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate MACD
        data['macd'] = fast_ema - slow_ema
        
        # Calculate Signal Line
        data['macds'] = data['macd'].ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate MACD Histogram
        data['macdh'] = data['macd'] - data['macds']
        
        # Generate Signal
        if data['macdh'].iloc[-1] > 0 and data['macdh'].iloc[-2] < 0:
            return "BUY"
        elif data['macdh'].iloc[-1] < 0 and data['macdh'].iloc[-2] > 0:
            return "SELL"
        else:
            return "HOLD"

    def get_order_size(self):
        # For simplicity, we'll return a fixed size.
        # In a real bot, this would be based on risk management.
        return 0.01

class StrategyFactory:
    @staticmethod
    def create_strategy(name, **params):
        if name == 'macd':
            return MACDStrategy(**params)
        else:
            raise ValueError(f"Strategy '{name}' not found.")
