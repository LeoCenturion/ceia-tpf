import unittest
import pandas as pd
from backtesting import Backtest, Strategy
from src.app.backtesting import run_backtest

class TestBacktesting(unittest.TestCase):
    def test_run_backtest(self):
        class SmaCross(Strategy):
            def init(self):
                self.sma1 = self.I(lambda x: pd.Series(x).rolling(10).mean(), self.data.Close)
                self.sma2 = self.I(lambda x: pd.Series(x).rolling(20).mean(), self.data.Close)

            def next(self):
                if self.sma1 > self.sma2:
                    self.buy()
                elif self.sma1 < self.sma2:
                    self.sell()

        data = {
            'Open': [10, 12, 15, 14, 13, 16, 18, 20, 19, 22, 25, 23, 21, 20, 19, 18, 17, 16],
            'High': [10, 12, 15, 14, 13, 16, 18, 20, 19, 22, 25, 23, 21, 20, 19, 18, 17, 16],
            'Low': [10, 12, 15, 14, 13, 16, 18, 20, 19, 22, 25, 23, 21, 20, 19, 18, 17, 16],
            'Close': [10, 12, 15, 14, 13, 16, 18, 20, 19, 22, 25, 23, 21, 20, 19, 18, 17, 16],
            'Volume': [100] * 18
        }
        df = pd.DataFrame(data)
        
        stats = run_backtest(df, SmaCross)
        self.assertIn('Return [%]', stats)

if __name__ == '__main__':
    unittest.main()
