from backtesting import Backtest

def run_backtest(data, strategy):
    bt = Backtest(data, strategy, cash=10000, commission=.002)
    stats = bt.run()
    return stats
