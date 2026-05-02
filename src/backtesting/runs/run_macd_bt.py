"""Simple backtesting.py run for the MACD strategy."""

import argparse
import logging

from backtesting import Backtest

from src.backtesting.strategies.trading_strategies import MACD
from src.data_analysis.data_analysis import adjust_data_to_ubtc, fetch_historical_data

logging.basicConfig(level=logging.INFO)


def run(
    data_path: str,
    start_date: str = "2020-01-01T00:00:00Z",
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
):
    data = fetch_historical_data(data_path=data_path, start_date=start_date, timeframe="1h")
    data = adjust_data_to_ubtc(data)

    bt = Backtest(data, MACD, cash=10000, commission=0.001)
    stats = bt.run(fast_span=fast_span, slow_span=slow_span, signal_span=signal_span)
    print(stats)
    bt.plot()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Backtest MACD strategy")
    parser.add_argument("--data-path", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--start-date", default="2020-01-01T00:00:00Z")
    parser.add_argument("--fast-span", type=int, default=12)
    parser.add_argument("--slow-span", type=int, default=26)
    parser.add_argument("--signal-span", type=int, default=9)
    args = parser.parse_args()

    run(
        data_path=args.data_path,
        start_date=args.start_date,
        fast_span=args.fast_span,
        slow_span=args.slow_span,
        signal_span=args.signal_span,
    )


if __name__ == "__main__":
    main()
