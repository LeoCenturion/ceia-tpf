"""CPCV run for MACD."""

import argparse
import logging

import pandas as pd

from src.backtesting.cpcv_runner import run_cpcv_for_strategy
from src.backtesting.strategies.cpcv_strategies import MACDCPCV
from src.data_analysis.data_analysis import adjust_data_to_ubtc, fetch_historical_data

logging.basicConfig(level=logging.INFO)


def run(
    data_path: str,
    start_date: str = "2020-01-01T00:00:00Z",
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
    n_groups: int = 10,
    k_test_groups: int = 2,
    embargo_pct: float = 0.01,
    experiment_name: str = "CPCV_MACD",
):
    data = fetch_historical_data(data_path=data_path, start_date=start_date, timeframe="1h")
    data = adjust_data_to_ubtc(data)
    t1 = pd.Series(data.index[1:], index=data.index[:-1])
    data = data.iloc[:-1]

    return run_cpcv_for_strategy(
        data=data,
        t1=t1,
        strategy_class=MACDCPCV,
        strategy_params={
            "fast_span": fast_span,
            "slow_span": slow_span,
            "signal_span": signal_span,
        },
        n_groups=n_groups,
        k_test_groups=k_test_groups,
        embargo_pct=embargo_pct,
        experiment_name=experiment_name,
    )


def main():
    parser = argparse.ArgumentParser(description="CPCV for MACD")
    parser.add_argument("--data-path", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--start-date", default="2020-01-01T00:00:00Z")
    parser.add_argument("--fast-span", type=int, default=12, help="Fast EMA span")
    parser.add_argument("--slow-span", type=int, default=26, help="Slow EMA span")
    parser.add_argument("--signal-span", type=int, default=9, help="Signal line EMA span")
    parser.add_argument("--n-groups", type=int, default=10)
    parser.add_argument("--k-test-groups", type=int, default=2)
    parser.add_argument("--embargo-pct", type=float, default=0.01)
    parser.add_argument("--experiment-name", default="CPCV_MACD")
    args = parser.parse_args()

    run(
        data_path=args.data_path,
        start_date=args.start_date,
        fast_span=args.fast_span,
        slow_span=args.slow_span,
        signal_span=args.signal_span,
        n_groups=args.n_groups,
        k_test_groups=args.k_test_groups,
        embargo_pct=args.embargo_pct,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
