"""CPCV run for MultiIndicatorStrategy (Bollinger Bands + dual SMA trend filter)."""

import argparse
import logging

import pandas as pd

from src.backtesting.cpcv_runner import run_cpcv_for_strategy
from src.backtesting.strategies.cpcv_strategies import MultiIndicatorStrategyCPCV
from src.data_analysis.data_analysis import adjust_data_to_ubtc, fetch_historical_data

logging.basicConfig(level=logging.INFO)


def run(
    data_path: str,
    start_date: str = "2020-01-01T00:00:00Z",
    bb_window: int = 20,
    bb_std: float = 2.0,
    fast_sma_window: int = 50,
    slow_sma_window: int = 100,
    n_groups: int = 10,
    k_test_groups: int = 2,
    embargo_pct: float = 0.01,
    experiment_name: str = "CPCV_MultiIndicator",
):
    data = fetch_historical_data(data_path=data_path, start_date=start_date, timeframe="1h")
    data = adjust_data_to_ubtc(data)
    t1 = pd.Series(data.index[1:], index=data.index[:-1])
    data = data.iloc[:-1]

    return run_cpcv_for_strategy(
        data=data,
        t1=t1,
        strategy_class=MultiIndicatorStrategyCPCV,
        strategy_params={
            "bb_window": bb_window,
            "bb_std": bb_std,
            "fast_sma_window": fast_sma_window,
            "slow_sma_window": slow_sma_window,
        },
        n_groups=n_groups,
        k_test_groups=k_test_groups,
        embargo_pct=embargo_pct,
        experiment_name=experiment_name,
    )


def main():
    parser = argparse.ArgumentParser(description="CPCV for MultiIndicatorStrategy")
    parser.add_argument("--data-path", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--start-date", default="2020-01-01T00:00:00Z")
    parser.add_argument("--bb-window", type=int, default=20, help="Bollinger Band period")
    parser.add_argument("--bb-std", type=float, default=2.0, help="Bollinger Band std deviations")
    parser.add_argument("--fast-sma-window", type=int, default=50, help="Fast SMA window")
    parser.add_argument("--slow-sma-window", type=int, default=100, help="Slow SMA window")
    parser.add_argument("--n-groups", type=int, default=10)
    parser.add_argument("--k-test-groups", type=int, default=2)
    parser.add_argument("--embargo-pct", type=float, default=0.01)
    parser.add_argument("--experiment-name", default="CPCV_MultiIndicator")
    args = parser.parse_args()

    run(
        data_path=args.data_path,
        start_date=args.start_date,
        bb_window=args.bb_window,
        bb_std=args.bb_std,
        fast_sma_window=args.fast_sma_window,
        slow_sma_window=args.slow_sma_window,
        n_groups=args.n_groups,
        k_test_groups=args.k_test_groups,
        embargo_pct=args.embargo_pct,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
