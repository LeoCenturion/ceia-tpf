"""
Functions for creating volume, dollar, and price change bars from tick data.
"""

import argparse
from typing import Tuple
import pandas as pd


def _aggregate_to_bars(
    df: pd.DataFrame, metric_col: str, threshold: float
) -> pd.DataFrame:
    """
    Aggregates tick data into bars based on a metric threshold.

    Args:
        df (pd.DataFrame): Input DataFrame with tick data.
        metric_col (str): The column to use for thresholding (e.g., 'Volume BTC' or 'Volume USDT').
        threshold (float): The threshold value for creating a new bar.

    Returns:
        pd.DataFrame: A DataFrame of aggregated bars.
    """
    if df.empty:
        return pd.DataFrame()

    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    bars = []
    start_idx = 0
    cumulative_metric = 0.0

    df_reset = df.reset_index(drop=True)

    for i in range(len(df_reset)):
        cumulative_metric += df_reset.at[i, metric_col]

        if cumulative_metric >= threshold:
            bar_slice = df_reset.iloc[start_idx : i + 1]

            bars.append(
                {
                    "date": bar_slice["date"].iloc[-1],
                    "open": bar_slice["open"].iloc[0],
                    "high": bar_slice["high"].max(),
                    "low": bar_slice["low"].min(),
                    "close": bar_slice["close"].iloc[-1],
                    "Volume BTC": bar_slice["Volume BTC"].sum(),
                    "Volume USDT": bar_slice["Volume USDT"].sum(),
                    "tradeCount": bar_slice["tradeCount"].sum(),
                }
            )

            start_idx = i + 1
            cumulative_metric = 0.0

    if not bars:
        return pd.DataFrame()

    result_df = pd.DataFrame(bars)
    if "date" in result_df.columns:
        result_df = result_df.set_index("date")
    return result_df


def create_volume_bars(df: pd.DataFrame, volume_threshold: float) -> pd.DataFrame:
    """
    Creates volume bars from a DataFrame of tick data.

    A new bar is created whenever the cumulative volume traded reaches the volume_threshold.
    The input DataFrame should have columns including: date, open, high, low, close,
    'Volume BTC', 'Volume USDT', and tradeCount.

    Args:
        df (pd.DataFrame): DataFrame with tick data.
        volume_threshold (float): The amount of BTC volume to accumulate for each bar.

    Returns:
        pd.DataFrame: A DataFrame containing the volume bars with OHLCV data.
    """
    return _aggregate_to_bars(df, "Volume BTC", volume_threshold)


def create_dollar_bars(df: pd.DataFrame, dollar_threshold: float) -> pd.DataFrame:
    """
    Creates dollar bars from a DataFrame of tick data.

    A new bar is created whenever the cumulative dollar value traded reaches the dollar_threshold.
    The input DataFrame should have columns including: date, open, high, low, close,
    'Volume BTC', 'Volume USDT', and tradeCount.

    Args:
        df (pd.DataFrame): DataFrame with tick data.
        dollar_threshold (float): The amount of USDT volume to accumulate for each bar.

    Returns:
        pd.DataFrame: A DataFrame containing the dollar bars with OHLCV data.
    """
    return _aggregate_to_bars(df, "Volume USDT", dollar_threshold)


def create_price_change_bars(
    df: pd.DataFrame, price_change_threshold: float
) -> pd.DataFrame:
    """
    Creates bars based on cumulative absolute price change.

    A new bar is formed whenever the cumulative absolute percentage change of the close price
    reaches the price_change_threshold. The change is calculated as a fraction (e.g., 0.01 for 1%).
    The input DataFrame should have columns including: date, open, high, low, close,
    'Volume BTC', 'Volume USDT', and tradeCount.

    Args:
        df (pd.DataFrame): DataFrame with tick or bar data.
        price_change_threshold (float): The amount of absolute fractional price change to accumulate.

    Returns:
        pd.DataFrame: A DataFrame containing the new bars.
    """
    df_copy = df.copy()
    # Calculate fractional change, not percentage
    df_copy["abs_pct_change"] = df_copy["close"].pct_change().abs()
    # The first value will be NaN, fill it with 0 so it doesn't break the cumulative sum
    df_copy["abs_pct_change"] = df_copy["abs_pct_change"].fillna(0)
    return _aggregate_to_bars(df_copy, "abs_pct_change", price_change_threshold)


def main():
    """
    Main function to run the bar aggregation script from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Aggregate time-series data into different bar types."
    )
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument(
        "bar_type",
        choices=["volume", "dollar", "price_change"],
        help="Type of bar to create.",
    )
    parser.add_argument(
        "threshold", type=float, help="Threshold value for bar creation."
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        help=("Path to the output CSV file. If not provided, prints to console."),
    )

    args = parser.parse_args()

    print(f"Loading data from {args.input_csv}...")
    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_csv}")
        return

    # The bar creation functions expect a 'date' column.
    if "date" not in df.columns:
        print(
            f"Error: Input CSV must contain a 'date' column. Found columns: {df.columns.tolist()}"
        )
        return

    print(f"Creating {args.bar_type} bars with threshold {args.threshold}...")

    if args.bar_type == "volume":
        result_df = create_volume_bars(df, args.threshold)
    elif args.bar_type == "dollar":
        result_df = create_dollar_bars(df, args.threshold)
    elif args.bar_type == "price_change":
        result_df = create_price_change_bars(df, args.threshold)
    else:
        # This case should not be reached due to argparse choices
        print(f"Error: Unknown bar type '{args.bar_type}'")
        return

    print(f"Successfully created {len(result_df)} bars.")

    if args.output_csv:
        print(f"Saving results to {args.output_csv}...")
        result_df.to_csv(args.output_csv, index=True)
        print("Done.")
    else:
        print("\n--- Resulting Bars ---")
        print(result_df.to_string())


def _get_signed_ticks(price_series: pd.Series) -> pd.Series:
    """
    Computes tick signs based on price changes.
    +1 for an uptick, -1 for a downtick.
    If price is unchanged, the previous sign is carried forward.
    """
    price_diffs = price_series.diff()
    tick_signs = price_diffs.copy()
    tick_signs.loc[price_diffs > 0] = 1
    tick_signs.loc[price_diffs < 0] = -1
    tick_signs.loc[price_diffs == 0] = None
    tick_signs = tick_signs.ffill()
    # Fill the first potential NaN with 1 (assuming an initial uptick)
    tick_signs = tick_signs.fillna(1).astype(int)
    return tick_signs


def create_tick_imbalance_bars(
    df: pd.DataFrame,
    initial_bar_size_estimate: int = 1,
    span_bar_size: int = 20,
    span_tick_imbalance: int = 20,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Creates tick imbalance bars (TIBs) with a dynamic threshold.

    A new bar is formed when the absolute cumulative tick imbalance exceeds a
    dynamic threshold. The threshold is based on the expected imbalance for the
    next bar, calculated from EWMA of recent bar sizes and tick imbalances.

    Expected Imbalance = E[T] * E[b_t], where:
    - E[T] is the EWMA of ticks per bar.
    - E[b_t] is the EWMA of tick signs (P(b_t=1) - P(b_t=-1)).

    Args:
        df (pd.DataFrame): DataFrame with tick data. Must include 'close' price.
        initial_bar_size_estimate (int): Initial estimate for ticks per bar.
        span_bar_size (int): EWMA span for calculating the expected bar size.
        span_tick_imbalance (int): EWMA span for calculating the expected tick imbalance.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - A DataFrame of the tick imbalance bars.
            - A Series of the imbalance thresholds over time, indexed like the input df.
    """
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    bars = []
    thresholds_over_time = []
    start_idx = 0
    cumulative_imbalance = 0.0

    df_reset = df.reset_index(drop=True)

    tick_signs = _get_signed_ticks(df_reset["close"])

    # EWMA parameters
    alpha_bar_size = 2 / (span_bar_size + 1)

    # Pre-compute the EWMA of tick imbalances
    ewma_tick_imbalances = tick_signs.ewm(span=span_tick_imbalance, adjust=False).mean()

    # EWMA state variables
    ewma_bar_size = float(initial_bar_size_estimate)

    for i, tick_sign in enumerate(tick_signs):
        ewma_tick_imbalance = ewma_tick_imbalances.iloc[i]
        expected_imbalance_threshold = abs(ewma_bar_size * ewma_tick_imbalance)
        thresholds_over_time.append(expected_imbalance_threshold)
        cumulative_imbalance += tick_sign

        if (
            expected_imbalance_threshold > 0
            and abs(cumulative_imbalance) >= expected_imbalance_threshold
        ):
            bar_slice = df_reset.iloc[start_idx : i + 1]
            bars.append(
                {
                    "date": bar_slice["date"].iloc[-1],
                    "open": bar_slice["open"].iloc[0],
                    "high": bar_slice["high"].max(),
                    "low": bar_slice["low"].min(),
                    "close": bar_slice["close"].iloc[-1],
                    "Volume BTC": bar_slice["Volume BTC"].sum(),
                    "Volume USDT": bar_slice["Volume USDT"].sum(),
                    "tradeCount": bar_slice["tradeCount"].sum(),
                }
            )

            # Update EWMA of bar size with the size of the bar just formed
            current_bar_size = i + 1 - start_idx
            ewma_bar_size = ((1 - alpha_bar_size) * ewma_bar_size) + (
                alpha_bar_size * current_bar_size
            )

            # Reset for next bar
            start_idx = i + 1
            cumulative_imbalance = 0.0

    thresholds_series = pd.Series(thresholds_over_time, index=df.index, dtype=float)

    if not bars:
        return pd.DataFrame(), thresholds_series

    result_df = pd.DataFrame(bars)
    if "date" in result_df.columns:
        result_df = result_df.set_index("date")
    return result_df, thresholds_series


if __name__ == "__main__":
    main()
