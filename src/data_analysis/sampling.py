import pandas as pd


def cusum_filter(df: pd.DataFrame, h: float, price_col="close"):
    """
    Implements the CUSUM filter for event-based sampling.

    This filter samples events when the cumulative sum of price changes exceeds a
    predefined threshold 'h'. It captures both upward and downward movements.
    An event is sampled at time t if S_t >= h, at which point S_t is reset.

    Args:
        df (pd.DataFrame): DataFrame with price data, indexed by timestamp.
        h (float): The threshold for triggering a sampling event.
        price_col (str): The name of the column containing the price data.

    Returns:
        pd.DatetimeIndex: A DatetimeIndex of the timestamps where events were triggered.
    """
    t_events = []
    s_pos = 0
    s_neg = 0

    # The CUSUM filter is applied to deviations from an expected value.
    prices = df[price_col]
    # Calculate the expected value as an expanding window mean of prices.
    expected_values = prices.expanding().mean()

    # Align prices and expected_values to handle potential NaNs at the start
    common_index = prices.index.intersection(expected_values.index)
    aligned_prices = prices[common_index]
    aligned_expected_values = expected_values[common_index]

    for t, y in aligned_prices.items():
        expected_value = aligned_expected_values[t]
        s_pos = max(0, s_pos + y - expected_value)
        s_neg = min(0, s_neg + y - expected_value)

        if s_pos >= h:
            s_pos = 0
            s_neg = 0
            t_events.append(t)
        elif s_neg <= -h:
            s_neg = 0
            s_pos = 0
            t_events.append(t)

    return pd.DatetimeIndex(t_events)


def count_concurrent_labels(
    event_end_times: pd.Series, price_series_index: pd.DatetimeIndex
) -> pd.Series:
    """
    Computes the number of concurrent labels at each point in time.

    This is useful for understanding the degree of overlap in labeled events, which
    can inform sample weighting in machine learning models.

    Args:
        event_end_times (pd.Series): A series where the index is the timestamp of the
            event start, and the values are the timestamps of the event end.
            This series should not contain NaT/NaN values for end times.
        price_series_index (pd.DatetimeIndex): The index of the original price series,
            representing all potential observation points.

    Returns:
        pd.Series: A series indexed by `price_series_index`, where each value
            is the count of active labels at that point in time.
    """
    # Drop events with no end time
    event_end_times = event_end_times.dropna()

    # Create a DataFrame to track starts and ends of events
    starts = pd.Series(1, index=event_end_times.index)
    ends = pd.Series(-1, index=event_end_times.values)

    # Combine starts and ends into a single series
    concurrency_events = pd.concat([starts, ends]).sort_index()

    # Group by index to consolidate multiple events at the same timestamp
    concurrency_events = concurrency_events.groupby(level=0).sum()

    # Calculate cumulative sum to get the count of active events at any time
    concurrency_count = concurrency_events.cumsum()

    # Align with the full price series index
    concurrency_series = concurrency_count.reindex(price_series_index).fillna(
        method="ffill"
    )

    # The first values might be NaN if the first event starts after the price series begins.
    concurrency_series = concurrency_series.fillna(0)

    return concurrency_series.astype(int)
