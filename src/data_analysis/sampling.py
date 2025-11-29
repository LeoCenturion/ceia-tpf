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

    # Using simple returns
    diff = df[price_col].diff().dropna()

    for t, y in diff.items():
        s_pos = max(0, s_pos + y)
        s_neg = min(0, s_neg + y)

        if s_pos >= h:
            s_pos = 0
            s_neg = 0
            t_events.append(t)
        elif s_neg <= -h:
            s_neg = 0
            s_pos = 0
            t_events.append(t)

    return pd.DatetimeIndex(t_events)
