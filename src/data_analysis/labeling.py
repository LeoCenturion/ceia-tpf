"""Functions for creating labels for machine learning models in finance."""
import numpy as np
import pandas as pd


def create_fixed_time_horizon_labels(
    close: pd.Series, look_forward: int, pt_pct: float, sl_pct: float
) -> pd.Series:
    """
    Creates labels based on a fixed-time horizon method (fixed labeling).

    A label is assigned based on the return after a fixed number of periods.
    - 1: If the return is >= profit_take_pct.
    - -1: If the return is <= -stop_loss_pct.
    - 0: Otherwise.

    Args:
        close (pd.Series): Series of closing prices.
        look_forward (int): The number of periods to look forward for the return calculation.
        pt_pct (float): The percentage threshold for a positive label (e.g., 0.02 for 2%).
        sl_pct (float): The percentage threshold for a negative label (e.g., 0.02 for 2%).

    Returns:
        pd.Series: A series of labels (1, -1, 0) with the same index as `close`.
    """
    future_returns = close.shift(-look_forward) / close - 1

    labels = pd.Series(0, index=close.index, dtype=int)
    labels.loc[future_returns >= pt_pct] = 1
    labels.loc[future_returns <= -sl_pct] = -1

    # Events that didn't reach either threshold are labeled 0.
    # The last `look_forward` periods cannot be labeled as their future is unknown.
    labels.iloc[-look_forward:] = np.nan

    return labels


def create_volatility_adjusted_labels(
    close: pd.Series, look_forward: int, vol_window: int, vol_multiplier: float
) -> pd.Series:
    """
    Creates labels based on a fixed-time horizon with dynamic, volatility-adjusted thresholds.

    For each point, it calculates rolling volatility and sets the profit-take and
    stop-loss thresholds as a multiple of this volatility. A label is then assigned
    based on the return after a fixed `look_forward` period.

    - 1: If the return is >= (volatility * vol_multiplier).
    - -1: If the return is <= -(volatility * vol_multiplier).
    - 0: Otherwise.

    Args:
        close (pd.Series): Series of closing prices.
        look_forward (int): The number of periods to look forward for the return calculation.
        vol_window (int): The window size for calculating rolling volatility.
        vol_multiplier (float): The multiplier for volatility to set thresholds.

    Returns:
        pd.Series: A series of labels (1, -1, 0) with the same index as `close`.
    """
    # Calculate daily returns to compute volatility
    returns = close.pct_change()

    # Calculate rolling volatility
    volatility = returns.rolling(window=vol_window).std()

    # Calculate dynamic thresholds
    pt_threshold = volatility * vol_multiplier
    sl_threshold = -volatility * vol_multiplier

    # Calculate future returns
    future_returns = close.shift(-look_forward) / close - 1

    # Create labels
    labels = pd.Series(0, index=close.index, dtype=int)
    labels.loc[future_returns >= pt_threshold] = 1
    labels.loc[future_returns <= sl_threshold] = -1

    # Events that didn't reach either threshold are labeled 0.
    # The last `look_forward` periods cannot be labeled.
    labels.iloc[-look_forward:] = np.nan

    return labels


def create_triple_barrier_labels(
    close: pd.Series,
    volatility: pd.Series,
    look_forward: int,
    pt_sl_multipliers: tuple,
) -> pd.Series:
    """
    Creates labels using the triple-barrier method (dynamic labeling).

    For each timestamp, it sets three barriers:
    1. Upper barrier (profit take): based on volatility.
    2. Lower barrier (stop loss): based on volatility.
    3. Vertical barrier (time limit): `look_forward` periods.

    The label is determined by which barrier is hit first.
    - 1: Upper barrier hit first.
    - -1: Lower barrier hit first.
    - 0: Vertical barrier hit first (timeout).

    Args:
        close (pd.Series): Series of closing prices.
        volatility (pd.Series): Series of volatility (e.g., rolling std of returns).
        look_forward (int): Maximum holding period (vertical barrier).
        pt_sl_multipliers (tuple): A tuple of (profit_take_multiplier, stop_loss_multiplier).
            These are multiplied by volatility to set the barriers.

    Returns:
        pd.Series: A series of labels (1, -1, 0) with the same index as `close`.
    """
    pt_mult, sl_mult = pt_sl_multipliers
    labels = pd.Series(np.nan, index=close.index)

    # Align volatility index with close index to prevent mismatches
    volatility = volatility.reindex(close.index).fillna(method="ffill")

    for i in range(len(close) - look_forward):
        entry_price = close.iloc[i]
        vol = volatility.iloc[i]

        if pd.isna(vol) or vol == 0:
            continue

        upper_barrier = entry_price * (1 + pt_mult * vol)
        lower_barrier = entry_price * (1 - sl_mult * vol)

        path = close.iloc[i + 1 : i + 1 + look_forward]

        # Find first touch time for each horizontal barrier
        touch_upper = path[path >= upper_barrier]
        time_to_upper = touch_upper.index[0] if not touch_upper.empty else None

        touch_lower = path[path <= lower_barrier]
        time_to_lower = touch_lower.index[0] if not touch_lower.empty else None

        # Determine label based on which barrier was hit first
        if time_to_upper is None and time_to_lower is None:
            labels.iloc[i] = 0  # Vertical barrier hit
        elif time_to_upper is not None and (
            time_to_lower is None or time_to_upper <= time_to_lower
        ):
            labels.iloc[i] = 1  # Upper barrier hit
        elif time_to_lower is not None and (
            time_to_upper is None or time_to_lower < time_to_upper
        ):
            labels.iloc[i] = -1  # Lower barrier hit
        else:
            labels.iloc[i] = 0  # Should not be reached, but as a fallback, timeout.

    return labels
