import pandas as pd


def _aggregate_to_bars(df: pd.DataFrame, metric_col: str, threshold: float) -> pd.DataFrame:
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

    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    bars = []
    start_idx = 0
    cumulative_metric = 0.0

    df_reset = df.reset_index(drop=True)

    for i in range(len(df_reset)):
        cumulative_metric += df_reset.at[i, metric_col]

        if cumulative_metric >= threshold:
            bar_slice = df_reset.iloc[start_idx : i + 1]

            bars.append({
                'date': bar_slice['date'].iloc[-1],
                'open': bar_slice['open'].iloc[0],
                'high': bar_slice['high'].max(),
                'low': bar_slice['low'].min(),
                'close': bar_slice['close'].iloc[-1],
                'Volume BTC': bar_slice['Volume BTC'].sum(),
                'Volume USDT': bar_slice['Volume USDT'].sum(),
                'tradeCount': bar_slice['tradeCount'].sum(),
            })

            start_idx = i + 1
            cumulative_metric = 0.0

    if not bars:
        return pd.DataFrame()

    result_df = pd.DataFrame(bars)
    if "date" in result_df.columns:
        result_df = result_df.set_index('date')
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
    return _aggregate_to_bars(df, 'Volume BTC', volume_threshold)


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
    return _aggregate_to_bars(df, 'Volume USDT', dollar_threshold)


def create_price_change_bars(df: pd.DataFrame, price_change_threshold: float) -> pd.DataFrame:
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
    df_copy['abs_pct_change'] = df_copy['close'].pct_change().abs()
    # The first value will be NaN, fill it with 0 so it doesn't break the cumulative sum
    df_copy['abs_pct_change'] = df_copy['abs_pct_change'].fillna(0)
    return _aggregate_to_bars(df_copy, 'abs_pct_change', price_change_threshold)

# AI add a main function that takes a csv 
