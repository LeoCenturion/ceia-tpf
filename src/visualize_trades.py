import argparse
import pandas as pd
import mplfinance as mpf
from binance.client import Client
import os

# Use environment variables for API keys if needed for non-public data, though not required for klines
# API_KEY = os.environ.get('BINANCE_API_KEY')
# API_SECRET = os.environ.get('BINANCE_API_SECRET')

def get_historical_data(symbol, interval, start_str, end_str):
    """
    Fetches historical klines from Binance.
    """
    # No API key needed for this endpoint
    client = Client()

    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convert to numeric and set index
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def plot_trades(transactions_file, symbol):
    """
    Plots candlestick chart with trade markers.
    """
    try:
        trades_df = pd.read_csv(transactions_file)
    except FileNotFoundError:
        print(f"Error: Transactions file not found at '{transactions_file}'")
        return

    if trades_df.empty:
        print("Transactions file is empty. Nothing to plot.")
        return

    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

    # Determine date range for fetching kline data
    start_date = trades_df['timestamp'].min() - pd.Timedelta(hours=1)
    end_date = trades_df['timestamp'].max() + pd.Timedelta(hours=1)

    print(f"Fetching historical data for {symbol} from {start_date} to {end_date}...")
    
    # Binance API expects strings for dates
    ohlc_df = get_historical_data(symbol, Client.KLINE_INTERVAL_1MINUTE, str(start_date), str(end_date))

    if ohlc_df.empty:
        print("Could not fetch historical data for the specified range. Cannot plot.")
        return

    # Create signals for plotting
    buy_signals = trades_df[trades_df['side'] == 'BUY']
    sell_signals = trades_df[trades_df['side'] == 'SELL']

    # Create additional plots for buy/sell markers
    # We need to align these with the ohlc_df index
    buy_markers = pd.Series(float('nan'), index=ohlc_df.index)
    sell_markers = pd.Series(float('nan'), index=ohlc_df.index)

    # Use asof to find the nearest kline timestamp for each trade
    for index, trade in buy_signals.iterrows():
        closest_index = ohlc_df.index.asof(trade['timestamp'])
        if pd.notna(closest_index):
            buy_markers[closest_index] = trade['price']
    
    for index, trade in sell_signals.iterrows():
        closest_index = ohlc_df.index.asof(trade['timestamp'])
        if pd.notna(closest_index):
            sell_markers[closest_index] = trade['price']
            
    add_plots = []
    # Check if there are any valid buy markers to plot before adding them
    if buy_markers.notna().any():
        add_plots.append(mpf.make_addplot(buy_markers, type='scatter', marker='^', color='g', markersize=100))
    
    # Check if there are any valid sell markers to plot
    if sell_markers.notna().any():
        add_plots.append(mpf.make_addplot(sell_markers, type='scatter', marker='v', color='r', markersize=100))

    print("Generating plot...")
    
    # Plotting
    mpf.plot(ohlc_df, 
             type='candle', 
             style='yahoo',
             title=f'{symbol} Trading Activity',
             ylabel='Price (USDT)',
             addplot=add_plots,
             volume=True,
             figratio=(16,9),
             panel_ratios=(3, 1))

    print("Plot displayed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize trading bot transactions.')
    parser.add_argument('transactions_file', type=str, default='transactions.csv', nargs='?',
                        help='Path to the transaction log CSV file (default: transactions.csv).')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='The trading symbol (default: BTCUSDT).')
    
    args = parser.parse_args()
    
    plot_trades(args.transactions_file, args.symbol)
