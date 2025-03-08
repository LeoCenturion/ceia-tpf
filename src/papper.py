import os
import json
from datetime import datetime, timedelta
import requests
import ccxt
import pandas as pd
import numpy as np

class BinancePaperTradingBot:
    initial_balance: int
    current_balance: int
    leverage: int
    def __init__(self, initial_balance=1000, leverage=1):
        """
        Initialize the paper trading bot

        :param initial_balance: Starting capital in USDT
        :param leverage: Trading leverage (1-25x)
        """
        # Trading configuration
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.leverage = max(1, min(leverage, 25))

        # Position tracking
        self.open_position = None
        self.trade_history = []

        # Binance API (using public endpoints for data)
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'  # Use futures market
            }
        })

        # Ensure data persistence
        self.load_state()

    def get_current_price(self, symbol='BTC/USDT'):
        """
        Fetch current market price

        :param symbol: Trading pair
        :return: Current market price
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            print(f"Price fetch error: {e}")
            return None

    def get_historical_data(self, symbol='BTC/USDT', timeframe='1h', limit=100):
        """
        Fetch historical price data

        :param symbol: Trading pair
        :param timeframe: Candle timeframe
        :param limit: Number of candles to fetch
        :return: DataFrame with OHLCV data
        """
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def simple_trading_strategy(self, data):
        """
        Basic trading strategy based on moving average crossover

        :param data: Historical price data
        :return: Trading signal (long, short, or hold)
        """
        data['MA50'] = data['close'].rolling(window=50).mean()
        data['MA200'] = data['close'].rolling(window=200).mean()

        latest = data.iloc[-1]

        if latest['MA50'] > latest['MA200']:
            return 'long'
        elif latest['MA50'] < latest['MA200']:
            return 'short'
        return 'hold'

    def execute_trade(self, symbol='BTC/USDT', signal=None):
        """
        Execute paper trade based on signal

        :param symbol: Trading pair
        :param signal: Trading signal
        :return: Trade details
        """
        current_price = self.get_current_price(symbol)
        if not current_price:
            return None

        trade = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'price': current_price,
            'signal': signal,
            'position_size': self.current_balance * self.leverage / current_price
        }


        if signal == 'long':
            trade['type'] = 'buy'
            self.open_position = trade
        elif signal == 'short':
            trade['type'] = 'sell'
            self.open_position = trade

        self.trade_history.append(trade)
        self.save_state()

        return trade

    def calculate_profit(self, close_price):
        """
        Calculate profit/loss for an open position

        :param close_price: Price at trade closure
        :return: Profit percentage
        """
        if not self.open_position:
            return 0

        entry_price = self.open_position['price']
        trade_type = self.open_position['signal']

        if trade_type == 'long':
            profit_percentage = ((close_price - entry_price) / entry_price) * 100
        else:  # short
            profit_percentage = ((entry_price - close_price) / entry_price) * 100

        return profit_percentage

    def save_state(self, filename='bot_state.json'):
        """
        Save bot's current state to a JSON file
        """
        state = {
            'current_balance': self.current_balance,
            'open_position': self.open_position,
            'trade_history': self.trade_history
        }

        with open(filename, 'w') as f:
            json.dump(state, f, default=str)

    def load_state(self, filename='bot_state.json'):
        """
        Load bot's previous state from JSON file
        """
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                state = json.load(f)
                self.current_balance = state.get('current_balance', self.initial_balance)
                self.open_position = state.get('open_position')
                self.trade_history = state.get('trade_history', [])

def main():
    # Initialize bot
    bot = BinancePaperTradingBot(initial_balance=1000, leverage=5)

    # Fetch historical data
    historical_data = bot.get_historical_data()

    # Generate trading signal
    signal = bot.simple_trading_strategy(historical_data)

    # Execute trade
    trade = bot.execute_trade(signal=signal)

    print(f"Trade Executed: {trade}")

if __name__ == "__main__":
    main()

# Requirements:
# pip install ccxt pandas requests numpy
