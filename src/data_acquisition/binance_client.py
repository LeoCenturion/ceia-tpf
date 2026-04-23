# src/data_acquisition/binance_client.py

import logging
from binance import Client
from binance.exceptions import BinanceAPIException

import time

class BinanceClient:
    """
    A client for interacting with the Binance API.
    """

    def __init__(self, api_key: str, api_secret: str, max_retries: int = 3, retry_delay: int = 60):
        """
        Initializes the Binance client.

        Args:
            api_key (str): The Binance API key.
            api_secret (str): The Binance API secret.
            max_retries (int): The maximum number of retries for rate limit errors.
            retry_delay (int): The delay in seconds between retries.
        """
        self.logger = logging.getLogger(__name__)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        try:
            self.client = Client(api_key, api_secret)
            self.client.ping()
            self.logger.info("Successfully connected to Binance API.")
        except BinanceAPIException as e:
            self.logger.error(f"Failed to connect to Binance API: {e}")
            raise

    def fetch_historical_data(self, symbol: str, interval: str, start_str: str, end_str: str = None) -> list:
        """
        Fetches historical candlestick data from Binance.

        Args:
            symbol (str): The symbol to fetch data for (e.g., 'BTCUSDT').
            interval (str): The interval of the candlesticks (e.g., '1m', '1h').
            start_str (str): The start time for the data (e.g., '1 day ago UTC').

        Returns:
            list: A list of candlestick data.
        """
        if end_str:
            self.logger.info(f"Fetching historical data for {symbol} with interval {interval} from {start_str} to {end_str}.")
        else:
            self.logger.info(f"Fetching historical data for {symbol} with interval {interval} from {start_str}.")
        
        for i in range(self.max_retries):
            try:
                return self.client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=start_str,
                    end_str=end_str
                )
            except BinanceAPIException as e:
                if e.status_code == 429:
                    self.logger.warning(f"Rate limit exceeded. Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"Error fetching historical data: {e}")
                    return []
        
        self.logger.error("Max retries exceeded. Failed to fetch historical data.")
        return []
