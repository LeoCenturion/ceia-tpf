# tests/data_acquisition/test_binance_client.py

import unittest
from unittest.mock import MagicMock, patch

from binance.exceptions import BinanceAPIException

from src.data_acquisition.binance_client import BinanceClient


class TestBinanceClient(unittest.TestCase):
    """
    Tests for the Binance API client.
    """

    @patch('src.data_acquisition.binance_client.Client')
    def test_binance_api_connectivity(self, mock_client):
        """
        Test that the Binance client can be instantiated and connects to the API.
        """
        mock_client.return_value.ping.return_value = {}
        client = BinanceClient(api_key='test_key', api_secret='test_secret')
        self.assertIsNotNone(client)
        mock_client.return_value.ping.assert_called_once()

    @patch('src.data_acquisition.binance_client.Client')
    def test_binance_api_connectivity_failure(self, mock_client):
        """
        Test that a BinanceAPIException is raised when the connection fails.
        """
        mock_client.return_value.ping.side_effect = BinanceAPIException(response=MagicMock(status_code=400, text="Connection failed"), status_code=400, text="Connection failed")
        with self.assertRaises(BinanceAPIException):
            BinanceClient(api_key='test_key', api_secret='test_secret')

    @patch('src.data_acquisition.binance_client.Client')
    def test_fetch_historical_data(self, mock_client):
        """
        Test fetching historical candlestick data.
        """
        mock_klines = [
            [1503388800000, '4235.43000000', '4235.43000000', '4235.43000000', '4235.43000000', '0.00000000', 1503388859999, '0.00000000', 0, '0.00000000', '0.00000000', '0']
        ]
        mock_client.return_value.get_historical_klines.return_value = mock_klines
        
        client = BinanceClient(api_key='test_key', api_secret='test_secret')
        data = client.fetch_historical_data('BTCUSDT', '1m', '1 day ago UTC')
        
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        mock_client.return_value.get_historical_klines.assert_called_once_with(
            symbol='BTCUSDT',
            interval='1m',
            start_str='1 day ago UTC',
            end_str=None
        )

    @patch('src.data_acquisition.binance_client.Client')
    def test_fetch_historical_data_with_date_range(self, mock_client):
        """
        Test fetching historical candlestick data with a specified date range.
        """
        mock_klines = [
            [1672531200000, '16541.39000000', '16541.39000000', '16541.39000000', '16541.39000000', '0.00000000', 1672531259999, '0.00000000', 0, '0.00000000', '0.00000000', '0']
        ]
        mock_client.return_value.get_historical_klines.return_value = mock_klines

        client = BinanceClient(api_key='test_key', api_secret='test_secret')
        data = client.fetch_historical_data('BTCUSDT', '1m', '2023-01-01', '2023-01-02')

        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        mock_client.return_value.get_historical_klines.assert_called_once_with(
            symbol='BTCUSDT',
            interval='1m',
            start_str='2023-01-01',
            end_str='2023-01-02'
        )

    @patch('src.data_acquisition.binance_client.Client')
    def test_fetch_historical_data_failure(self, mock_client):
        """
        Test that an empty list is returned when fetching historical data fails.
        """
        mock_client.return_value.get_historical_klines.side_effect = BinanceAPIException(response=MagicMock(status_code=400, text="Failed to fetch data"), status_code=400, text="Failed to fetch data")
        
        client = BinanceClient(api_key='test_key', api_secret='test_secret')
        data = client.fetch_historical_data('BTCUSDT', '1m', '1 day ago UTC')
        
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 0)

    @patch('src.data_acquisition.binance_client.Client')
    def test_fetch_historical_data_rate_limit(self, mock_client):
        """
        Test that the client handles rate limits with retries.
        """
        mock_client.return_value.get_historical_klines.side_effect = [
            BinanceAPIException(response=MagicMock(status_code=429, text="Rate limit exceeded"), status_code=429, text="Rate limit exceeded"),
            []
        ]
        
        client = BinanceClient(api_key='test_key', api_secret='test_secret')
        with patch('time.sleep') as mock_sleep:
            data = client.fetch_historical_data('BTCUSDT', '1m', '1 day ago UTC')
            self.assertEqual(len(data), 0)
            self.assertEqual(mock_client.return_value.get_historical_klines.call_count, 2)
            mock_sleep.assert_called_once_with(60)

    @patch('src.data_acquisition.binance_client.Client')
    def test_fetch_historical_data_invalid_request(self, mock_client):
        """
        Test that an invalid request returns an empty list.
        """
        mock_client.return_value.get_historical_klines.side_effect = BinanceAPIException(response=MagicMock(status_code=400, text="Invalid request"), status_code=400, text="Invalid request")
        
        client = BinanceClient(api_key='test_key', api_secret='test_secret')
        data = client.fetch_historical_data('BTCUSDT', '1m', '1 day ago UTC')
        
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 0)

if __name__ == '__main__':
    unittest.main()
