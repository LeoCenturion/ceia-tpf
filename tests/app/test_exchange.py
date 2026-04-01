import unittest
import os
from unittest.mock import patch
from src.app.exchange import BinanceClient

class TestBinanceClient(unittest.TestCase):
    @patch('src.app.exchange.Client')
    def test_successful_connection(self, mock_client):
        # Mock the binance client to avoid actual API calls
        mock_client.return_value.ping.return_value = {}
        
        client = BinanceClient(api_key='test_key', api_secret='test_secret', testnet=True)
        self.assertTrue(client.test_connection())

    @patch('src.app.exchange.Client')
    def test_failed_connection(self, mock_client):
        # Mock the binance client to raise an exception
        mock_client.return_value.ping.side_effect = Exception("Connection failed")
        
        client = BinanceClient(api_key='invalid_key', api_secret='invalid_secret', testnet=True)
        self.assertFalse(client.test_connection())

if __name__ == '__main__':
    unittest.main()
