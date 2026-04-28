import unittest
from unittest.mock import patch

from src.app.exchange import BinanceClient


class TestBinanceClient(unittest.TestCase):
    @patch("src.app.exchange.Client")
    def test_successful_connection(self, mock_client):
        # Mock the binance client to avoid actual API calls
        mock_client.return_value.ping.return_value = {}

        client = BinanceClient(
            api_key="test_key", api_secret="test_secret", testnet=True
        )
        self.assertTrue(client.test_connection())

    @patch("src.app.exchange.Client")
    def test_failed_connection(self, mock_client):
        # Mock the binance client to raise an exception
        mock_client.return_value.ping.side_effect = Exception("Connection failed")

        client = BinanceClient(
            api_key="invalid_key", api_secret="invalid_secret", testnet=True
        )
        self.assertFalse(client.test_connection())

    @patch("src.app.exchange.Client")
    def test_get_historical_klines(self, mock_client):
        mock_client.return_value.get_historical_klines.return_value = [
            [1, 2, 3],
            [4, 5, 6],
        ]
        client = BinanceClient(
            api_key="test_key", api_secret="test_secret", testnet=True
        )
        klines = client.get_historical_klines("BTCUSDT", "1m", "1 day ago UTC")
        self.assertEqual(len(klines), 2)

    @patch("src.app.exchange.Client")
    def test_get_latest_price(self, mock_client):
        mock_client.return_value.get_symbol_ticker.return_value = {"price": "50000.00"}
        client = BinanceClient(
            api_key="test_key", api_secret="test_secret", testnet=True
        )
        price = client.get_latest_price("BTCUSDT")
        self.assertEqual(price, "50000.00")

    @patch("src.app.exchange.Client")
    def test_create_order(self, mock_client):
        mock_client.return_value.create_order.return_value = {"orderId": "12345"}
        client = BinanceClient(
            api_key="test_key", api_secret="test_secret", testnet=True
        )
        order = client.create_order(
            symbol="BTCUSDT", side="BUY", type="MARKET", quantity=1
        )
        self.assertEqual(order["orderId"], "12345")

    @patch("src.app.exchange.Client")
    def test_cancel_order(self, mock_client):
        mock_client.return_value.cancel_order.return_value = {"orderId": "12345"}
        client = BinanceClient(
            api_key="test_key", api_secret="test_secret", testnet=True
        )
        order = client.cancel_order(symbol="BTCUSDT", orderId="12345")
        self.assertEqual(order["orderId"], "12345")


if __name__ == "__main__":
    unittest.main()
