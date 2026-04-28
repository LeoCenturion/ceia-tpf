from binance.client import Client


class BinanceClient:
    def __init__(self, api_key, api_secret, testnet=False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = self._create_client()

    def _create_client(self):
        return Client(self.api_key, self.api_secret, testnet=self.testnet)

    def test_connection(self):
        try:
            self.client.ping()
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def get_historical_klines(self, symbol, interval, start_str=None, limit=500):
        return self.client.get_historical_klines(
            symbol=symbol, interval=interval, start_str=start_str, limit=limit
        )

    def get_latest_price(self, symbol):
        return self.client.get_symbol_ticker(symbol=symbol)["price"]

    def create_order(self, symbol, side, type, quantity):
        return self.client.create_order(
            symbol=symbol, side=side, type=type, quantity=quantity
        )

    def cancel_order(self, symbol, orderId):
        return self.client.cancel_order(symbol=symbol, orderId=orderId)
