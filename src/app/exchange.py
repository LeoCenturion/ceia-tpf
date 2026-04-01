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
