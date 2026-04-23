from src.data_acquisition.binance_client import BinanceClient
from src.data_storage.data_storage import DataStorage

# 1. Initialize the Binance client
client = BinanceClient(api_key='', api_secret='')

# 2. Fetch some data
data = client.fetch_historical_data('BTCUSDT', '1m', '1 day ago UTC')

# 3. Store the data
storage = DataStorage('btc_data.csv')
storage.save_data(data)

print("Data saved to btc_data.csv")
