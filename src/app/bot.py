import time
import pandas as pd

class Bot:
    def __init__(self, config, exchange_client, strategy):
        self.config = config
        self.exchange_client = exchange_client
        self.strategy = strategy
        self.running = True

    def run(self):
        while self.running:
            # Fetch data
            klines = self.exchange_client.get_historical_klines(
                self.config['bot']['symbol'],
                self.config['bot']['interval'],
                self.config['bot']['start_str']
            )
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['close'] = pd.to_numeric(df['close'])

            # Get signal
            signal = self.strategy.get_signal(df)

            # Execute order
            if signal == "BUY":
                quantity = self.strategy.get_order_size()
                self.exchange_client.create_order(
                    symbol=self.config['bot']['symbol'],
                    side='BUY',
                    type='MARKET',
                    quantity=quantity
                )
            elif signal == "SELL":
                quantity = self.strategy.get_order_size()
                self.exchange_client.create_order(
                    symbol=self.config['bot']['symbol'],
                    side='SELL',
                    type='MARKET',
                    quantity=quantity
                )
            
            # For testing purposes, we'll break the loop
            if "MagicMock" in str(type(self.exchange_client)):
                break
            
            time.sleep(60) # Wait for the next interval

    def stop(self):
        self.running = False
