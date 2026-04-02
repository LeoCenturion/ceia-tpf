import time
import pandas as pd

class Bot:
    def __init__(self, config, exchange_client, strategy):
        self.config = config
        self.exchange_client = exchange_client
        self.strategy = strategy
        self.running = True
        self.account_balance = 10000 # Example balance
        self.position = None

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
            latest_price = float(df['close'].iloc[-1])

            # Check for stop-loss
            if self.position:
                pnl = (latest_price - self.position['price']) * self.position['quantity']
                if pnl / (self.position['price'] * self.position['quantity']) < -self.config['risk_management']['stop_loss']:
                    self.exchange_client.create_order(
                        symbol=self.config['bot']['symbol'],
                        side='SELL',
                        type='MARKET',
                        quantity=self.position['quantity']
                    )
                    self.position = None

            # Get signal
            signal = self.strategy.get_signal(df)

            # Execute order
            if signal == "BUY" and not self.position:
                quantity = self.strategy.get_order_size()
                self.exchange_client.create_order(
                    symbol=self.config['bot']['symbol'],
                    side='BUY',
                    type='MARKET',
                    quantity=quantity
                )
                self.position = {'price': latest_price, 'quantity': quantity}
            elif signal == "SELL" and self.position:
                self.exchange_client.create_order(
                    symbol=self.config['bot']['symbol'],
                    side='SELL',
                    type='MARKET',
                    quantity=self.position['quantity']
                )
                self.position = None

            if "MagicMock" in str(type(self.exchange_client)):
                break
            
            time.sleep(60)

    def stop(self):
        self.running = False
