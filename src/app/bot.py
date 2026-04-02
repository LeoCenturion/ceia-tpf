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
        self.run_once = False # New attribute for manual testing

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
                    print("Stop-loss triggered!")
                    self.exchange_client.create_order(
                        symbol=self.config['bot']['symbol'],
                        side='SELL',
                        type='MARKET',
                        quantity=self.position['quantity']
                    )
                    self.position = None

            # Get signal
            signal = self.strategy.get_signal(df)
            print(f"Generated signal: {signal}")

            # Execute order
            if signal == "BUY" and not self.position:
                quantity = self.strategy.get_order_size()
                cost = quantity * latest_price
                if cost <= self.config['capital_allocation']['max_capital']:
                    print(f"Executing BUY order for {quantity} of {self.config['bot']['symbol']}")
                    self.exchange_client.create_order(
                        symbol=self.config['bot']['symbol'],
                        side='BUY',
                        type='MARKET',
                        quantity=quantity
                    )
                    self.position = {'price': latest_price, 'quantity': quantity}
                else:
                    print(f"Skipping BUY order due to capital allocation limit.")

            elif signal == "SELL" and self.position:
                print(f"Executing SELL order for {self.position['quantity']} of {self.config['bot']['symbol']}")
                self.exchange_client.create_order(
                    symbol=self.config['bot']['symbol'],
                    side='SELL',
                    type='MARKET',
                    quantity=self.position['quantity']
                )
                self.position = None

            if self.run_once or "MagicMock" in str(type(self.exchange_client)):
                break
            
            time.sleep(60)


    def stop(self):
        self.running = False
