import time
import pandas as pd
import logging

class Bot:
    def __init__(self, config, exchange_client, strategy):
        self.config = config
        self.exchange_client = exchange_client
        self.strategy = strategy
        self.running = True
        self.account_balance = 10000  # Example starting balance
        self.position = None

    def run(self):
        logging.info("Bot is starting its trading loop.")
        while self.running:
            try:
                klines = self.exchange_client.get_historical_klines(
                    self.config['bot']['symbol'],
                    self.config['bot']['interval'],
                    self.config['bot']['start_str']
                )
                df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
                df['close'] = pd.to_numeric(df['close'])
                latest_price = float(df['close'].iloc[-1])

                if self.position:
                    pnl_percentage = (latest_price - self.position['price']) / self.position['price']
                    if pnl_percentage < -self.config['risk_management']['stop_loss']:
                        logging.warning(f"Stop-loss triggered at {pnl_percentage:.2%}. Selling position.")
                        self.execute_sell()
                        continue

                signal = self.strategy.get_signal(df)
                logging.info(f"Generated signal: {signal} at price {latest_price}")

                if signal == "BUY" and not self.position:
                    self.execute_buy(latest_price)
                elif signal == "SELL" and self.position:
                    self.execute_sell()

            except Exception as e:
                logging.error(f"An error occurred in the trading loop: {e}", exc_info=True)
            
            time.sleep(60)
        
        logging.info("Bot trading loop has gracefully stopped.")

    def execute_buy(self, latest_price):
        quantity = self.strategy.get_order_size()
        cost = quantity * latest_price
        if cost <= self.config['capital_allocation']['max_capital'] and cost <= self.account_balance:
            logging.info(f"Executing BUY order for {quantity} {self.config['bot']['symbol']}.")
            # self.exchange_client.create_order(...) # Uncomment for live trading
            self.position = {'price': latest_price, 'quantity': quantity}
            self.account_balance -= cost
        else:
            logging.info("Skipping BUY order due to capital or balance constraints.")

    def execute_sell(self):
        logging.info(f"Executing SELL order for {self.position['quantity']} {self.config['bot']['symbol']}.")
        # self.exchange_client.create_order(...) # Uncomment for live trading
        self.account_balance += self.position['quantity'] * self.position['price'] # simplified PnL
        self.position = None

    def stop(self):
        logging.info("Stop signal received. Halting trading loop.")
        self.running = False
