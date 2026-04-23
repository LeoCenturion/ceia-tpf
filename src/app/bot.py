import logging
import time
from typing import Optional, Any, Literal, Union

import pandas as pd

from src.app.strategy import Strategy


class Bot:
    def __init__(self, config: dict[str, Any], exchange_client: Any, strategy: Strategy):
        self.config: dict[str, Any] = config
        self.exchange_client: Any = exchange_client
        self.strategy: Strategy = strategy
        
        self.symbol: str = config['bot']['symbol']
        self.timeframe: str = config['bot']['timeframe']
        self.start_str: Optional[str] = config['bot'].get('start_str', None)
        self.tick_interval: int = config['bot'].get('tick_interval', 60)
        self.capital_allocation: float = config['bot']['capital_allocation']['max_capital']
        self.stop_loss_percentage: float = config['bot']['risk_management']['stop_loss']

        self.running: bool = True
        self.account_balance: float = config['bot']['capital_allocation']['initial_balance'] # Use initial_balance from config
        self.position: Optional[dict[str, Any]] = None

    def run(self) -> None:
        logging.info("Bot is starting its trading loop.")
        while self.running:
            try:
                klines: list[list[Union[str, float, int]]] = self.exchange_client.get_historical_klines(
                    self.symbol,
                    self.timeframe,
                    limit=self.config['bot'].get('klines_limit', 500), # Default limit to 500
                    start_str=self.start_str
                )
                df_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
                df: pd.DataFrame = pd.DataFrame(data=klines)
                df.columns = df_columns # Assign columns after creation
                for col in ('open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'):
                    df[col] = pd.to_numeric(df[col])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp') # Set timestamp as index

                latest_price: float = float(df['close'].iloc[-1])
                logging.debug(
                    f"Fetched {len(df)} klines | latest timestamp: {df.index[-1]} | "
                    f"open: {float(df['open'].iloc[-1])} high: {float(df['high'].iloc[-1])} "
                    f"low: {float(df['low'].iloc[-1])} close: {latest_price}"
                )

                if self.position is not None:
                    pnl_percentage: float = (latest_price - self.position['price']) / self.position['price']
                    if pnl_percentage < -self.stop_loss_percentage:
                        logging.warning(f"Stop-loss triggered at {pnl_percentage:.2%}. Selling position.")
                        self.execute_sell()
                        continue

                signal: Literal["BUY", "SELL", "HOLD"] = self.strategy.get_signal(df)
                logging.info(f"Generated signal: {signal} at price {latest_price}")

                if signal == "BUY" and not self.position:
                    self.execute_buy(latest_price)
                elif signal == "SELL" and self.position:
                    self.execute_sell()

            except Exception as e:
                logging.error(f"An error occurred in the trading loop: {e}", exc_info=True)
            
            time.sleep(self.tick_interval)
        
        logging.info("Bot trading loop has gracefully stopped.")

    def execute_buy(self, latest_price: float) -> None:
        quantity: float = self.strategy.get_order_size()
        cost: float = quantity * latest_price
        if cost <= self.capital_allocation and cost <= self.account_balance:
            logging.info(f"Executing BUY order for {quantity} {self.symbol}.")
            # self.exchange_client.create_order(...) # Uncomment for live trading
            self.position = {'price': latest_price, 'quantity': quantity}
            self.account_balance -= cost
        else:
            logging.info("Skipping BUY order due to capital or balance constraints.")

    def execute_sell(self) -> None:
        if self.position is not None:
            logging.info(f"Executing SELL order for {self.position['quantity']} {self.config['bot']['symbol']}.")
            # self.exchange_client.create_order(...) # Uncomment for live trading
            self.account_balance += self.position['quantity'] * self.position['price'] # simplified PnL
            self.position = None
        else:
            logging.info("No position to sell.")

    def stop(self) -> None:
        logging.info("Stop signal received. Halting trading loop.")
        self.running = False
