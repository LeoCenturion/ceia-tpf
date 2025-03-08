import time
from binance.client import Client

# API credentials
API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'

# Initialize Binance client
client = Client(API_KEY, API_SECRET)

# Trading parameters
symbol = 'BTCUSDT'
buy_price_threshold = 0.98  # Buy if price drops by 2%
sell_price_threshold = 1.02  # Sell if price rises by 2%
investment_amount = 0.001  # Amount of BTC to buy/sell

def get_current_price(symbol):
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker['price'])

def buy_btc(amount):
    print(f"Buying {amount} BTC...")
    # Uncomment the line below to execute the order
    # order = client.order_market_buy(symbol=symbol, quantity=amount)
    # print(order)

def sell_btc(amount):
    print(f"Selling {amount} BTC...")
    # Uncomment the line below to execute the order
    # order = client.order_market_sell(symbol=symbol, quantity=amount)
    # print(order)

def main():
    initial_price = get_current_price(symbol)
    print(f"Initial price: {initial_price}")

    while True:
        current_price = get_current_price(symbol)
        print(f"Current price: {current_price}")

        if current_price <= initial_price * buy_price_threshold:
            buy_btc(investment_amount)
            initial_price = current_price  # Update initial price to the new buy price

        elif current_price >= initial_price * sell_price_threshold:
            sell_btc(investment_amount)
            initial_price = current_price  # Update initial price to the new sell price

        time.sleep(10)  # Wait for 10 seconds before checking again

if __name__ == '__main__':
    main()
