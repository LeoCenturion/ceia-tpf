# Trading Bot

This is a flexible trading bot that allows the use of multiple trading strategies on the Binance Spot Testnet.

## Prerequisites

- Python 3.12+
- Poetry

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <project_directory>
   ```
3. Install the dependencies:
   ```bash
   poetry install
   ```

## Usage

### Configuration

1. Create a `config.yaml` file in the root directory of the project with the following content:
   ```yaml
   bot:
     symbol: 'BTCUSDT'
     interval: '1m'
     start_str: '1 day ago UTC'
   exchange:
     name: 'binance_testnet'
   strategy:
     name: 'macd' # Name of the strategy to use
     params:
       fast_period: 12
       slow_period: 26
       signal_period: 9
   risk_management:
     stop_loss: 0.1
     max_drawdown: 0.2
   capital_allocation:
     max_capital: 1000
   ```
2. Set up your Binance Testnet API keys as environment variables:
   ```bash
   export BINANCE_TESTNET_API_KEY="your_api_key"
   export BINANCE_TESTNET_API_SECRET="your_api_secret"
   ```

## Configuration Schema

- `bot`:
  - `symbol` (string): The trading symbol (e.g., 'BTCUSDT').
  - `interval` (string): The candlestick interval (e.g., '1m', '5m', '1h').
  - `start_str` (string): The start time for historical data (e.g., '1 day ago UTC').
- `exchange`:
  - `name` (string): The name of the exchange (e.g., 'binance_testnet').
- `strategy`:
  - `name` (string): The name of the strategy to use (e.g., 'macd').
  - `params` (dict): A dictionary of parameters for the chosen strategy.
- `risk_management`:
  - `stop_loss` (float): The stop-loss percentage (e.g., 0.1 for 10%).
  - `max_drawdown` (float): The maximum drawdown percentage (e.g., 0.2 for 20%).
- `capital_allocation`:
  - `max_capital` (float): The maximum capital to allocate for a single trade.

### Commands

- **Start the bot:**
  ```bash
  poetry run python -m src.app.main start
  ```
- **Stop the bot:**
  ```bash
  poetry run python -m src.app.main stop
  ```
- **Get the bot's status:**
  ```bash
  poetry run python -m src.app.main status
  ```

## Backtesting

To run a backtest of a strategy, you can run the `src/app/backtesting.py` script:
```bash
poetry run python -m src.app.backtesting
```
This will run a backtest of the `SmaCross` strategy with sample data and print the results to the console.
