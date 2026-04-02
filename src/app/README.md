# Trading Bot

This is a simple trading bot that uses a MACD strategy to trade on the Binance Spot Testnet.

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
