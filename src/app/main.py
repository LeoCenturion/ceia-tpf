import click
import os
from src.app.bot import Bot
from src.app.config import load_config
from src.app.exchange import BinanceClient
from src.app.strategy import StrategyFactory
from src.app.logging import setup_logging

@click.group()
def cli():
    """A simple trading bot CLI."""
    pass

@cli.command()
def start():
    """Starts the trading bot."""
    setup_logging()
    config = load_config('config.yaml')
    
    api_key = os.environ.get("BINANCE_TESTNET_API_KEY")
    api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET")

    if not api_key or not api_secret:
        click.echo("Please set BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET environment variables.")
        return

    exchange_client = BinanceClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=True
    )
    
    strategy = StrategyFactory.create_strategy(
        name=config['strategy']['name'],
        **config['strategy']['params']
    )
    
    bot = Bot(config, exchange_client, strategy)
    click.echo('Starting bot...')
    bot.run()

@cli.command()
def stop():
    """Stops the trading bot."""
    click.echo('Stopping bot...')

@cli.command()
def status():
    """Gets the status of the trading bot."""
    click.echo('Bot status: running')

if __name__ == '__main__':
    cli()
