import click
import os
import sys
import time
import requests
import logging
import subprocess
from flask import Flask, jsonify, request
from threading import Thread

# It's better to import these inside the functions that need them
# to avoid circular dependencies and issues with forking.

PID_FILE = "bot.pid"
APP = Flask(__name__)
BOT_INSTANCE = None

def start_bot_thread():
    """Initializes and runs the bot in a separate thread."""
    global BOT_INSTANCE
    
    # These imports are inside the thread to ensure they are loaded in the new process context
    from src.app.bot import Bot
    from src.app.config import load_config
    from src.app.exchange import BinanceClient
    from src.app.strategy import StrategyFactory
    from src.app.logging import setup_logging
    
    setup_logging()
    logging.info("Bot thread starting.")
    
    try:
        config = load_config('config.yaml')
        api_key = os.environ.get("BINANCE_TESTNET_API_KEY")
        api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET")

        if not api_key or not api_secret:
            logging.error("API keys not set. Bot thread cannot start.")
            return

        exchange_client = BinanceClient(api_key, api_secret, testnet=True)
        strategy = StrategyFactory.create_strategy(name=config['strategy']['name'], **config['strategy']['params'])
        
        BOT_INSTANCE = Bot(config, exchange_client, strategy)
        BOT_INSTANCE.run()
        logging.info("Bot thread finished.")
    except Exception as e:
        logging.error(f"Critical error in bot thread: {e}", exc_info=True)


@APP.route('/status', methods=['GET'])
def status():
    if BOT_INSTANCE and BOT_INSTANCE.running:
        return jsonify({
            "status": "running",
            "position": BOT_INSTANCE.position,
            "balance": BOT_INSTANCE.account_balance
        })
    return jsonify({"status": "stopped"})

@APP.route('/shutdown', methods=['POST'])
def shutdown():
    global BOT_INSTANCE
    if BOT_INSTANCE:
        BOT_INSTANCE.stop()
    
    # Use Werkzeug's shutdown function for a graceful server stop
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        logging.error('Not running with the Werkzeug Server, cannot shut down gracefully.')
        return jsonify({"status": "shutdown failed: not on Werkzeug server"}), 500
    
    func()
    return jsonify({"status": "shutting down..."})

@click.group()
def cli():
    """A trading bot CLI with background process management."""
    pass

@cli.command()
def start():
    """Starts the bot server as a background process."""
    if os.path.exists(PID_FILE):
        click.echo("Bot server may already be running (PID file exists).")
        return

    command = [sys.executable, "-c", "from src.app.main import run_server; run_server()"]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    with open(PID_FILE, "w") as f:
        f.write(str(process.pid))
    click.echo(f"Bot server started in background with PID: {process.pid}")

@cli.command()
def stop():
    """Stops the bot server."""
    if not os.path.exists(PID_FILE):
        click.echo("Bot server is not running (PID file not found).")
        return
    
    try:
        response = requests.post("http://127.0.0.1:5000/shutdown")
        if response.status_code == 200:
            click.echo("Shutdown signal sent successfully.")
        else:
            click.echo(f"Failed to send shutdown signal: {response.status_code} {response.text}", err=True)
    except requests.exceptions.ConnectionError:
        click.echo("Could not connect to bot server. It might already be stopped.", err=True)
    finally:
        # Clean up the PID file regardless
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        click.echo("Cleaned up PID file.")


@cli.command()
def status():
    """Gets the status of the running bot."""
    if not os.path.exists(PID_FILE):
        click.echo("Bot server is stopped (PID file not found).")
        return
    
    try:
        response = requests.get("http://127.0.0.1:5000/status")
        if response.status_code == 200:
            click.echo(f"Bot Status: {response.json()}")
        else:
            click.echo(f"Error getting status: {response.status_code} {response.text}", err=True)
    except requests.exceptions.ConnectionError:
        click.echo("Could not connect to bot server. It may be stopped or starting up.", err=True)

def run_server():
    """Runs the Flask server and starts the bot thread."""
    bot_thread = Thread(target=start_bot_thread, daemon=True)
    bot_thread.start()
    APP.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    cli()
