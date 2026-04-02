import click
import os
import sys
import time
import requests
import logging
import subprocess
from flask import Flask, jsonify, request
from threading import Thread

# Imports for the bot thread are moved inside the thread function
# to ensure they are loaded in the correct process context.

APP = Flask(__name__)
BOT_INSTANCE = None

def start_bot_thread():
    """Initializes and runs the bot in a separate thread."""
    global BOT_INSTANCE
    
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
def status_endpoint():
    if BOT_INSTANCE and BOT_INSTANCE.running:
        return jsonify({
            "status": "running",
            "position": BOT_INSTANCE.position,
            "balance": BOT_INSTANCE.account_balance
        })
    return jsonify({"status": "stopped"})

@APP.route('/shutdown', methods=['POST'])
def shutdown_endpoint():
    global BOT_INSTANCE
    if BOT_INSTANCE:
        BOT_INSTANCE.stop()
    
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        logging.error('Not running with the Werkzeug Server, cannot shut down gracefully.')
        return jsonify({"status": "shutdown failed"}), 500
    
    func()
    return jsonify({"status": "shutting down..."})

@click.group()
def cli():
    """A trading bot CLI that communicates with a background server."""
    pass

@cli.command()
def start():
    """Starts the bot server as a background process."""
    try:
        # Check if server is already running
        requests.get("http://127.0.0.1:5000/status", timeout=0.1)
        click.echo("Bot server is already running.")
        return
    except requests.exceptions.ConnectionError:
        # Server is not running, so we can start it.
        pass

    command = [sys.executable, "-c", "from src.app.main import run_server; run_server()"]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    click.echo("Bot server started in the background.")
    time.sleep(2) # Give it a moment to initialize

@cli.command()
def stop():
    """Stops the bot server via HTTP request."""
    try:
        response = requests.post("http://127.0.0.1:5000/shutdown", timeout=5)
        if response.status_code == 200:
            click.echo("Shutdown signal sent successfully.")
        else:
            click.echo(f"Failed to send shutdown signal: {response.status_code} {response.text}", err=True)
    except requests.exceptions.ConnectionError:
        click.echo("Could not connect to bot server. It might already be stopped.", err=True)

@cli.command()
def status():
    """Gets the status of the running bot via HTTP request."""
    try:
        response = requests.get("http://127.0.0.1:5000/status")
        if response.status_code == 200:
            click.echo(f"Bot Status: {response.json()}")
        else:
            click.echo(f"Error getting status: {response.status_code} {response.text}", err=True)
    except requests.exceptions.ConnectionError:
        click.echo("Bot server is stopped (could not connect).", err=True)

def run_server():
    """Runs the Flask server and starts the bot thread."""
    bot_thread = Thread(target=start_bot_thread, daemon=True)
    bot_thread.start()
    APP.run(host='127.0.0.1', port=5000)

if __name__ == '__main__':
    cli()
