import click
import os
import sys
import time
import requests
import logging
import subprocess
from flask import Flask, jsonify, request
from threading import Thread

from src.app.bot import Bot
from src.app.config import load_config
from src.app.exchange import BinanceClient
from src.app.strategy import StrategyFactory
from src.app.logging import setup_logging

PID_FILE = "bot.pid"
BOT_THREAD = None
APP = Flask(__name__)

def start_bot_thread():
    """Initializes and runs the bot in a separate thread."""
    global BOT_THREAD
    setup_logging()
    logging.info("Bot thread started.")
    
    config = load_config('config.yaml')
    api_key = os.environ.get("BINANCE_TESTNET_API_KEY")
    api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET")

    if not api_key or not api_secret:
        logging.error("API keys not set.")
        return

    exchange_client = BinanceClient(api_key, api_secret, testnet=True)
    strategy = StrategyFactory.create_strategy(name=config['strategy']['name'], **config['strategy']['params'])
    
    bot = Bot(config, exchange_client, strategy)
    BOT_THREAD = bot  # Make bot instance accessible
    bot.run()
    logging.info("Bot thread finished.")

@APP.route('/status', methods=['GET'])
def status():
    if BOT_THREAD and BOT_THREAD.running:
        return jsonify({
            "status": "running", 
            "position": BOT_THREAD.position,
            "balance": BOT_THREAD.account_balance
        })
    return jsonify({"status": "stopped"})

@APP.route('/shutdown', methods=['POST'])
def shutdown():
    global BOT_THREAD
    if BOT_THREAD:
        BOT_THREAD.stop()
    
    # The shutdown function below is a bit abrupt, good for development
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
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
        click.echo("Bot is already running.")
        return

    # Use Popen to run the Flask app in a new process
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
            click.echo("Shutdown signal sent to bot server.")
        else:
            click.echo(f"Failed to send shutdown signal: {response.text}", err=True)
    except requests.exceptions.ConnectionError:
        click.echo("Could not connect to bot server. It might already be stopped.", err=True)
    finally:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)

@cli.command()
def status():
    """Gets the status of the running bot."""
    if not os.path.exists(PID_FILE):
        click.echo("Bot server is stopped.")
        return
    
    try:
        response = requests.get("http://127.0.0.1:5000/status")
        if response.status_code == 200:
            click.echo(f"Bot Status: {response.json()}")
        else:
            click.echo(f"Error getting status: {response.text}", err=True)
    except requests.exceptions.ConnectionError:
        click.echo("Could not connect to bot server. It may be stopped or starting up.", err=True)

def run_server():
    """Runs the Flask server and starts the bot thread."""
    bot_thread = Thread(target=start_bot_thread)
    bot_thread.start()
    APP.run(port=5000)

if __name__ == '__main__':
    cli()
