import logging
import os
import subprocess
import sys
from threading import Thread

import click
import requests
from flask import Flask, jsonify, request

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
    from src.app.logging import setup_logging
    from src.app.mock_exchange import MockBinanceClient
    from src.app.strategy_factory import StrategyFactory

    setup_logging()
    logging.info("Bot thread starting.")

    try:
        config = load_config("config.yaml")

        if "mock_file" in config["exchange"]:
            logging.info(
                f"Using mock exchange with file: {config['exchange']['mock_file']}"
            )
            exchange_client = MockBinanceClient(
                mock_file=config["exchange"]["mock_file"], api_key=None, api_secret=None
            )
        else:
            api_key = config["exchange"].get("api_key") or os.environ.get(
                "BINANCE_TESTNET_API_KEY"
            )
            api_secret = config["exchange"].get("api_secret") or os.environ.get(
                "BINANCE_TESTNET_API_SECRET"
            )
            testnet = config["exchange"].get("testnet", True)

            if not api_key or not api_secret:
                logging.error("API keys not set. Bot thread cannot start.")
                return
            logging.info(f"Using Binance {'testnet' if testnet else 'live'} exchange.")
            exchange_client = BinanceClient(api_key, api_secret, testnet=testnet)

        strategy_config = config["bot"][
            "strategy"
        ].copy()  # Use .copy() to avoid modifying the original config dict
        strategy_name = strategy_config.pop("name")
        strategy = StrategyFactory.create_strategy(
            name=strategy_name, **strategy_config
        )

        BOT_INSTANCE = Bot(config, exchange_client, strategy)
        BOT_INSTANCE.run()
        logging.info("Bot thread finished.")
    except Exception as e:
        logging.error(f"Critical error in bot thread: {e}", exc_info=True)


@APP.route("/status", methods=["GET"])
def status_endpoint():
    if BOT_INSTANCE and BOT_INSTANCE.running:
        return jsonify(
            {
                "status": "running",
                "position": BOT_INSTANCE.position,
                "balance": BOT_INSTANCE.account_balance,
            }
        )
    return jsonify({"status": "stopped"})


@APP.route("/shutdown", methods=["POST"])
def shutdown_endpoint():
    global BOT_INSTANCE
    if BOT_INSTANCE:
        BOT_INSTANCE.stop()

    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        logging.error(
            "Not running with the Werkzeug Server, cannot shut down gracefully."
        )
        return jsonify({"status": "shutdown failed"}), 500

    func()
    return jsonify({"status": "shutting down..."})


PID_FILE = "bot.pid"


@click.group()
def cli():
    """A trading bot CLI that communicates with a background server."""
    pass


@cli.command()
@click.option("--port", default=5000, help="Port to run the server on.")
def start(port):
    """Starts the bot server as a background process."""
    if os.path.exists(PID_FILE):
        click.echo("Bot server is already running (PID file exists).")
        return

    command = [
        sys.executable,
        "-c",
        f"from src.app.main import run_server; run_server(port={port})",
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    with open(PID_FILE, "w") as f:
        f.write(str(process.pid))

    click.echo(
        f"Bot server started in the background on port {port} with PID {process.pid}."
    )


@cli.command()
def stop():
    """Stops the bot server process."""
    if not os.path.exists(PID_FILE):
        click.echo("Bot server is not running (no PID file).")
        return

    with open(PID_FILE, "r") as f:
        pid = int(f.read())

    try:
        os.kill(pid, 15)  # Send SIGTERM
        click.echo(f"Sent shutdown signal to process {pid}.")
    except ProcessLookupError:
        click.echo(
            f"Process {pid} not found. It might have already been stopped.", err=True
        )
    finally:
        os.remove(PID_FILE)


@cli.command()
@click.option("--port", default=5000, help="Port the server is running on.")
def status(port):
    """Gets the status of the running bot via HTTP request."""
    if not os.path.exists(PID_FILE):
        click.echo("Bot server is stopped (no PID file).", err=True)
        return

    try:
        response = requests.get(f"http://127.0.0.1:{port}/status")
        if response.status_code == 200:
            click.echo(f"Bot Status: {response.json()}")
        else:
            click.echo(
                f"Error getting status: {response.status_code} {response.text}",
                err=True,
            )
    except requests.exceptions.ConnectionError:
        click.echo("Bot server is stopped (could not connect).", err=True)


def run_server(port=5000):
    """Runs the Flask server and starts the bot thread."""
    bot_thread = Thread(target=start_bot_thread, daemon=True)
    bot_thread.start()
    APP.run(host="127.0.0.1", port=port)


if __name__ == "__main__":
    cli()
