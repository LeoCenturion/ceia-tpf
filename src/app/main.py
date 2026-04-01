import click

@click.group()
def cli():
    """A simple trading bot CLI."""
    pass

@cli.command()
def start():
    """Starts the trading bot."""
    click.echo('Starting bot...')

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
