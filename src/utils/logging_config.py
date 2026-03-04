import logging
import logging.config
import os
from functools import wraps

def setup_logging(func):
    """
    Decorator to configure logging from a config file.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        config_file = 'logging.ini'
        if os.path.exists(config_file):
            logging.config.fileConfig(config_file, disable_existing_loggers=False)
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            logging.warning(f"'{config_file}' not found. Using basic logging configuration.")
        return func(*args, **kwargs)
    return wrapper
