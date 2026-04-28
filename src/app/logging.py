import logging.config
import os


def setup_logging(
    default_path="logging.ini", default_level=logging.INFO, env_key="LOG_CFG"
):
    """
    Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        logging.config.fileConfig(path, disable_existing_loggers=False)
    else:
        logging.basicConfig(level=default_level)
