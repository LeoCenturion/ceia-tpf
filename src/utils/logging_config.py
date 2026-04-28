import logging
import logging.config
import os
from functools import wraps
from typing import Callable, Any


def setup_logging(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to configure logging from a config file.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        config_file = "logging.ini"
        if os.path.exists(config_file):
            logging.config.fileConfig(config_file, disable_existing_loggers=False)
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            logging.warning(
                f"'{config_file}' not found. Using basic logging configuration."
            )
        return func(*args, **kwargs)

    return wrapper
