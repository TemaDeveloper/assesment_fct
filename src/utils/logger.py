import logging
import sys
from datetime import datetime


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
