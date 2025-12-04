import logging
import sys


class CustomFormatter(logging.Formatter):
    """Custom logging formatter to add colors based on log level."""

    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def get_logger(name: str) -> logging.Logger:
    """Creates and returns a logger with the specified name and custom
    formatting."""
    logger = logging.getLogger(name)

    # Only add handler if not already added to avoid duplicate logs
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # Create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)

        # Add formatter to handler
        ch.setFormatter(CustomFormatter())

        # Add handler to logger
        logger.addHandler(ch)

    return logger
