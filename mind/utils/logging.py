"""
Logging configuration for the project.
"""
import os
import logging
from pathlib import Path
import sys
from typing import Optional, Union


def setup_logging(log_level: str = 'INFO',
                  log_file: Optional[Union[str, Path]] = None,
                  console: bool = True) -> logging.Logger:
    """
    Set up logging configuration.

    Parameters
    ----------
    log_level : str, optional
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'), by default 'INFO'
    log_file : Optional[Union[str, Path]], optional
        Path to log file, by default None (no file logging)
    console : bool, optional
        Whether to log to console, by default True

    Returns
    -------
    logging.Logger
        Logger object
    """
    # Convert log level string to logging level
    level = getattr(logging, log_level.upper())

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log file is provided
    if log_file is not None:
        # Create directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


