"""Logging utility functions."""
import logging
import os
import sys
from typing import Optional
import datetime


def get_logger(name: str) -> logging.Logger:
    """
    Get logger with default configuration.

    Parameters
    ----------
    name : str
        Logger name

    Returns
    -------
    logging.Logger
        Configured logger
    """
    return logging.getLogger(name)


def setup_logging(
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
        log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> None:
    """
    Set up logging configuration with auto-timestamped log file.

    Parameters
    ----------
    log_file : Optional[str], optional
        Path to log file, by default None (log to console only)
    log_level : int, optional
        Logging level, by default logging.INFO
    log_format : str, optional
        Log format string, by default '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    """
    # Add timestamp to log_file if provided
    if log_file:
        log_dir = os.path.dirname(log_file)
        log_name = os.path.basename(log_file)

        # Split log name and extension
        log_name_parts = os.path.splitext(log_name)

        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"{log_name_parts[0]}_{timestamp}{log_name_parts[1]}"

        # Create final log file path
        log_file = os.path.join(log_dir, log_name)

        # Create directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

    # Configure logging
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)

    # File handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format=log_format
    )

    # Suppress excessive logging from libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('wandb').setLevel(logging.WARNING)

    # Log configuration setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level {logging.getLevelName(log_level)}")
    if log_file:
        logger.info(f"Logging to file: {log_file}")

