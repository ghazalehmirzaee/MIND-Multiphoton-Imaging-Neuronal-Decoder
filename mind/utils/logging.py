"""Logging utility functions."""
import logging
import os
import sys
from typing import Optional


def setup_logging(
        log_file: Optional[str] = None,
        log_level: int = logging.INFO
) -> None:
    """
    Set up logging configuration.

    Parameters
    ----------
    log_file : Optional[str], optional
        Path to log file, by default None (log to console only)
    log_level : int, optional
        Logging level, by default logging.INFO
    """
    # Create logs directory if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configure logging
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    handlers.append(console_handler)

    # File handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers
    )

    # Suppress excessive logging from libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('wandb').setLevel(logging.WARNING)

