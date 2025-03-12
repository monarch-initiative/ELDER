"""
Logging module for Elder.

This module provides a logging system for Elder that can be used across
the codebase for consistent logging.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Set up logging for Elder.
    
    Args:
        level: Logging level
        log_file: Path to log file (optional)
        log_format: Logging format
        date_format: Date format for logging
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger("elder")
    logger.setLevel(level)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter(log_format, date_format)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file).parent
        os.makedirs(log_path, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


logger = setup_logging()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger.
    
    Args:
        name: Logger name (optional)
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"elder.{name}")
    return logger