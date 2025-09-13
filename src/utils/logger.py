"""Logger configuration for the GSM pipeline.

This module provides logging functionality with:
- Console output with colors
- File output
- Support for Jupyter notebooks
- Emoji indicators for different log levels

Key Functions:
- setup_logger: Configure logging with handlers
- log_info: Log info messages
- log_warning: Log warning messages 
- log_error: Log error messages
- log_debug: Log debug messages
"""

import logging
import sys
import os
from typing import Optional

# Default logging configuration for classification workflow
DEFAULT_LOGGING_LEVEL = 'INFO'
DEFAULT_LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOGGING_OUTPUT_FILE = 'classification_workflow.log'

# ANSI color codes
COLORS = {
    'DEBUG': '\033[36m',    # Cyan
    'INFO': '\033[32m',     # Green  
    'WARNING': '\033[33m',  # Yellow
    'ERROR': '\033[31m',    # Red
    'RESET': '\033[0m'      # Reset
}

# Emoji indicators
EMOJIS = {
    'DEBUG': '🔍',
    'INFO': '✨',
    'WARNING': '⚠️',
    'ERROR': '❌'
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter adding colors and emojis to log messages"""
    
    def format(self, record):
        # Add emoji prefix
        record.emoji = EMOJIS.get(record.levelname, '')
        # Add color
        record.color = COLORS.get(record.levelname, COLORS['RESET'])
        record.reset = COLORS['RESET']
        
        return super().format(record)

def setup_logger(log_file: Optional[str] = None, 
                level: int = logging.INFO,
                logger_name: str = 'classification_pipeline') -> logging.Logger:
    """Sets up the logger with colored console and file output.
    
    Args:
        log_file: Full path or relative path for log file. Directories will be created if needed.
        level: Logging level (default: logging.INFO)
        logger_name: Name for the logger (default: 'classification_pipeline')
    
    Returns:
        Configured logger instance
    """
    
    # Use default log file if none provided
    if log_file is None:
        log_file = DEFAULT_LOGGING_OUTPUT_FILE
    
    logger = logging.getLogger(logger_name)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(level)

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    colored_formatter = ColoredFormatter(
        '%(color)s%(emoji)s %(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s'
    )
    console_handler.setFormatter(colored_formatter)
    console_handler.flush = sys.stdout.flush  # Force flush

    # File handler (no colors)
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.addHandler(console_handler)
    return logger

def get_logger(logger_name: str = 'classification_pipeline') -> logging.Logger:
    """Returns the logger instance."""
    return logging.getLogger(logger_name)

def log_debug(message: str):
    """Logs a debug message."""
    get_logger().debug(message)

def log_info(message: str):
    """Logs an informational message."""
    get_logger().info(message)

def log_warning(message: str):
    """Logs a warning message."""
    get_logger().warning(message)

def log_error(message: str):
    """Logs an error message."""
    get_logger().error(message)
