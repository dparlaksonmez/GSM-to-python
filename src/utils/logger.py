"""Logger configuration for the GSM pipeline.

This module provides logging functionality with:
- Console output with colors
- File output
- Support for Jupyter notebooks
- Emoji indicators for different log levels
- Cross-platform compatibility (Linux/Windows)

Color Support:
- Linux/Mac: ANSI colors work natively ✅
- Windows 10+: Automatically enables VT100 mode for colors ✅
- Windows (older): Falls back to colorama if installed, otherwise no colors
  To enable colors on older Windows: pip install colorama

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
import platform
from pathlib import Path
from typing import Optional

# Default logging configuration for classification workflow
DEFAULT_LOGGING_LEVEL = 'INFO'
DEFAULT_LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOGGING_OUTPUT_FILE = 'workflow.log'

# Detect operating system for cross-platform compatibility
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'

# Enable ANSI colors on Windows 10+ by enabling VT100 mode
if IS_WINDOWS:
    try:
        # Try to enable VT100 mode on Windows 10+
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        USE_COLORS = True
    except Exception:
        # If that fails, try importing colorama as fallback
        try:
            import colorama  # type: ignore
            colorama.init(autoreset=True)
            USE_COLORS = True
        except ImportError:
            # No color support on Windows without colorama
            # To enable colors on Windows, install: pip install colorama
            USE_COLORS = False
else:
    # Linux/Mac support ANSI colors natively
    USE_COLORS = True

# ANSI color codes
COLORS = {
    'DEBUG': '\033[36m',    # Cyan
    'INFO': '\033[32m',     # Green  
    'WARNING': '\033[33m',  # Yellow
    'ERROR': '\033[31m',    # Red
    'RESET': '\033[0m'      # Reset
}

# Emoji indicators (work on both Windows 10+ and Linux with UTF-8 support)
EMOJIS = {
    'DEBUG': '🔍',
    'INFO': '✨',
    'WARNING': '⚠️',
    'ERROR': '❌'
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter adding colors and emojis to log messages.
    
    Colors are only applied on Linux/Mac or Windows with proper terminal support.
    Emojis work on Windows 10+ and modern Linux terminals.
    """
    
    def format(self, record):
        # Add emoji prefix
        record.emoji = EMOJIS.get(record.levelname, '')
        
        # Add color only if supported
        if USE_COLORS:
            record.color = COLORS.get(record.levelname, COLORS['RESET'])
            record.reset = COLORS['RESET']
        else:
            record.color = ''
            record.reset = ''
        
        return super().format(record)


class LazyErrorFileHandler(logging.Handler):
    """A file handler that only creates the log file when the first error occurs.
    
    This prevents empty error.log files from being created when there are no errors.
    """
    
    def __init__(self, filename: str, encoding: str = 'utf-8'):
        super().__init__()
        self.filename = filename
        self.encoding = encoding
        self._file_handler: Optional[logging.FileHandler] = None
    
    def _ensure_file_handler(self):
        """Create the actual file handler on first use."""
        if self._file_handler is None:
            # Create parent directory if needed
            log_path = Path(self.filename)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            self._file_handler = logging.FileHandler(self.filename, encoding=self.encoding)
            self._file_handler.setFormatter(self.formatter)
            self._file_handler.setLevel(self.level)
    
    def emit(self, record):
        """Emit a record - creates file on first error."""
        self._ensure_file_handler()
        if self._file_handler:
            self._file_handler.emit(record)
    
    def close(self):
        """Close the file handler if it was created."""
        if self._file_handler:
            self._file_handler.close()
        super().close()


def setup_logger(log_file: Optional[str] = None,
                level: int = logging.INFO,
                logger_name: str = 'classification_pipeline') -> logging.Logger:
    """Sets up the logger with colored console and file output.
    
    Cross-platform compatible logger that works on both Linux and Windows.
    Uses pathlib.Path for proper path handling across operating systems.
    
    Args:
        log_file: Full path or relative path for log file. Directories will be created if needed.
                 Can be a string or Path object. Uses pathlib for cross-platform compatibility.
        level: Logging level (default: logging.INFO)
        logger_name: Name for the logger (default: 'classification_pipeline')
    
    Returns:
        Configured logger instance
        
    Example:
        >>> # Works on both Linux and Windows
        >>> logger = setup_logger("output/workflow.log")
        >>> logger = setup_logger(Path("output") / "workflow.log")
    """
    
    # Use default log file if none provided
    if log_file is None:
        log_file = DEFAULT_LOGGING_OUTPUT_FILE
    
    # Convert to Path object for cross-platform compatibility
    log_file_path = Path(log_file).resolve()
    
    logger = logging.getLogger(logger_name)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(level)

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    colored_formatter = ColoredFormatter(
        '%(color)s%(emoji)s %(asctime)s - %(levelname)s - %(message)s%(reset)s'
    )
    console_handler.setFormatter(colored_formatter)
    console_handler.flush = sys.stdout.flush  # Force flush

    # File handler (no colors)
    if log_file:
        # Create directory if it doesn't exist (cross-platform)
        log_dir = log_file_path.parent
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
            
        # Use the resolved path string for FileHandler
        file_handler = logging.FileHandler(str(log_file_path), encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Lazy error file handler - only creates file when first error occurs
    if log_file:
        error_log_path = log_file_path.parent / 'error.log'
    else:
        error_log_path = Path('error.log').resolve()
        
    error_handler = LazyErrorFileHandler(str(error_log_path), encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    error_handler.setFormatter(error_formatter)
    logger.addHandler(error_handler)

    logger.addHandler(console_handler)
    
    # Log system information for debugging
    logger.debug(f"🖥️ Operating System: {platform.system()} {platform.release()}")
    logger.debug(f"📝 Log file: {log_file_path}")
    
    return logger

def get_logger(logger_name: str = 'workflow') -> logging.Logger:
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
