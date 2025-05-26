"""Centralized logging configuration for the stock agent.

This module configures logging for the entire application to ensure consistent
formatting and avoid duplicate log messages.
"""

import logging
import sys

def configure_logging():
    """Configure global logging settings.
    
    This function should be called once at application startup.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Clear any existing handlers to prevent duplicates
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Set default level
    root_logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(console_handler)
    
    # Configure library-specific loggers to reduce verbosity
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    
    logging.info("Logging configured.")

# Auto-configure logging when this module is imported
configure_logging() 