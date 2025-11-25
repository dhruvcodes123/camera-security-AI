"""
Logging setup for the Hybrid Person Tracking System.
"""

import logging
import os
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_to_file: bool = True) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Basic configuration for logging to console
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Add file handler if enabled
    if log_to_file:
        # Ensure 'logs' directory exists
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"hybrid_tracker_{timestamp}.log")
        
        # Also keep a consistent main log file
        main_log_file = "app.log"
        
        # File handler for main log file
        file_handler = logging.FileHandler(main_log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            "%Y-%m-%d %H:%M:%S"
        ))
        
        # Add handler to root logger
        logging.getLogger("").addHandler(file_handler)
        
        # Also create a timestamped log file
        if log_dir != "":
            ts_file_handler = logging.FileHandler(log_file)
            ts_file_handler.setLevel(numeric_level)
            ts_file_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                "%Y-%m-%d %H:%M:%S"
            ))
            
            # Add handler to root logger
            logging.getLogger("").addHandler(ts_file_handler)
        
        logging.info(f"Logging to file: {os.path.abspath(main_log_file)}")
        if log_dir != "":
            logging.info(f"Also logging to timestamped file: {os.path.abspath(log_file)}")
    
    logging.info(f"Logging initialized at {log_level} level") 