"""
Centralized logging service.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class LoggingService:
    """Service for configuring logging."""
    
    @staticmethod
    def setup_logging(
        log_level: int = logging.INFO,
        log_file: Optional[Path] = None,
        format_string: Optional[str] = None
    ) -> None:
        """
        Setup logging configuration.
        
        Args:
            log_level: Logging level (default: INFO)
            log_file: Optional log file path
            format_string: Optional custom format string
        """
        if format_string is None:
            format_string = '[%(asctime)s - %(levelname)s - %(name)s] %(message)s'
        
        formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        root_logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        logging.info(f"Logging configured (level: {logging.getLevelName(log_level)})")

