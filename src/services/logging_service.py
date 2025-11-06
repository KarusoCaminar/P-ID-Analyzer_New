"""
Centralized logging service.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


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
    
    @staticmethod
    def setup_llm_logging(
        log_dir: Path = Path("outputs/logs"),
        log_level: int = logging.DEBUG
    ) -> logging.Logger:
        """
        Setup dedicated logging for LLM calls (requests/responses).
        
        Creates a separate log file for all LLM interactions with structured logging.
        
        Args:
            log_dir: Directory for log files
            log_level: Logging level (default: DEBUG for full visibility)
            
        Returns:
            Configured LLM logger
        """
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dedicated LLM logger
        llm_logger = logging.getLogger('llm_calls')
        llm_logger.setLevel(log_level)
        llm_logger.propagate = False  # Prevent log duplication on console (LLM logs should not propagate to root logger)
        
        # Remove existing handlers to avoid duplicates
        for handler in llm_logger.handlers[:]:
            llm_logger.removeHandler(handler)
        
        # Create separate file handler for LLM calls
        llm_log_file = log_dir / f"llm_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(llm_log_file, encoding='utf-8')
        
        # Structured format for easy parsing (request_id is optional)
        formatter = logging.Formatter(
            '[%(asctime)s - %(levelname)s - LLM] [%(request_id)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        llm_logger.addHandler(file_handler)
        
        # Also add console handler for visibility
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        llm_logger.addHandler(console_handler)
        
        logging.info(f"LLM logging configured: {llm_log_file}")
        return llm_logger

