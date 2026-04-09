"""
Logging configuration utilities for the Senate LLM Pipeline

Provides centralized logging setup with file and console handlers,
hierarchical logger naming, and suppression of noisy third-party loggers.
"""

import logging
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_dir: Path,
    log_level: int = logging.DEBUG,
    console_level: int = logging.INFO
) -> None:
    """
    Configure centralized logging for the pipeline.
    
    Args:
        log_dir: Directory where log files will be saved
        log_level: Logging level for file output (default: DEBUG=10)
        console_level: Logging level for console output (default: INFO=20)
    """
    # Create log directory with parents
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # File handler — verbose, includes timestamp and logger name
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler — less verbose, simple format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("groq").setLevel(logging.WARNING)
    
    # Log initialization message
    root_logger.info(f"Logging initialized. Log file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a hierarchical logger instance for a module.
    
    Creates loggers under the "senate_pipeline" namespace for organized,
    hierarchical logging output.
    
    Args:
        name: Module name (e.g., "api", "ablation", "notebook", "evaluator")
    
    Returns:
        logging.Logger instance with hierarchical name "senate_pipeline.{name}"
    """
    return logging.getLogger(f"senate_pipeline.{name}")
