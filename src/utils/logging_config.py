"""
Centralized Logging Configuration using Loguru

This module provides a unified logging setup for all pipeline components,
eliminating the need for manual logging configuration in each file.

Features:
- Automatic file rotation and compression
- Consistent formatting across all components
- Component-specific log files
- Configurable log levels and outputs
- JSON structured logging support

Usage:
    from src.utils.logging_config import setup_logging
    
    logger = setup_logging(component_name="data_pipeline")
    logger.info("Pipeline started with {files} files", files=87)
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from loguru import logger


def setup_logging(
    component_name: Optional[str] = None,
    log_level: str = "INFO",
    log_dir: Union[str, Path] = "logs",
    enable_console: bool = True,
    enable_file: bool = True,
    json_format: bool = False,
    rotation: str = "10 MB",
    retention: str = "30 days",
    compression: str = "zip",
    use_timestamp: bool = True
) -> "logger":
    """
    Setup centralized logging configuration for pipeline components.
    
    Parameters:
    ----------
    component_name : str, optional
        Name of the component (creates component-specific log file)
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_dir : str or Path
        Directory to store log files
    enable_console : bool
        Enable console (terminal) output
    enable_file : bool
        Enable file logging
    json_format : bool
        Use JSON structured logging format
    rotation : str
        Log file rotation size/time (e.g., "10 MB", "1 day")
    retention : str
        How long to keep log files (e.g., "30 days", "1 week")
    compression : str
        Compression format for rotated logs ("zip", "gz", etc.)
    use_timestamp : bool
        Add timestamp to log filenames to prevent overwriting
    
    Returns:
    -------
    logger
        Configured Loguru logger instance
    
    Examples:
    --------
    # Basic usage with timestamp
    >>> logger = setup_logging("multi_horizon_training")
    >>> logger.info("Training started")
    
    # With structured logging
    >>> logger = setup_logging("data_pipeline", json_format=True)
    >>> logger.info("Processing files", count=87, stage="validation")
    
    # Debug mode
    >>> logger = setup_logging("debug_session", log_level="DEBUG")
    >>> logger.debug("Detailed debugging information")
    
    # Without timestamp (legacy mode)
    >>> logger = setup_logging("visualization", use_timestamp=False)
    >>> logger.info("Creating visualizations")
    """
    
    # Remove all existing handlers to start fresh
    from loguru import logger
    logger.remove()
    
    # Setup log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Console logging setup
    if enable_console:
        if json_format:
            console_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{extra[component]}</cyan> | "
                "{message}"
            )
        else:
            console_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{extra[component]}</cyan> | "
                "<level>{message}</level>"
            )
        
        logger.add(
            sys.stdout,
            level=log_level,
            format=console_format,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
    
    # File logging setup
    if enable_file:
        # Determine log file name with optional timestamp
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if component_name:
                log_file = log_path / f"{component_name}_{timestamp}.log"
            else:
                log_file = log_path / f"pipeline_{timestamp}.log"
        else:
            if component_name:
                log_file = log_path / f"{component_name}.log"
            else:
                log_file = log_path / "pipeline.log"
        
        if json_format:
            file_format = (
                '{{"time": "{time:YYYY-MM-DD HH:mm:ss}", '
                '"level": "{level}", '
                '"component": "{extra[component]}", '
                '"function": "{function}", '
                '"line": {line}, '
                '"message": "{message}"}}'
            )
        else:
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | "
                "{extra[component]} | "
                "{function}:{line} | "
                "{message}"
            )
        
        logger.add(
            log_file,
            level=log_level,
            format=file_format,
            rotation=rotation,
            retention=retention,
            compression=compression,
            backtrace=True,
            diagnose=True,
            enqueue=True  # Thread-safe logging
        )
    
    # Add component context
    component_name = component_name or "pipeline"
    logger = logger.bind(component=component_name)
    
    # Log initialization message
    logger.info("Logging initialized for component: {component}", component=component_name)
    if enable_file:
        logger.info("Log files will be saved to: {log_dir}", log_dir=log_path.absolute())
    
    return logger


def get_logger(component_name: str, **kwargs) -> "logger":
    """
    Convenience function to get a configured logger for a component.
    
    Parameters:
    ----------
    component_name : str
        Name of the component
    **kwargs
        Additional arguments passed to setup_logging()
    
    Returns:
    -------
    logger
        Configured logger instance
    
    Examples:
    --------
    >>> logger = get_logger("data_validation")
    >>> logger.info("Validation started")
    """
    return setup_logging(component_name=component_name, **kwargs)


def setup_structured_logging(component_name: str, **kwargs) -> "logger":
    """
    Setup structured (JSON) logging for a component.
    
    Parameters:
    ----------
    component_name : str
        Name of the component
    **kwargs
        Additional arguments passed to setup_logging()
    
    Returns:
    -------
    logger
        Configured logger with JSON output
    
    Examples:
    --------
    >>> logger = setup_structured_logging("ml_training")
    >>> logger.info("Model training", model="xgboost", accuracy=0.95)
    """
    return setup_logging(component_name=component_name, json_format=True, **kwargs)


# Convenience loggers for common components
def get_pipeline_logger(**kwargs) -> "logger":
    """Get logger for data pipeline components."""
    return get_logger("data_pipeline", **kwargs)


def get_training_logger(**kwargs) -> "logger":
    """Get logger for ML training components."""
    return get_logger("ml_training", **kwargs)


def get_validation_logger(**kwargs) -> "logger":
    """Get logger for data validation components."""
    return get_logger("data_validation", **kwargs)


def get_scraping_logger(**kwargs) -> "logger":
    """Get logger for web scraping components."""
    return get_logger("web_scraping", **kwargs)


def get_visualization_logger(**kwargs) -> "logger":
    """Get logger for visualization components."""
    return get_logger("visualization", **kwargs)


# Global logger for quick access
def get_default_logger() -> "logger":
    """Get a default configured logger."""
    return setup_logging("aurevia_pipeline")