"""
Logging module for the Knowledge Graph Agent.

This module provides structured logging capabilities for the application.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

from loguru import logger

from src.config.settings import LogLevel, settings


class InterceptHandler(logging.Handler):
    """
    Intercept standard logging messages and redirect them to loguru.
    
    This handler intercepts all standard library logging calls
    and redirects them to loguru for consistent formatting.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """
        Intercept logging messages and pass them to loguru.
        
        Args:
            record: The logging record to intercept
        """
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging() -> None:
    """
    Configure logging for the application.
    
    This function sets up loguru for the application with appropriate
    log level, format, and output destination based on the application
    environment.
    """
    # Clear any existing handlers
    logger.remove()
    
    # Log level from settings
    log_level = settings.log_level.value
    
    # Determine log format based on environment
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    # Set up console logging
    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        colorize=True,
    )
    
    # Set up file logging in production environment
    if settings.app_env.value == "production":
        log_file = f"logs/knowledge_graph_agent_{datetime.now().strftime('%Y%m%d')}.log"
        logger.add(
            log_file,
            format=log_format,
            level=log_level,
            rotation="10 MB",
            compression="zip",
            retention="30 days",
        )
    
    # Intercept standard library logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # List of standard library loggers to redirect
    loggers = (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
        "langchain",
        "langgraph",
        "openai",
        "httpx",
    )
    
    # Redirect standard library loggers to loguru
    for logger_name in loggers:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False
    
    logger.info(f"Logging configured with level {log_level}")


class WorkflowLogger:
    """
    Logger for LangGraph workflows with structured context.
    
    This class provides logging capabilities for workflows with
    additional context such as workflow ID, state, and progress.
    """

    def __init__(self, workflow_id: str, workflow_type: str) -> None:
        """
        Initialize workflow logger.
        
        Args:
            workflow_id: Unique identifier for the workflow
            workflow_type: Type of workflow (e.g., "indexing", "query")
        """
        self.workflow_id = workflow_id
        self.workflow_type = workflow_type
        self.context: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
        }
    
    def update_context(self, **kwargs) -> None:
        """
        Update workflow context with additional information.
        
        Args:
            **kwargs: Additional context information
        """
        self.context.update(kwargs)
    
    def _log(self, level: str, message: str, **kwargs) -> None:
        """
        Log a message with workflow context.
        
        Args:
            level: Log level
            message: Log message
            **kwargs: Additional context information for this log entry
        """
        # Combine persistent context with log-specific context
        context = {**self.context, **kwargs}
        
        # Create structured log message
        log_message = f"[{self.workflow_type}] [{self.workflow_id}] {message}"
        
        # Log with the appropriate level
        logger.opt(depth=1).log(level, log_message, **context)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        self._log("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        self._log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        self._log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        self._log("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message."""
        self._log("CRITICAL", message, **kwargs)


def get_workflow_logger(workflow_id: str, workflow_type: str) -> WorkflowLogger:
    """
    Get a workflow logger instance.
    
    Args:
        workflow_id: Unique identifier for the workflow
        workflow_type: Type of workflow (e.g., "indexing", "query")
        
    Returns:
        WorkflowLogger instance
    """
    return WorkflowLogger(workflow_id, workflow_type)
