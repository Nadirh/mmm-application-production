"""
Logging configuration utilities.
"""
import logging
import logging.handlers
import structlog
from pathlib import Path
from typing import Any, Dict

from mmm.config.settings import settings


def setup_logging():
    """Configure structured logging for the application."""
    
    # Create log directory
    log_dir = Path(settings.logging.log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Configure structlog
    timestamper = structlog.processors.TimeStamper(fmt="ISO")
    
    if settings.logging.use_json:
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]
    else:
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer()
        ]
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format=settings.logging.format,
        level=getattr(logging, settings.logging.level.upper()),
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                filename=log_dir / "mmm_app.log",
                maxBytes=settings.logging.max_file_size_mb * 1024 * 1024,
                backupCount=settings.logging.backup_count
            )
        ]
    )
    
    # Set logging levels for third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)


def get_logger(name: str = None) -> Any:
    """Get a structured logger instance."""
    return structlog.get_logger(name)