"""
Logger Utility
===============
Centralized logging configuration using Loguru.
Logs are written to both console and rotating log files.
"""

import sys
from pathlib import Path

from loguru import logger

from config.settings import settings

# Remove default logger
logger.remove()

# --- Console Handler ---
logger.add(
    sys.stdout,
    level=settings.LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# --- File Handler (rotating) ---
log_dir = Path(settings.LOG_DIR)
log_dir.mkdir(parents=True, exist_ok=True)

logger.add(
    str(log_dir / "app_{time:YYYY-MM-DD}.log"),
    level=settings.LOG_LEVEL,
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} - "
        "{message}"
    ),
    rotation="00:00",       # Rotate at midnight
    retention="30 days",    # Keep logs for 30 days
    compression="zip",      # Compress old logs
    encoding="utf-8",
)

# Export logger for use in other modules
__all__ = ["logger"]
