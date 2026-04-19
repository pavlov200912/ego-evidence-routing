"""Structured logging setup for the EER project."""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """Configure root logger with a consistent format.

    Args:
        level: Logging level (default INFO).
        log_file: Optional path to write logs to in addition to stderr.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Args:
        name: Typically __name__ of the calling module.
    """
    return logging.getLogger(name)
