"""Logging helpers (the library never prints)."""
from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: int | str = logging.INFO, log_file: str | None = None) -> None:
    """Configure the root logger for command-line scripts.

    Args:
        level: Logging level.
        log_file: Optional file path that also receives the log records.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=_FORMAT, handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (child of the ``atandn`` namespace)."""
    return logging.getLogger(name)
