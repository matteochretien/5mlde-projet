"""Logs applicatifs homogènes entre la CLI, les flows Prefect et l'API."""

from __future__ import annotations

import logging
import os
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_configured = False


def setup_logging(level: str | int | None = None) -> None:
    """Configure le logging racine une seule fois (idempotent)."""
    global _configured
    if _configured:
        return
    resolved = level or os.environ.get("HDD_RUL_LOG_LEVEL", "INFO")
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(resolved if isinstance(resolved, int) else str(resolved).upper())
    # Les bibliothèques tierces sont bavardes au niveau INFO.
    for noisy in ("urllib3", "botocore", "matplotlib", "great_expectations", "py4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
