from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


_LOGGER_ROOT_NAME = "trading_bot"
_LOGGING_INITIALIZED = False


def _repo_root() -> Path:
    # src/bot/system_log.py -> repo root
    return Path(__file__).resolve().parents[2]


def resolve_system_log_path(path: str | Path | None = None) -> Path:
    raw = path or os.getenv("BOT_SYSTEM_LOG_PATH") or "system_log.log"
    p = Path(raw)
    if not p.is_absolute():
        p = _repo_root() / p
    return p


def setup_system_logger(
    path: str | Path | None = None,
    *,
    level: int = logging.DEBUG,
    max_bytes: int = 50 * 1024 * 1024,
    backup_count: int = 3,
) -> Path:
    """Configure the trading-bot file logger once.

    All loggers under `trading_bot.*` propagate to this handler.
    """
    global _LOGGING_INITIALIZED

    log_path = resolve_system_log_path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger(_LOGGER_ROOT_NAME)
    root_logger.setLevel(level)
    root_logger.propagate = False

    existing = None
    for handler in root_logger.handlers:
        if getattr(handler, "_system_log_path", None) == str(log_path):
            existing = handler
            break

    if existing is None:
        file_handler = RotatingFileHandler(
            log_path,
            mode="a",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(process)d:%(threadName)s | %(name)s | %(message)s"
            )
        )
        setattr(file_handler, "_system_log_path", str(log_path))
        root_logger.addHandler(file_handler)

    if not _LOGGING_INITIALIZED:
        logging.captureWarnings(True)

        old_hook = sys.excepthook

        def _hook(exc_type, exc, tb):
            try:
                logging.getLogger(f"{_LOGGER_ROOT_NAME}.uncaught").exception(
                    "Uncaught exception",
                    exc_info=(exc_type, exc, tb),
                )
            finally:
                old_hook(exc_type, exc, tb)

        sys.excepthook = _hook
        _LOGGING_INITIALIZED = True

    return log_path


def get_system_logger(name: str) -> logging.Logger:
    if name.startswith(_LOGGER_ROOT_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_ROOT_NAME}.{name}")
