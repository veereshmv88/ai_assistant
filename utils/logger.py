"""
utils/logger.py — Structured Logging
======================================
Provides a Rich-formatted console logger + rotating file logger.
All modules call: log = get_logger(__name__)

Features:
  • Colour-coded log levels in terminal (via Rich)
  • Auto-rotating log file (10 MB, keeps 5 backups)
  • Per-module logger hierarchy
  • Timestamps in ISO 8601 format
"""

import logging
import logging.handlers
from pathlib import Path

try:
    from rich.logging import RichHandler
    from rich.console import Console
    _RICH_AVAILABLE = True
    _console = Console(stderr=True)
except ImportError:
    _RICH_AVAILABLE = False

# Global log level (overridden by config after init)
_DEFAULT_LEVEL = logging.INFO
_LOG_FILE: Path = Path("logs/assistant.log")
_FILE_HANDLER: logging.handlers.RotatingFileHandler | None = None
_CONFIGURED = False


def _configure(log_file: Path = _LOG_FILE, level: int = _DEFAULT_LEVEL):
    """One-time configuration of the root logger."""
    global _FILE_HANDLER, _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    root = logging.getLogger()
    root.setLevel(level)

    # ── Console handler ───────────────────────────────────────────────────────
    if _RICH_AVAILABLE:
        console_handler = RichHandler(
            console=_console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))

    root.addHandler(console_handler)

    # ── File handler (rotating) ───────────────────────────────────────────────
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _FILE_HANDLER = logging.handlers.RotatingFileHandler(
            str(log_file),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        _FILE_HANDLER.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
        root.addHandler(_FILE_HANDLER)
    except Exception as e:
        print(f"[WARN] Could not create log file {log_file}: {e}")


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for the given module name.
    Configures the root logger on first call.
    """
    if not _CONFIGURED:
        _configure()
    return logging.getLogger(name)


def set_level(level: str):
    """Change global log level at runtime (e.g., from config)."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger().setLevel(numeric)
