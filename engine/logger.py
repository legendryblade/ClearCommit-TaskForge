"""Structured logging helpers shared across the pipeline."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_SESSION_HANDLERS: list[logging.Handler] = []


def get_logger(name: str = "voice_translator",
               level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that emits to stderr.

    The logger is idempotent: repeated calls for the same name will not
    attach duplicate handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        # Avoid double-logging through ancestor loggers.
        logger.propagate = False
    return logger


def configure_root(level: int = logging.INFO, fmt: Optional[str] = None) -> None:
    """Set a baseline level/format for the root logger (used by libraries)."""
    logging.basicConfig(
        level=level,
        format=fmt or _DEFAULT_FORMAT,
        datefmt=_DATE_FORMAT,
        stream=sys.stderr,
        force=True,
    )


def attach_session_file(log_dir: str | Path, keep: int = 10) -> Optional[Path]:
    """Mirror all INFO+ logs into ``log_dir/session_*.log`` (keeps last N).

    Returns the created log path, or None when the directory cannot be used.
    Safe to call once per process; subsequent calls are no-ops.
    """
    if _SESSION_HANDLERS:  # already attached
        return None
    try:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        # Housekeeping: keep only the newest ``keep`` session logs.
        old = sorted(log_dir.glob("session_*.log"))
        for stale in old[:-keep] if len(old) >= keep else []:
            try:
                stale.unlink(missing_ok=True)
            except OSError:  # noqa: PERF203
                pass
        path = log_dir / f"session_{__import__('time').strftime('%Y%m%d_%H%M%S')}.log"
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)
        )
        handler.setLevel(logging.INFO)
        root = logging.getLogger()
        root.addHandler(handler)
        if root.level > logging.INFO or root.level == 0:
            root.setLevel(logging.INFO)
        # Named app loggers (stream_gui, stream_pipeline, ...) set
        # propagate=False in get_logger(), so records would never reach the
        # root handler — attach the file handler to them directly too.
        for lg in list(logging.Logger.manager.loggerDict.values()):
            if isinstance(lg, logging.Logger) and not lg.propagate:
                lg.addHandler(handler)
        _SESSION_HANDLERS.append(handler)
        return path
    except Exception:  # noqa: BLE001 - logging must never crash the app
        return None
