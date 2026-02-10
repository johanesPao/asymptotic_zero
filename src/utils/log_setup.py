"""
Centralized log configuration for Asymptotic Zero.

Both trading_bot.py and database/logger.py import from here so rotation
settings are defined once and applied consistently.

Rotation policy:
  - Rotates daily at midnight
  - Keeps 30 days of compressed history  (.log.YYYY-MM-DD.gz)
  - UTF-8 encoding
  - One named logger per log file — safe to call get_logger() multiple times
    (won't add duplicate handlers)

Usage:
    from src.utils.log_setup import get_logger

    log = get_logger("trading")   # writes to logs/trading_errors.log
    log = get_logger("db")        # writes to logs/db_errors.log
"""

import gzip
import logging
import shutil
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────────
LOGS_DIR = Path("logs")
KEEP_DAYS = 30

LOG_FILES = {
    "trading": LOGS_DIR / "trading_errors.log",
    "db":      LOGS_DIR / "db_errors.log",
}

_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ── Compressor ─────────────────────────────────────────────────────────────────

def _compress_rotated(source: str, dest: str):
    """
    Called by the handler after each rotation.
    Compresses the just-rotated file (.log.YYYY-MM-DD) to .log.YYYY-MM-DD.gz
    and removes the uncompressed original.
    """
    with open(source, "rb") as f_in, gzip.open(dest + ".gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    Path(source).unlink(missing_ok=True)


# ── Public API ─────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.ERROR) -> logging.Logger:
    """
    Return a named logger that writes to the appropriate log file with
    daily rotation and 30-day retention.

    Args:
        name:  One of 'trading' or 'db'.
        level: Logging level (default ERROR — only errors and above).

    Returns:
        A configured logging.Logger instance.
        Calling this multiple times with the same name is safe — handlers
        are only added once.
    """
    if name not in LOG_FILES:
        raise ValueError(f"Unknown logger name {name!r}. Valid: {list(LOG_FILES)}")

    logger = logging.getLogger(f"asymptotic_zero.{name}")

    # Guard: don't add duplicate handlers on re-import / re-use
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False   # Don't bubble up to root logger

    # Ensure logs/ dir exists
    LOGS_DIR.mkdir(exist_ok=True)

    # Daily rotating handler
    handler = TimedRotatingFileHandler(
        filename=str(LOG_FILES[name]),
        when="midnight",        # rotate at midnight each day
        interval=1,             # every 1 day
        backupCount=KEEP_DAYS,  # keep 30 rotated files
        encoding="utf-8",
        utc=False,              # use local time (matches trading session timezone)
    )

    # Hook the compressor so rotated files become .gz automatically
    handler.rotator = _compress_rotated
    # Update the suffix so filenames look like: trading_errors.log.2026-02-09
    handler.suffix = "%Y-%m-%d"

    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)

    return logger
