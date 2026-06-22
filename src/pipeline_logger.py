"""
pipeline_logger.py — Centralised logging factory for the Churn Prediction Pipeline.

Creates a named logger per pipeline run with:
    - FileHandler  : writes INFO + ERROR to  logs/<filename>.log
    - StreamHandler: writes INFO + ERROR to  console (notebook or .py script)

Log filename is unique per run:
    pipeline_<MODE>_<MODEL>_<YYYYMMDD_HHMMSS>.log
    e.g.  pipeline_binary_random_forest_20260318_141032.log

Usage (in notebook or main .py script)
---------------------------------------
    from src.pipeline_logger import get_logger

    logger, log_path = get_logger(mode="binary", model="random_forest")

    # Pass logger to every pipeline function
    X, y, cat_features = load_data(logger=logger)
    ...
    # After MLflow run ends, upload log file as artifact
    log_pipeline_log_file(log_path)

DEBUG note
----------
All debug-level messages are currently written as INFO so they appear in both
the file and console.  To promote them to true DEBUG in future, change the two
constants below:

    FILE_LEVEL    = logging.DEBUG
    CONSOLE_LEVEL = logging.DEBUG

and replace  logger.info("DEBUG: ...")  →  logger.debug("...")  across src files.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


# ════════════════════════════════════════════════════════
# Constants — change here to adjust levels globally
# ════════════════════════════════════════════════════════

FILE_LEVEL    = logging.INFO   # Level written to .log file
CONSOLE_LEVEL = logging.INFO   # Level written to console / notebook

LOG_DIR       = Path("logs")   # Relative to project root

# Format: timestamp | LEVEL    | message
# %-8s left-pads level name to 8 chars so columns align (INFO vs ERROR)
LOG_FORMAT    = "%(asctime)s | %(levelname)-5s | %(message)s"
DATE_FORMAT   = "%Y-%m-%d %H:%M:%S"


# ════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════

def get_logger(mode: str, model: str) -> tuple[logging.Logger, Path]:
    """
    Create and return a configured logger for one pipeline run.

    Parameters
    ----------
    mode  : str   Classification mode string, e.g. "binary" or "multiclass"
    model : str   Model key string,           e.g. "random_forest" or "xgboost"

    Returns
    -------
    logger   : logging.Logger   Ready-to-use logger (INFO + ERROR to file + console)
    log_path : Path             Absolute path to the .log file (pass to MLflow later)

    Example
    -------
    >>> logger, log_path = get_logger("binary", "random_forest")
    >>> logger.info("Pipeline started")
    2026-03-18 14:10:32 | INFO  | Pipeline started
    """
    log_path  = _build_log_path(mode, model)
    logger    = _build_logger(log_path)

    logger.info(f"Logger initialised — mode={mode}  model={model}")
    logger.info(f"Log file: {log_path}")

    return logger, log_path


# ════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════

def _build_log_path(mode: str, model: str) -> Path:
    """
    Construct a unique log file path.

    Pattern: logs/pipeline_<mode>_<model>_<YYYYMMDD_HHMMSS>.log
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"pipeline_{mode}_{model}_{timestamp}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / filename


def _build_logger(log_path: Path) -> logging.Logger:
    """
    Build a Logger with one FileHandler and one StreamHandler.

    Uses the log_path stem as the logger name so multiple runs in the same
    Python session each get an independent logger (no handler duplication).
    """
    logger_name = log_path.stem          # unique per run timestamp
    logger      = logging.getLogger(logger_name)

    # Guard: if this logger already has handlers (e.g. notebook re-run), clear them
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)       # root level — handlers filter further

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── File handler ──────────────────────────────────────────────────────────
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(FILE_LEVEL)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # ── Console / stream handler ──────────────────────────────────────────────
    # stdout so Jupyter notebook cells capture it correctly
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(CONSOLE_LEVEL)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # Prevent messages from propagating to the root logger (avoids duplicates)
    logger.propagate = False

    return logger


def get_null_logger() -> logging.Logger:
    """
    Return a no-op logger.

    Used as the default inside src modules when no logger is injected:
        log = logger or get_null_logger()

    This keeps module output silent when called standalone (e.g. unit tests)
    without requiring callers to pass a logger.
    """
    null_logger = logging.getLogger("null")
    if not null_logger.hasHandlers():
        null_logger.addHandler(logging.NullHandler())
    null_logger.propagate = False
    return null_logger
