"""
model_saver.py — Save best estimator and metadata to disk.

Functions:
    save_model  — saves .pkl + companion .json metadata file

Logging
-------
    Pass a logger from pipeline_logger.get_logger() for full run tracing.
    If omitted, a no-op logger is used — module behaves exactly as before.
"""


import json
import logging
import joblib
from datetime import datetime
from pathlib import Path

from src import config
from src.pipeline_logger import get_null_logger


def save_model(grid_search, 
               metrics: dict, 
               logger: logging.Logger = None
               ) -> Path:
    """
    Save best estimator from GridSearchCV to model/ folder.

    Filenames (config.VERSIONED = True):
        model/<model>_<mode>_<YYYYMMDD_HHMMSS>.pkl
        model/<model>_<mode>_<YYYYMMDD_HHMMSS>.json
    Filenames (config.VERSIONED = False):
        model/<model>_<mode>.pkl
        model/<model>_<mode>.json

    Companion JSON contains:
        mode, model, best_params, scoring, score_value, saved_at

    IN  : grid_search  fitted GridSearchCV
          metrics      dict from evaluate()
            logger       logging.Logger  optional run logger

    OUT : model_path   Path  path to saved .pkl

    RAISES: OSError — model_dir not writable
    """
    log = logger or get_null_logger()
    log.info(
        f"[model_saver] Saving model — "
        f"mode={config.MODE.value}  model={config.MODEL}  versioned={config.VERSIONED}"
    )

    # ── Build filename stem ───────────────────────────────────────────────────

    stem = f"{config.MODEL}_{config.MODE.value}"
    if config.VERSIONED:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{stem}_{timestamp}"

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path    = config.MODEL_DIR / f"{stem}.pkl"
    metadata_path = config.MODEL_DIR / f"{stem}.json"

    # ── Save model ────────────────────────────────────────────────────────────
    try:
        joblib.dump(grid_search.best_estimator_, model_path)
        log.info(f"[model_saver] Model saved: {model_path}")
    except OSError as e:
        log.error(f"[model_saver] OSError saving model to {model_path}: {e}")
        raise

    # ── Save metadata ─────────────────────────────────────────────────────────
    # Extract scalar score value from metrics — key differs by mode
    score_value = metrics.get("roc_auc") or metrics.get("per_class_roc_auc")
    if isinstance(score_value, dict):
        score_value = {k: round(v, 4) for k, v in score_value.items()}

    metadata = {
        "mode":        config.MODE.value,
        "model":       config.MODEL,
        "scoring":     config.MODE_CONFIGS[config.MODE]["scoring"],
        "score_value": score_value,
        "best_params": grid_search.best_params_,
        "saved_at":    datetime.now().isoformat(),
    }

    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        log.info(f"[model_saver] Metadata saved: {metadata_path}")
    except OSError as e:
        log.error(f"[model_saver] OSError saving metadata to {metadata_path}: {e}")
        raise
    
    log.info(f"[model_saver] Best params: {grid_search.best_params_}")
    log.info(f"[model_saver] Score ({config.MODE_CONFIGS[config.MODE]['scoring']}): {score_value}")

    return model_path
