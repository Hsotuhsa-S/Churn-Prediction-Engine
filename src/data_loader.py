"""
data_loader.py — Load and prepare data for the Churn Prediction Pipeline.

Responsibilities:
    - Load raw_churndata_00.csv
    - Drop irrelevant columns and the unused target column (based on MODE)
    - Encode target labels to integers using LABEL_ENCODER from config
    - Identify categorical feature columns for the ColumnTransformer
    - Return X, y, cat_features

Logging
-------
    Pass a logger from pipeline_logger.get_logger() for full run tracing.
    If omitted, a no-op logger is used — module behaves exactly as before.
"""

import logging
import pandas as pd
from pathlib import Path

from src import config
from src.pipeline_logger import get_null_logger


def load_data(input_path: Path = config.INPUT_PATH, 
              logger: logging.Logger = None) -> tuple:
    """
    Load and prepare data for the pipeline.

    Parameters
    ----------
    input_path : Path
        Path to the cleaned CSV file (default: config.INPUT_PATH).
    logger : logging.Logger, optional
        Logger from pipeline_logger.get_logger(). Uses no-op logger if None.

        
    Returns
    -------
    X            : pd.DataFrame  — features (includes KEY_ID_COLS for splitting)
    y            : pd.Series     — encoded integer target labels
    cat_features : list[str]     — categorical column names for OneHotEncoder

    Raises
    ------
    FileNotFoundError — input file does not exist
    KeyError          — expected columns missing from the data
    ValueError        — unknown labels found in target column
    """
    log = logger or get_null_logger()

    log.info(f"[data_loader] Loading data from: {input_path}")
    log.info(f"[data_loader] Mode: {config.MODE.value}  |  Model: {config.MODEL}")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    if not Path(input_path).exists():
        msg = (
            f"Input file not found: {input_path}\n"
            "Ensure raw_churndata_00.csv exists in the data/ folder."
        )
        log.error(f"[data_loader] FileNotFoundError: {msg}")
        raise FileNotFoundError(msg)

    df = pd.read_csv(input_path)
    log.info(f"[data_loader] Raw file loaded — shape: {df.shape}")

    # ── 2. Validate expected columns are present ──────────────────────────────
    mode_cfg      = config.MODE_CONFIGS[config.MODE]
    required_cols = config.KEY_ID_COLS + [mode_cfg['target_col']]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        msg = f"Expected columns missing from data: {missing}"
        log.error(f"[data_loader] KeyError: {msg}")
        raise KeyError(msg)

    log.info(f"[data_loader] Required columns present: {required_cols}")
    
    # ── 3. Drop irrelevant columns + the unused target column ─────────────────
    # Derive the unused target col from the two known column names — not stored in config.
    # e.g. BINARY mode → TARGET_COL = "outcome_binary", so drop "outcome"
    all_target_cols  = {config.BINARY_TARGET_COL, config.MCLASS_TARGET_COL}   # named constants
    other_target_col = (all_target_cols - {mode_cfg['target_col']}).pop()
    cols_to_drop = config.DROP_COLS + [other_target_col]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]  # skip if already absent
    df = df.drop(columns=cols_to_drop)
    log.info(f"[data_loader] Dropped {len(cols_to_drop)} columns: {cols_to_drop}")
    
    # ── 4. Split into X and y ─────────────────────────────────────────────────
    X = df.drop(columns=[mode_cfg['target_col']])
    y_raw = df[mode_cfg['target_col']]

    # ── 5. Encode target labels to integers ───────────────────────────────────
    unknown_labels = set(y_raw.unique()) - set(mode_cfg['label_encoder'].keys())
    if unknown_labels:
        msg = (
            f"Unknown labels in target column '{mode_cfg['target_col']}': {unknown_labels}\n"
            f"Expected: {list(mode_cfg['label_encoder'].keys())}"
        )
        log.error(f"[data_loader] ValueError: {msg}")
        raise ValueError(msg)

    y = y_raw.map(mode_cfg['label_encoder'])

    # ── 6. Identify categorical features ──────────────────────────────────────
    # Exclude KEY_ID_COLS — kept in X for GroupShuffleSplit but must not be
    # treated as model features by the ColumnTransformer.
    cat_features = [
        col for col in X.select_dtypes(include=["object", "category", "bool"]).columns
        if col not in config.KEY_ID_COLS
    ]

    # ── 7. Summary ────────────────────────────────────────────────────────────
    log.info(f"[data_loader] Categorical features: {len(cat_features)}  → {cat_features}")
    log.info(f"[data_loader] Target distribution :\n{y_raw.value_counts().to_string()}")
    log.info(f"[data_loader] Done.")

    return X, y, cat_features