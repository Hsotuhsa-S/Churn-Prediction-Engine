"""
preprocessor.py — Modular preprocessing functions for raw source data.

Converts trajectory_snapshots.csv → raw_churndata_00.csv.
Main pipeline checks skip condition before calling these functions.

All column names and values read from config — no hardcoded strings.

Logging
-------
    Pass a logger from pipeline_logger.get_logger() for full run tracing.
    If omitted, a no-op logger is used — module behaves exactly as before.
"""

import logging
from pathlib import Path
import pandas as pd

from src import config
from src.pipeline_logger import get_null_logger


def load_source_data(
    raw_path: Path = config.SOURCE_DATA_PATH,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Load source data from CSV file.

    IN  : raw_path  Path   source CSV (default: config.SOURCE_DATA_PATH)
          logger    Logger optional run logger
    OUT : DataFrame        loaded data
    RAISES: FileNotFoundError — raw_path does not exist
    """
    log      = logger or get_null_logger()
    raw_path = Path(raw_path)

    log.info(f"[preprocessor] Loading source data: {raw_path}")

    if not raw_path.exists():
        msg = (
            f"Source data file not found: {raw_path}\n"
            "Update config.SOURCE_DATA_PATH to point to trajectory_snapshots.csv."
        )
        log.error(f"[preprocessor] FileNotFoundError: {msg}")
        raise FileNotFoundError(msg)

    df = pd.read_csv(raw_path)
    log.info(f"[preprocessor] Source data loaded — shape={df.shape}")

    return df


def validate_source_columns(
    df: pd.DataFrame,
    logger: logging.Logger = None,
) -> None:
    """
    Validate that required source columns exist in the dataframe.

    IN  : df      DataFrame   data to validate
          logger  Logger      optional run logger
    RAISES: KeyError — MCLASS_TARGET_COL not found in df
    """
    log = logger or get_null_logger()

    log.info(f"[preprocessor] Validating source columns — looking for '{config.MCLASS_TARGET_COL}'")

    if config.MCLASS_TARGET_COL not in df.columns:
        msg = (
            f"Source column '{config.MCLASS_TARGET_COL}' not found in source data.\n"
            f"Available columns: {df.columns.tolist()}"
        )
        log.error(f"[preprocessor] KeyError: {msg}")
        raise KeyError(msg)

    log.info(f"[preprocessor] Source column validation passed.")


def create_binary_target(
    df: pd.DataFrame,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Create binary target column from multiclass outcome column.

    IN  : df      DataFrame   data with multiclass target
          logger  Logger      optional run logger
    OUT : DataFrame           data with added binary target column
    """
    log = logger or get_null_logger()

    log.info(
        f"[preprocessor] Creating binary target '{config.BINARY_TARGET_COL}' "
        f"— positive class='{config.BINARY_POSITIVE_CLASS}'"
    )

    non_positive = next(
        k for k, v in config.MODE_CONFIGS[config.MODE]["label_encoder"].items()
        if v != config.BINARY_POSITIVE_CLASS
    )

    df[config.BINARY_TARGET_COL] = df[config.MCLASS_TARGET_COL].apply(
        lambda x: config.BINARY_POSITIVE_CLASS if x == config.BINARY_POSITIVE_CLASS
                  else non_positive
    )

    dist = df[config.BINARY_TARGET_COL].value_counts().to_dict()
    log.info(f"[preprocessor] Binary target distribution: {dist}")

    return df


def apply_missing_value_imputation(
    df: pd.DataFrame,
    fillna_cols: dict = None,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Apply missing value imputation.

    IN  : df           DataFrame   data with missing values
          fillna_cols  dict        column:fill_value mapping (default: config.FILLNA_COLS)
          logger       Logger      optional run logger
    OUT : DataFrame                data with imputed values
    RAISES: KeyError — fillna_cols references columns not in df
    """
    log = logger or get_null_logger()

    if fillna_cols is None:
        fillna_cols = config.FILLNA_COLS

    log.info(f"[preprocessor] Applying missing value imputation — columns: {list(fillna_cols.keys())}")

    missing_cols = [c for c in fillna_cols if c not in df.columns]
    if missing_cols:
        msg = (
            f"fillna_cols references columns not found in source data: {missing_cols}\n"
            "Verify column names match actual data."
        )
        log.error(f"[preprocessor] KeyError: {msg}")
        raise KeyError(msg)

    for col, fill_value in fillna_cols.items():
        n_missing = df[col].isna().sum()
        df[col]   = df[col].fillna(fill_value)
        log.info(f"[preprocessor] '{col}': filled {n_missing} missing values with {fill_value!r}")

    return df


def save_preprocessed_data(
    df: pd.DataFrame,
    out_path: Path = config.INPUT_PATH,
    logger: logging.Logger = None,
) -> Path:
    """
    Save preprocessed data to CSV file.

    IN  : df        DataFrame   data to save
          out_path  Path        output CSV (default: config.INPUT_PATH)
          logger    Logger      optional run logger
    OUT : Path      path to saved file
    """
    log      = logger or get_null_logger()
    out_path = Path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    log.info(f"[preprocessor] Preprocessed data saved — path={out_path}  shape={df.shape}")

    return out_path


def preprocess_raw_data(
    raw_path: Path = config.SOURCE_DATA_PATH,
    out_path: Path = config.INPUT_PATH,
    logger: logging.Logger = None,
) -> Path:
    """
    Orchestrate all preprocessing steps.

    Steps:
        1. Load source data
        2. Validate required source columns
        3. Create binary target column
        4. Apply missing value imputation
        5. Save to out_path

    IN  : raw_path  Path   source CSV (default: config.SOURCE_DATA_PATH)
          out_path  Path   output CSV (default: config.INPUT_PATH)
          logger    Logger optional run logger
    OUT : out_path  Path   path to saved file
    RAISES: FileNotFoundError — raw_path does not exist
            KeyError          — required columns missing from data
    """
    log = logger or get_null_logger()

    log.info(f"[preprocessor] Starting preprocessing: {raw_path} → {out_path}")

    df       = load_source_data(raw_path, logger=log)
    validate_source_columns(df, logger=log)
    df       = create_binary_target(df, logger=log)
    df       = apply_missing_value_imputation(df, logger=log)
    out_path = save_preprocessed_data(df, out_path, logger=log)

    log.info(f"[preprocessor] Preprocessing complete.")

    return out_path
