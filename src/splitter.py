"""
splitter.py — Group-aware train/val/test splitting for the Churn Prediction Pipeline.

All snapshots of a customer (deal_id, company_id) are kept in the same set
to prevent data leakage. Uses GroupShuffleSplit for both splits.

Functions:
    split_data                   — 70/15/15 GroupShuffleSplit
    verify_split                 — assert zero customer overlap across sets
    summarise_target_distribution — class balance table per split

Logging
-------
    Pass a logger from pipeline_logger.get_logger() for full run tracing.
    If omitted, a no-op logger is used — module behaves exactly as before.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src import config
from src.pipeline_logger import get_null_logger


def split_data(
    X:        pd.DataFrame,
    y:        pd.Series,
    KEY_ID_COLS: list[str],
    logger: logging.Logger = None
) -> tuple:
    """
    Split data into Train (70%) / Val (15%) / Test (15%) using GroupShuffleSplit.
    All snapshots of one customer stay in the same set.

    deal_key (MultiIndex group key) is built internally from config.KEY_ID_COLS —
    KEY_ID_COLS must still be present in X at this point.

    Two-step split (reused from notebook):
        Step 1: 70% Train, 30% Temp
        Step 2: Temp split 50/50 → 15% Val, 15% Test

    IN  : X         pd.DataFrame   full feature set (KEY_ID_COLS still present)
          y         pd.Series      encoded integer labels
          KEY_ID_COLS  list[str]  columns used to identify unique customers
          logger    logging.Logger  optional run logger

    OUT : X_train, X_val, X_test   pd.DataFrame
          y_train, y_val, y_test   pd.Series

    NOTE: KEY_ID_COLS are still present in X_* splits.
          Drop them in the orchestrator AFTER verify_split(), BEFORE build_pipeline().
    """
    log = logger or get_null_logger()
    total = len(X)

    log.info(f"[splitter] Starting GroupShuffleSplit — total rows: {total}")
    log.info(
        f"[splitter] Split config — train={config.TRAIN_SIZE:.0%}  "
        f"val={config.VAL_SIZE:.0%}  test={config.VAL_SIZE:.0%}  "
        f"random_state={config.RANDOM_STATE}"
    )

    # ── Step 1: 70% Train, 30% Temp ──────────────────────────────────────────
    gss_train = GroupShuffleSplit(
        n_splits=1,
        train_size=config.TRAIN_SIZE,
        random_state=config.RANDOM_STATE,
    )
    
    # ── Build deal_key (MultiIndex group key for GroupShuffleSplit) ──────────────────────────────
    # MultiIndex of (deal_id, company_id) used by GroupShuffleSplit to keep all snapshots of a customer together.
    deal_key = pd.MultiIndex.from_frame(X[KEY_ID_COLS])

    train_idx, temp_idx = next(gss_train.split(X, y, groups=deal_key))

    X_train, X_temp = X.iloc[train_idx], X.iloc[temp_idx]
    y_train, y_temp = y.iloc[train_idx], y.iloc[temp_idx]

    # ── Step 2: Temp → 50/50 → Val (15%) + Test (15%) ────────────────────────
    groups_temp = deal_key[temp_idx]

    gss_val_test = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=config.RANDOM_STATE,
    )
    val_idx, test_idx = next(gss_val_test.split(X_temp, y_temp, groups=groups_temp))

    X_val,  X_test  = X_temp.iloc[val_idx],  X_temp.iloc[test_idx]
    y_val,  y_test  = y_temp.iloc[val_idx],  y_temp.iloc[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


def verify_split(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
    logger:  logging.Logger = None
) -> None:
    """
    Assert no customer (deal_id, company_id) appears in more than one set.
    Print customer counts and snapshot isolation check per set.

    IN  : X_train, X_val, X_test  pd.DataFrame  (KEY_ID_COLS still present)
         logger    logging.Logger  optional run logger
    OUT : None — prints verification summary

    RAISES: AssertionError — if any customer leaks across sets
    """
    log = logger or get_null_logger()
    
    train_customers = set(map(tuple, X_train[config.KEY_ID_COLS].values))
    val_customers   = set(map(tuple, X_val[config.KEY_ID_COLS].values))
    test_customers  = set(map(tuple, X_test[config.KEY_ID_COLS].values))

    train_val_overlap  = train_customers & val_customers
    train_test_overlap = train_customers & test_customers
    val_test_overlap   = val_customers   & test_customers

    log.info(f"[Verifying split] — checking for customer leakage across sets")
    log.info(f"Customer counts per set:")
    log.info(f"  Train : {len(train_customers)}")
    log.info(f"  Val   : {len(val_customers)}")
    log.info(f"  Test  : {len(test_customers)}")
    log.info(f"Overlap check:")
    log.info(f"  Train ∩ Val  : {len(train_val_overlap)} customers")
    log.info(f"  Train ∩ Test : {len(train_test_overlap)} customers")
    log.info(f"  Val   ∩ Test : {len(val_test_overlap)} customers")

    total_overlap = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
    assert total_overlap == 0, (
        f"Data leakage detected — {total_overlap} customers appear in multiple sets.\n"
        f"  Train ∩ Val  : {train_val_overlap}\n"
        f"  Train ∩ Test : {train_test_overlap}\n"
        f"  Val   ∩ Test : {val_test_overlap}"
    )
    log.info("  No leakage detected ✓")


def summarise_target_distribution(
    y_train: pd.Series,
    y_val:   pd.Series,
    y_test:  pd.Series,
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
    logger:  logging.Logger = None
) -> pd.DataFrame:
    """
    Class balance table per split — one row per class, one column per set.
    Verifies GroupShuffleSplit did not skew class ratios.

    Uses mode per customer (reused from notebook) so each customer counts once
    regardless of how many snapshots they have.

    IN  : y_*  pd.Series      encoded integer labels
          X_*  pd.DataFrame   (KEY_ID_COLS still present for groupby)
          logger  logging.Logger  optional run logger        

    OUT : pd.DataFrame
          rows  = class names  (decoded via inverse LABEL_ENCODER)
          cols  = Train | Val | Test
          cells = "N (pct%)"

    Example:
          Set        Train        Val          Test
          nonchurn   245 (72%)    52 (71%)     53 (73%)
          churn       95 (28%)    21 (29%)     20 (27%)
    """
    # Build group keys from KEY_ID_COLS still present in X_* at this stage
    train_key = pd.MultiIndex.from_frame(X_train[config.KEY_ID_COLS])
    val_key   = pd.MultiIndex.from_frame(X_val[config.KEY_ID_COLS])
    test_key  = pd.MultiIndex.from_frame(X_test[config.KEY_ID_COLS])

    # Mode per customer — each customer counts once (reused from notebook)
    get_mode = lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]

    outcomes = {
        "Train":      y_train.groupby(train_key).apply(get_mode),
        "Validation": y_val.groupby(val_key).apply(get_mode),
        "Test":       y_test.groupby(test_key).apply(get_mode),
    }

    # Count + percentage per class per set
    outcome_summary = pd.DataFrame({
        set_name: outcomes[set_name].value_counts().sort_index()
        for set_name in outcomes
    }).fillna(0).astype(int)

    for col in outcome_summary.columns:
        outcome_summary[f"{col}_%"] = (
            outcome_summary[col] / outcome_summary[col].sum() * 100
        ).round(2)

    # Decode integer labels back to class names
    inverse_label_map = {v: k for k, v in config.MODE_CONFIGS[config.MODE]['label_encoder'].items()}
    outcome_summary.index = outcome_summary.index.map(inverse_label_map)

    # Build final display table: "N (pct%)" per cell
    sets       = ["Train", "Validation", "Test"]
    set_labels = ["Train", "Val", "Test"]

    set_class_table = pd.DataFrame({
        cls: [
            f"{outcome_summary.loc[cls, s]} ({outcome_summary.loc[cls, f'{s}_%']}%)"
            for s in sets
        ]
        for cls in outcome_summary.index
    }, index=set_labels)

    set_class_table.index.name = "Set"

    # log the class balance table
    # log = logger or get_null_logger()
    # log.info("[splitter] Target distribution per set (class balance):")
    # log.info("\n" + set_class_table.to_string())

    return set_class_table
