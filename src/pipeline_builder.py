"""
pipeline_builder.py — Build sklearn Pipeline and GridSearchCV search set.

Functions:
    build_pipeline    — ColumnTransformer + estimator from model registry
    build_search_set  — combine train + val with PredefinedSplit for GridSearchCV

Logging
-------
    Pass a logger from pipeline_logger.get_logger() for full run tracing.
    If omitted, a no-op logger is used — module behaves exactly as before.      
"""

import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config
from src.model_registry import get_model_config
from src.pipeline_logger import get_null_logger


def build_pipeline(cat_features: list,
                   logger: logging.Logger = None
                   ) -> Pipeline:
    """
    Build sklearn Pipeline: ColumnTransformer → estimator.

    ColumnTransformer (matches notebook exactly):
        - OHE on cat_features   (handle_unknown='ignore')
        - remainder='passthrough'          numeric cols pass through untouched
        - verbose_feature_names_out=False  keeps original feature names

    Estimator and step_name are read from model_registry via config.MODEL.
    KEY_ID_COLS must already be dropped from X before calling this function.

    IN  : cat_features  list[str]   categorical column names from load_data()
            logger        logging.Logger     optional run logger    

    OUT : pipeline  sklearn.Pipeline
              steps: [('prepro', ColumnTransformer), (step_name, estimator)]

    RAISES: KeyError     — config.MODEL not found in registry
            RuntimeError — model package not installed
    """
    log = logger or get_null_logger()
    log.info(f"[pipeline_builder] Building pipeline — model='{config.MODEL}'")
    log.info(f"[pipeline_builder] Categorical features to encode ({len(cat_features)}): {cat_features}")

    try:
        model_cfg  = get_model_config(config.MODEL)
    except (KeyError, RuntimeError) as e:
        log.error(f"[pipeline_builder] Failed to load model config: {e}")   
        raise

    estimator  = model_cfg["estimator"]
    step_name  = model_cfg["step_name"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline([
        ("prepro", preprocessor),
        (step_name, estimator),
    ])

    log.info(
        f"[pipeline_builder] Pipeline built — "
        f"steps: prepro → {step_name} ({type(estimator).__name__})"
    )
    
    return pipeline


def build_search_set(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    y_train: pd.Series,
    y_val:   pd.Series,
    logger:  logging.Logger = None
) -> tuple:
    """
    Combine train + val into a single search set for GridSearchCV
    with a PredefinedSplit that preserves the original boundary.

    PredefinedSplit assignment (reused from notebook):
        -1 → train rows   (used for fitting in each CV iteration)
         0 → val rows     (used for scoring in each CV iteration)

    This gives GridSearchCV exactly 1 fold — no random reshuffling —
    preserving the group-aware split from GroupShuffleSplit.

    IN  : X_train, X_val  pd.DataFrame  (KEY_ID_COLS already dropped)
          y_train, y_val  pd.Series
            logger          logging.Logger optional run logger

    OUT : X_search  pd.DataFrame    train + val concatenated
          y_search  pd.Series       train + val concatenated
          pds       PredefinedSplit

    RAISES: ValueError — if X_train or X_val is empty
    """
    log = logger or get_null_logger()

    log.info(
        f"[pipeline_builder] Building search set — "
        f"train={len(X_train)}  val={len(X_val)}"
    )

    if len(X_train) == 0:
        msg = "X_train is empty — cannot build search set."
        log.error(f"[pipeline_builder] ValueError: {msg}")
        raise ValueError(msg)
    if len(X_val) == 0:
        msg = "X_val is empty — cannot build search set."
        log.error(f"[pipeline_builder] ValueError: {msg}")
        raise ValueError(msg)

    X_search = pd.concat([X_train, X_val], ignore_index=True)
    y_search = pd.concat([y_train, y_val], ignore_index=True)

    # -1 = train rows, 0 = val rows
    split_indices = np.full(len(X_search), -1)
    split_indices[len(X_train):] = 0

    pds = PredefinedSplit(test_fold=split_indices)

    log.info(f"[pipeline_builder] Search set built -")
    log.info(f"  Train rows ={len(X_train)}  Val rows ={len(X_val)}")
    log.info(f"  Total ={len(X_search)} CV folds ={pds.get_n_splits()}")
    log.info(f"[pipeline_builder] PredefinedSplit: -1=train rows  0=val rows")


    return X_search, y_search, pds