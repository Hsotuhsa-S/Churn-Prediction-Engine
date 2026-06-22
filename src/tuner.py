"""
tuner.py — Hyperparameter tuning with GridSearchCV and PredefinedSplit.

Logging
-------
    Pass a logger from pipeline_logger.get_logger() for full run tracing.
    If omitted, a no-op logger is used — module behaves exactly as before.
"""

import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src import config
from src.model_registry import get_model_config
from src.pipeline_logger import get_null_logger


def run_grid_search(
    pipeline,
    X_search: pd.DataFrame,
    y_search:  pd.Series,
    pds,
    logger: logging.Logger = None,
) -> GridSearchCV:
    """
    Fit GridSearchCV using PredefinedSplit.

    IN  : pipeline   sklearn.Pipeline
          X_search   pd.DataFrame
          y_search   pd.Series
          pds        PredefinedSplit
          logger     logging.Logger   optional run logger
    OUT : grid_search  fitted GridSearchCV
    RAISES: KeyError — config.MODEL not found in registry
    """
    log            = logger or get_null_logger()
    param_grid     = get_model_config(config.MODEL)["param_grid"]
    scoring_metric = config.MODE_CONFIGS[config.MODE]["scoring"]
    n_combos       = _count_combinations(param_grid)

    log.info(f"[tuner] Starting GridSearchCV — model={config.MODEL}  scoring={scoring_metric}")
    log.info(f"[tuner] Hyperparameter grid: {param_grid}")
    log.info(f"[tuner] Total combinations to evaluate: {n_combos}")

    grid_search = GridSearchCV(
        estimator  = pipeline,
        param_grid = param_grid,
        cv         = pds,
        scoring    = scoring_metric,
        n_jobs     = -1,
        verbose    = 1,
    )

    grid_search.fit(X_search, y_search)

    log.info(f"[tuner] GridSearchCV complete.")
    log.info(f"[tuner] Best score : {grid_search.best_score_:.4f}  ({scoring_metric})")
    log.info(f"[tuner] Best params: {grid_search.best_params_}")

    return grid_search


def get_top_results(
    grid_search: GridSearchCV,
    n: int = 5,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Extract top-n hyperparameter combinations from cv_results_.

    IN  : grid_search  fitted GridSearchCV
          n            int  number of top results (default 5)
          logger       logging.Logger  optional run logger
    OUT : pd.DataFrame  cols: Rank | Mean Test Score | <hyperparameters>
    RAISES: ValueError — if n < 1
    """
    log            = logger or get_null_logger()
    scoring_metric = config.MODE_CONFIGS[config.MODE]["scoring"]

    if n < 1:
        msg = f"n must be >= 1, got {n}"
        log.error(f"[tuner] ValueError: {msg}")
        raise ValueError(msg)

    log.info(f"[tuner] Extracting top-{n} results from cv_results_")

    results_df = pd.DataFrame(grid_search.cv_results_)

    top = (
        results_df[["params", "mean_test_score", "rank_test_score"]]
        .sort_values("rank_test_score")
        .head(n)
        .copy()
    )

    top["mean_test_score"] = top["mean_test_score"].round(4)
    top["rank_test_score"] = top["rank_test_score"].astype(int)
    top = top.rename(columns={
        "mean_test_score": f"Mean Test {scoring_metric.upper()}",
        "rank_test_score": "Rank",
    })

    params_df = top["params"].apply(pd.Series)
    top       = pd.concat([top.drop("params", axis=1), params_df], axis=1)
    top       = top.reset_index(drop=True)

    log.info(f"[tuner] Top-{n} results extracted.")

    return top


# ── Internal helper ───────────────────────────────────────────────────────────

def _count_combinations(param_grid: dict) -> int:
    """Return total number of hyperparameter combinations in param_grid."""
    count = 1
    for values in param_grid.values():
        count *= len(values)
    return count
