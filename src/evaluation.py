"""
evaluator.py — Final model evaluation on the held-out test set.

Functions:
    evaluate              — public dispatcher, reads config.MODE
    _evaluate_binary      — classification report + confusion matrix + ROC curve
    _evaluate_multiclass  — classification report + confusion matrix + per-class OvR ROC
Logging
-------
    Pass a logger from pipeline_logger.get_logger() for full run tracing.
    If omitted, a no-op logger is used — module behaves exactly as before.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src import config
from src.pipeline_logger import get_null_logger


def evaluate(grid_search, 
             X_test: pd.DataFrame, 
             y_test: pd.Series,
             logger: logging.Logger = None
             ) -> dict:
    """
    Run final evaluation on held-out test set.
    Dispatches to binary or multiclass based on config.MODE.

    IN  : grid_search  fitted GridSearchCV
          X_test       pd.DataFrame  (KEY_ID_COLS already dropped)
          y_test       pd.Series     encoded integer labels
            logger       logging.Logger  optional run logger

    OUT : metrics  dict {
              'classification_report' : str,
              'confusion_matrix'      : np.ndarray,
              'roc_auc'               : float,        # binary only
              'per_class_roc_auc'     : dict, 
              'figure'                : matplotlib Figure        # multiclass only
          }

    RAISES: ValueError — unknown config.MODE
    """
    log = logger or get_null_logger()
    log.info(f"[evaluation] Starting evaluation — mode={config.MODE.value}  test_rows={len(X_test)}")

    if config.MODE == config.ClassificationType.BINARY:
        return _evaluate_binary(grid_search, X_test, y_test)
    elif config.MODE == config.ClassificationType.MULTICLASS:
        return _evaluate_multiclass(grid_search, X_test, y_test)
    else:
        msg = f"Unknown MODE: {config.MODE}"
        log.error(f"[evaluation] ValueError: {msg}")
        raise ValueError(msg)

# ── Binary ────────────────────────────────────────────────────────────────────

def _evaluate_binary(grid_search, 
                     X_test: pd.DataFrame, 
                     y_test: pd.Series,
                     logger: logging.Logger = None
                     ) -> dict:
    """
    Binary evaluation (reused from notebook):
        1. Classification report
        2. Confusion matrix
        3. ROC-AUC curve

    Target names decoded from config.LABEL_ENCODER.
    """
    log = logger or get_null_logger()
    log.info(f"[evaluation] Starting binary evaluation — test_rows={len(X_test)}")

    # Decode target names from config for display in report and plots
    label_encoder = config.MODE_CONFIGS[config.MODE]["label_encoder"]
    target_names = list(label_encoder.keys())

    # Get predictions and probabilities for evaluation
    y_pred  = grid_search.predict(X_test) 
    y_proba = grid_search.predict_proba(X_test)[:, 1]  # positive class probability

    # 1. Classification report + scalar ROC-AUC + confusion matrix    
    report = classification_report(y_test, y_pred, target_names=target_names)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    # 2. Figure: Confusion matrix + ROC curve side by side 
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=target_names,
        ax=ax[0],
        cmap="Blues",
    )
    ax[0].set_title("Confusion Matrix")

    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax[1])
    ax[1].set_title("ROC Curve")

    plt.tight_layout()
    # Add title for clarity when comparing binary vs multiclass
    plt.suptitle(f"Evaluation Metrics : ({config.MODE.value})", y=1.02)
    # plt.show() - dont show here, return figure for potential saving in pipeline

    # Log key metrics for traceability
    log.info(f"[evaluation] Binary classification report:\n{report}")
    log.info(f"[evaluation] Test ROC-AUC: {roc_auc:.4f}")
    log.info(
        f"[evaluation] Confusion matrix (rows=actual, cols=predicted):\n"
        f"{cm}"
    )
    log.info(f"[evaluation] Binary evaluation complete.")

    return {
        "classification_report": report,
        "confusion_matrix":      cm,
        "roc_auc":               roc_auc,
        "figure":                fig,  # Return the figure object for potential saving
    }


# ── Multiclass ────────────────────────────────────────────────────────────────

def _evaluate_multiclass(grid_search, 
                         X_test: pd.DataFrame, 
                         y_test: pd.Series,
                         logger: logging.Logger = None
                         ) -> dict:
    """
    Multiclass evaluation (reused from notebook):
        1. Classification report
        2. Confusion matrix
        3. Per-class One-vs-Rest ROC curves on single plot
    """
    log = logger or get_null_logger()
    log.info(f"[evaluation] Starting multiclass evaluation — test_rows={len(X_test)}")

    # For example, if label_encoder = {"renewal": 0, "churn": 1, "expansion": 2, "downsell": 3}
    label_encoder = config.MODE_CONFIGS[config.MODE]["label_encoder"]
    target_names = list(label_encoder.keys())
    classes      = list(label_encoder.values())

    # Get predictions and probabilities for evaluation
    y_pred  = grid_search.predict(X_test) 
    y_proba = grid_search.predict_proba(X_test)   # shape (n_samples, n_classes)

    # 1. Classification report
    report = classification_report(y_test, y_pred, target_names=target_names)

    # 2. Confusion matrix + ROC curves side by side
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Confusion matrix on left
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=target_names,
        ax=ax[0],
        cmap="Blues",
    )
    ax[0].set_title("Multi-Class Confusion Matrix")

    # 3. Per-class OvR ROC curves on right (reused from notebook)
    y_test_binarized = label_binarize(y_test, classes=classes)
    n_classes        = len(classes)

    per_class_auc = {}

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
        roc_auc_val  = auc(fpr, tpr)
        class_name   = target_names[i]
        per_class_auc[class_name] = round(roc_auc_val, 4)
        ax[1].plot(fpr, tpr, label=f"ROC {class_name} (AUC = {roc_auc_val:.2f})")

    ax[1].plot([0, 1], [0, 1], "k--", lw=2)
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].set_title("One-vs-Rest ROC Curves")
    ax[1].legend(loc="lower right")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.suptitle(f"Evaluation Metrics : ({config.MODE.value})", y=1.02)
    # plt.show() - dont show here, return figure for potential saving in pipeline

    # Log key metrics for traceability
    log.info(f"[evaluation] Multiclass classification report:\n{report}")
    log.info(f"[evaluation] Per-class ROC-AUC: {per_class_auc}")
    log.info(f"[evaluation] Multiclass evaluation complete.")

    return {
        "classification_report": report,
        "per_class_roc_auc":     per_class_auc,
        "figure":                fig,  # Return the figure object for potential saving
    }