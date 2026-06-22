"""
feature_importance.py — Extract and plot feature importances from best estimator.

Functions:
    get_feature_importance  — returns sorted DataFrame + bar plot
Logging
-------
    Pass a logger from pipeline_logger.get_logger() for full run tracing.
    If omitted, a no-op logger is used — module behaves exactly as before.
"""

import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV

from src import config
from src.model_registry import get_model_config
from src.pipeline_logger import get_null_logger

def get_feature_importance(
    grid_search: GridSearchCV,
    top_n: int = 10,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Extract feature importances from best estimator and plot top_n features.
    Reused directly from notebook logic (both binary and multiclass identical).

    Steps:
        1. Get best pipeline from grid_search
        2. Extract feature names from ColumnTransformer (handles OHE expansion)
        3. Get feature_importances_ from the tree-based model
        4. Return sorted DataFrame + display bar plot

    IN  : grid_search  fitted GridSearchCV
          top_n        int  number of features shown in plot (default 10)
            logger       logging.Logger  optional run logger

    OUT : pd.DataFrame
          cols: Feature | Importance
          sorted descending by Importance

    RAISES: ValueError — top_n < 1
            AttributeError — model does not have feature_importances_
                             (non-tree model added to registry)
    """
    log = logger or get_null_logger()
    log.info(f"[feature_importance] Extracting feature importances — top_n={top_n}")

    if top_n < 1:
        msg = f"top_n must be >= 1, got {top_n}"
        log.error(f"[feature_importance] ValueError: {msg}")
        raise ValueError(msg)
    
    # 1. Extract pipeline components
    best_pipe  = grid_search.best_estimator_
    preprocessor = best_pipe.named_steps["prepro"]
    step_name    = get_model_config(config.MODEL)["step_name"]
    classifier   = best_pipe.named_steps[step_name]

    # 2. Feature names — handles OHE column expansion
    feature_names = preprocessor.get_feature_names_out()

    # 3. Importances from tree model
    if not hasattr(classifier, "feature_importances_"):
        msg = (
            f"Model '{config.MODEL}' does not have feature_importances_. "
            "Only tree-based models are supported."
        )
        log.error(f"[feature_importance] AttributeError: {msg}")
        raise AttributeError(msg)
    
    # Extract and log total features
    importances = classifier.feature_importances_
    log.info(f"[feature_importance] Total features after encoding: {len(feature_names)}")

    # 4. Build sorted DataFrame (reused from notebook)
    feat_imp_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances.round(4),
    }).sort_values("Importance", ascending=False, ignore_index=True)

    top_features = feat_imp_df.head(top_n)[["Feature", "Importance"]].to_string(index=False)
    log.info(f"[feature_importance] Top-{top_n} most influential features:\n{top_features}")

    # 5. Bar plot
    plot_n = min(top_n, len(feat_imp_df))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=feat_imp_df.head(plot_n),
        palette="viridis",
        ax=ax
    )
    ax.set_title(f"Top {plot_n} Feature Importances ({config.MODE.value})")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    # plt.show() - dont show here, return figure for potential saving in pipeline

    log.info(f"[feature_importance] Feature importance extraction complete.")

    return feat_imp_df, fig