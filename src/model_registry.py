"""
model_registry.py — Available models for the Churn Prediction Pipeline.

HOW TO ADD A NEW MODEL:
    1. Add a new entry to MODEL_REGISTRY.
    2. Set a unique string key — this is what you put in config.py MODEL = "..."
    3. Specify: estimator, step_name, param_grid.
    4. Nothing else changes.
"""

from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


MODEL_REGISTRY = {

    "random_forest": {
        "estimator": RandomForestClassifier(random_state=42),
        # step_name is the Pipeline key AND the prefix for param_grid entries
        "step_name": "rf",
        "param_grid": {
            "rf__n_estimators":      [100, 200],
            "rf__max_depth":         [10, 20],
            "rf__min_samples_split": [2, 5, 10],
            "rf__min_samples_leaf":  [1, 2, 4],
            "rf__max_features":      ["sqrt", "log2", 0.3, 0.5],
            "rf__class_weight":      ["balanced", "balanced_subsample"],
        },
    },

    "xgboost": {
        "estimator": (
            XGBClassifier(random_state=42)
            if _XGBOOST_AVAILABLE else None
        ),
        "step_name": "xgb",
        "param_grid": {
            "xgb__n_estimators":     [100, 200],
            "xgb__max_depth":        [3, 6, 10],
            "xgb__learning_rate":    [0.01, 0.1, 0.3],
            "xgb__subsample":        [0.7, 1.0],
            "xgb__colsample_bytree": [0.7, 1.0],
            "xgb__scale_pos_weight": [1, 5],
        },
    },

}


def get_model_config(model_name: str) -> dict:
    """
    Return model config from the registry.

    Raises
    ------
    KeyError     — model_name not in registry
    RuntimeError — model registered but package not installed (e.g. xgboost)
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{model_name}' not found in MODEL_REGISTRY. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    cfg = MODEL_REGISTRY[model_name]

    if cfg["estimator"] is None:
        raise RuntimeError(
            f"Model '{model_name}' is registered but its package is not installed. "
            f"Run: pip install {model_name.replace('_', '-')}"
        )

    return cfg