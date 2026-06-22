"""
config.py — Central configuration for the Churn Prediction Pipeline.

TO SWITCH MODE : change MODE  (binary ↔ multiclass)
TO SWITCH MODEL: change MODEL (must match a key in model_registry.MODEL_REGISTRY)
Everything else (scoring, target column, label encoder) resolves automatically.
"""

from enum import Enum           
from pathlib import Path

# ════════════════════════════════════════════════════════
# MLflow
# ════════════════════════════════════════════════════════

MLFLOW_TRACKING_URI    = "mlruns"
MLFLOW_EXPERIMENT_NAME = "churn-prediction"


# ════════════════════════════════════════════════════════
# Classification Mode & Model Configuration
# ════════════════════════════════════════════════════════

class ClassificationType(Enum): 
    BINARY     = "binary"
    MULTICLASS = "multiclass"

MODE  = ClassificationType.BINARY    # ← MODE = "binary"
MODEL = "random_forest" # (must match a key in model_registry.MODEL_REGISTRY)
PRE_PROCESSING_FLAG = False  # Whether to run the preprocessing step (if False, assumes preprocessed data is available)

# ════════════════════════════════════════════════════════
# MODE SETTINGS  (target col · label encoder · scoring)
# ════════════════════════════════════════════════════════

MODE_CONFIGS = {
    ClassificationType.BINARY: {         # ← : "binary"
        "target_col":    "outcome_binary",
        "label_encoder": {"nonchurn": 0, "churn": 1},
        "scoring":       "roc_auc",
    },
    ClassificationType.MULTICLASS: {     # ← : "multiclass"
        "target_col":    "outcome",
        "label_encoder": {"renewal": 0, "churn": 1, "expansion": 2, "downsell": 3},
        "scoring":       "f1_weighted",
    },
}

# Named constants derived from MODE_CONFIGS — no duplication
BINARY_TARGET_COL = MODE_CONFIGS[ClassificationType.BINARY]["target_col"]      
MCLASS_TARGET_COL = MODE_CONFIGS[ClassificationType.MULTICLASS]["target_col"]  

BINARY_POSITIVE_CLASS = "churn"  # Used in preprocessing to create binary target

# ════════════════════════════════════════════════════════
# COLUMNS
# ════════════════════════════════════════════════════════

# Columns that uniquely identify a deal (used for merging, indexing)
KEY_ID_COLS = ["deal_id", "company_id"]

# Columns to drop because of non informative.
DROP_COLS = [
    "snapshot_date",
    "renewal_stage",
    "renewal_discussions_started",
    "days_since_renewal_discussion_started",
    "final_arr",
    "industry",
    "region",
    "forecast_amount"
]

# For columns with missing values, we can specify a fill value here (instead of dropping the column).
FILLNA_COLS = {
    "days_since_renewal_discussion_started": -1,
}


# ════════════════════════════════════════════════════════
# INPUT FILES
# ════════════════════════════════════════════════════════

SOURCE_DATA_PATH = Path("data/trajectory_snapshots.csv")
INPUT_PATH    = Path("data/raw_churndata.csv")


# ════════════════════════════════════════════════════════
# Model Pipeline - SPLITTING
# ════════════════════════════════════════════════════════

TRAIN_SIZE   = 0.70
VAL_SIZE     = 0.15
RANDOM_STATE = 42


# ════════════════════════════════════════════════════════
# OUTPUT
# ════════════════════════════════════════════════════════

MODEL_DIR   = Path("model/")
RESULTS_DIR = Path("results/")
VERSIONED   = True
