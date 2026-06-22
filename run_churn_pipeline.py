"""
run_churn_pipeline.py — Command-line entry point for the Churn Prediction Pipeline.

Executes the full pipeline (preprocessing → load → split → tune → evaluate →
save → MLflow log) for either binary or multiclass churn classification.

MODEL is hard-fixed to "random_forest".  All other pipeline logic is
delegated to the existing src/* modules unchanged.

Usage
-----
    python run_churn_pipeline.py --classification_type binary
    python run_churn_pipeline.py --classification_type multiclass
    python run_churn_pipeline.py --classification_type multi    # alias

Exit codes
----------
    0  — pipeline completed successfully
    1  — pipeline failed (exception logged to MLflow run + .log file, then re-raised)
    2  — bad CLI argument (argparse default)
"""

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

# ── Project root on sys.path (handles running from project root or subdirectory)
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import mlflow
import src.config as cfg
from src.config import ClassificationType

from src.mlflow_logger import (
    enable_mlflow_autolog,
    init_mlflow,
    log_mlflow_evaluation,
    log_mlflow_feature_importance,
    log_mlflow_run_config,
    log_mlflow_sklearn_model,
    make_run_name,
    register_model,
    start_run,
    tag_gridsearch_child_runs,
    upload_pipeline_log_file,
)
from src.pipeline_logger import get_logger, get_null_logger


# ════════════════════════════════════════════════════════
# Hard-fixed constants — not exposed as CLI arguments
# ════════════════════════════════════════════════════════

_FIXED_MODEL = "random_forest"  # only model supported via this script


# ════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════

def _parse_args(argv=None) -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Returns
    -------
    argparse.Namespace
        .classification_type : ClassificationType enum value
    """
    parser = argparse.ArgumentParser(
        prog="run_churn_pipeline.py",
        description=(
            "Run the Churn Prediction Pipeline.\n"
            "MODEL is hard-fixed to 'random_forest'.\n\n"
            "Examples:\n"
            "  python run_churn_pipeline.py --classification_type binary\n"
            "  python run_churn_pipeline.py --classification_type multiclass\n"
            "  python run_churn_pipeline.py --classification_type multi"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--classification_type",
        required=True,
        choices=["binary", "multiclass", "multi"],
        metavar="TYPE",
        help=(
            "Classification mode to run. "
            "Allowed values: binary | multiclass | multi  "
            "(binary → ClassificationType.BINARY, "
            "multiclass/multi → ClassificationType.MULTICLASS)"
        ),
    )

    args = parser.parse_args(argv)

    # Map CLI string → enum
    _type_map = {
        "binary":     ClassificationType.BINARY,
        "multiclass": ClassificationType.MULTICLASS,
        "multi":      ClassificationType.MULTICLASS,
    }
    args.classification_type = _type_map[args.classification_type]

    return args


# ════════════════════════════════════════════════════════
# Preprocessing gate
# ════════════════════════════════════════════════════════

def _maybe_preprocess(logger) -> None:
    """
    Run preprocessing only when necessary.

    Skip condition (matches notebook logic exactly):
        raw_churndata.csv exists  AND  using default source path
    Otherwise run preprocess_raw_data() end-to-end.
    """
    import src.preprocessor as preprocessor

    out_path    = cfg.INPUT_PATH
    source_path = cfg.SOURCE_DATA_PATH

    using_default_source = (source_path == cfg.SOURCE_DATA_PATH)

    if out_path.exists() and using_default_source:
        logger.info(f"[PROCESSING] Source path       : {source_path}")
        logger.info(
            f"[PROCESSING] Preprocessing skipped — output already exists: {out_path}"
        )
    else:
        logger.info(f"[PROCESSING] Running preprocessing: {source_path} → {out_path}")
        preprocessor.preprocess_raw_data(source_path, out_path, logger=logger)


# ════════════════════════════════════════════════════════
# Core pipeline
# ════════════════════════════════════════════════════════

def _run_pipeline(logger, log_path) -> None:
    """
    Execute all pipeline steps inside an active MLflow run.

    Steps
    -----
    1.  Preprocessing (conditional)
    2.  Load data
    3.  Split data  +  verify  +  summarise class balance
    4.  Drop KEY_ID_COLS from feature sets
    5.  Build sklearn Pipeline  +  PredefinedSplit search set
    6.  MLflow run
        a. autolog + config params
        b. log dataset inputs
        c. GridSearchCV tuning
        d. tag child runs
        e. evaluate on test set
        f. feature importance
        g. save model (joblib)
        h. log sklearn model artifact
        i. register in MLflow Model Registry
    7.  Upload pipeline .log file as MLflow artifact (always, even on failure)
    """
    from src.data_loader import load_data
    from src.evaluation import evaluate
    from src.feature_importance import get_feature_importance
    from src.model_saver import save_model
    from src.pipeline_builder import build_pipeline, build_search_set
    from src.splitter import split_data, summarise_target_distribution, verify_split
    from src.tuner import get_top_results, run_grid_search

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    _maybe_preprocess(logger)

    # ── Step 2: Load data ─────────────────────────────────────────────────────
    logger.info("[pipeline] Loading data ...")
    X, y, cat_features = load_data(input_path=cfg.INPUT_PATH, logger=logger)

    # ── Step 3: Split + verify + summarise ───────────────────────────────────
    logger.info("[pipeline] Splitting data ...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, cfg.KEY_ID_COLS, logger=logger
    )

    verify_split(X_train, X_val, X_test, logger=logger)

    dist_table = summarise_target_distribution(
        y_train, y_val, y_test,
        X_train, X_val, X_test,
        logger=logger,
    )
    logger.info(f"[pipeline] Target distribution per split:\n{dist_table.to_string()}")

    # ── Step 4: Drop KEY_ID_COLS (must happen after verify_split) ─────────────
    X_train = X_train.drop(columns=cfg.KEY_ID_COLS)
    X_val   = X_val.drop(columns=cfg.KEY_ID_COLS)
    X_test  = X_test.drop(columns=cfg.KEY_ID_COLS)

    # ── Step 5: Build pipeline + search set ──────────────────────────────────
    logger.info("[pipeline] Building sklearn pipeline and search set ...")
    pipeline = build_pipeline(cat_features, logger=logger)
    X_search, y_search, pds = build_search_set(
        X_train, X_val, y_train, y_val, logger=logger
    )

    # ── Step 6: MLflow run ────────────────────────────────────────────────────
    run_name = make_run_name(cfg.MODEL, cfg.MODE.value, logger=logger)
    logger.info(f"[pipeline] Starting MLflow run: {run_name}")

    with start_run(run_name=run_name):
        try:
            # 6a. Autolog + config params
            enable_mlflow_autolog(logger=logger)
            log_mlflow_run_config(logger=logger)

            # 6b. Log dataset inputs
            mlflow.log_input(
                mlflow.data.from_pandas(
                    X_train,
                    source=str(cfg.INPUT_PATH),
                    name="train",
                ),
                context="train",
            )
            mlflow.log_input(
                mlflow.data.from_pandas(
                    X_test,
                    source=str(cfg.INPUT_PATH),
                    name="test",
                ),
                context="test",
            )

            # 6c. Hyperparameter tuning
            logger.info("[pipeline] Running GridSearchCV ...")
            grid_search = run_grid_search(
                pipeline, X_search, y_search, pds, logger=logger
            )

            # 6d. Tag child runs for readable MLflow UI
            tag_gridsearch_child_runs(
                mlflow.active_run().info.run_id, logger=logger
            )

            top_results = get_top_results(grid_search, n=5, logger=logger)
            logger.info(
                f"[pipeline] Top-5 hyperparameter results:\n{top_results.to_string(index=False)}"
            )

            # 6e. Evaluate on test set
            logger.info("[pipeline] Evaluating on test set ...")
            metrics = evaluate(grid_search, X_test, y_test, logger=logger)
            log_mlflow_evaluation(metrics, logger=logger)

            # 6f. Feature importance
            logger.info("[pipeline] Extracting feature importances ...")
            feat_imp_df, feat_imp_fig = get_feature_importance(
                grid_search, top_n=10, logger=logger
            )
            log_mlflow_feature_importance(feat_imp_df, feat_imp_fig, logger=logger)

            # 6g. Save model artefact (joblib + JSON metadata)
            logger.info("[pipeline] Saving model to disk ...")
            model_path = save_model(grid_search, metrics, logger=logger)

            # 6h. Log as proper MLflow sklearn model (includes env files + signature)
            log_mlflow_sklearn_model(grid_search, X_test[:5], logger=logger)

            logger.info(f"[pipeline] Pipeline complete. Model saved to: {model_path}")

            # 6i. Register in MLflow Model Registry
            register_model(cfg.MODEL, cfg.MODE.value, logger=logger)

            current_run = mlflow.active_run()
            logger.info(
                f"[pipeline] Registered model — "
                f"Run ID: {current_run.info.run_id}  "
                f"Run Name: {current_run.info.run_name}  "
                f"Experiment ID: {current_run.info.experiment_id}"
            )

        except Exception as exc:
            # Log to file so the artifact contains the full error
            logger.error(f"[pipeline] Run FAILED — {type(exc).__name__}: {exc}")

            # Surface in MLflow UI without opening the run
            mlflow.set_tag("run_status",        "FAILED")
            mlflow.set_tag("failure_exception", type(exc).__name__)
            mlflow.set_tag("failure_message",   str(exc)[:250])

            raise  # propagate → sys.exit(1) in main()

        finally:
            # Always upload the log file — captures both success and failure paths
            upload_pipeline_log_file(log_path, logger=logger)


# ════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════

def main(argv=None) -> int:
    """
    Parse CLI args, patch config, and run the pipeline.

    Returns
    -------
    int  exit code (0 = success, 1 = pipeline failure)
    """
    args = _parse_args(argv)

    # ── Patch config in-process ───────────────────────────────────────────────
    # MODEL is hard-fixed; MODE comes from CLI
    cfg.MODEL = _FIXED_MODEL
    cfg.MODE  = args.classification_type

    print(
        f"\n{'='*60}\n"
        f"  Churn Prediction Pipeline\n"
        f"  mode  : {cfg.MODE.value}\n"
        f"  model : {cfg.MODEL}  (fixed)\n"
        f"{'='*60}\n"
    )

    # ── Initialise MLflow + logger ────────────────────────────────────────────
    init_mlflow()
    logger, log_path = get_logger(mode=cfg.MODE.value, model=cfg.MODEL)

    if logger is None:
        logger = get_null_logger()

    logger.info(
        f"[pipeline] CLI invoked — "
        f"mode={cfg.MODE.value}  model={cfg.MODEL}"
    )

    # ── Execute pipeline ──────────────────────────────────────────────────────
    try:
        _run_pipeline(logger, log_path)
        print("\nPipeline finished successfully.")
        return 0

    except Exception as exc:
        print(
            f"\n[ERROR] Pipeline failed — {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
