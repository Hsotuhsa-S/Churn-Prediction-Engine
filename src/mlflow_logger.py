"""
mlflow_logger.py — Centralised MLflow experiment tracking.

Functions:
    init_mlflow              — set tracking URI + experiment
    make_run_name            — build versioned run name: {model}_{mode}_v{N}_{date}
    start_run                — context manager for an MLflow run
    enable_mlflow_autolog           — turn on sklearn autologging
    log_mlflow_run_config           — log pipeline config params
    log_mlflow_evaluation           — log test-set metrics + plots
    log_mlflow_model_artifact       — log joblib .pkl + .json to MLflow artifacts (obsolete — log_mlflow_sklearn_model is preferred)
    log_mlflow_sklearn_model        — log best_estimator_ as a proper MLflow sklearn model
                               (replaces log_model_artifact — includes env files,
                                model signature, and input example automatically)
    log_mlflow_feature_importance   — log feature importance table + plot
    upload_pipeline_log_file        — log any file as an MLflow artifact (e.g. pipeline log file)
    tag_gridsearch_child_runs— rename + tag autolog child runs after GridSearchCV.fit()
                               naming: hyperparameterSet-{ID}  (no fold dim: PredefinedSplit=1 fold)
    register_model           — register in MLflow Model Registry
                               (prototype: no stage, links run for traceability)

Logging
-------
    All functions accept an optional  logger=None  parameter.
    When a logger is supplied (from pipeline_logger.get_logger()), every
    print() status message is also written to the run's .log file so the
    MLflow layer is fully represented in the persistent audit trail.
    print() calls are kept alongside log.info() so console output is unchanged.
    Warnings (e.g. deleted experiment, no child runs) are logged as INFO.
"""

import logging
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import LifecycleStage

from src import config
from src.model_registry import get_model_config
from src.pipeline_logger import get_null_logger


# ── Setup ─────────────────────────────────────────────────────────────────────

def init_mlflow(logger: logging.Logger = None) -> None:
    """
    Initialise MLflow: set tracking URI and experiment.
    Call once at the start of the notebook.

    Handles edge cases:
    - Creates experiment if it doesn't exist
    - Restores experiment if it was deleted
    - Handles corrupted metadata from filesystem cleanup (recreates experiment if needed)

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger from pipeline_logger.get_logger(). Uses no-op logger if None.
    """
    log = logger or get_null_logger()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    msg_uri = f"MLflow tracking URI : {config.MLFLOW_TRACKING_URI}"
    msg_exp = f"MLflow experiment   : {config.MLFLOW_EXPERIMENT_NAME}"
    print(msg_uri)
    print(msg_exp)
    log.info(f"[mlflow_logger] {msg_uri}")
    log.info(f"[mlflow_logger] {msg_exp}")

    # Get or create experiment
    try:
        experiment = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    except Exception as e:
        msg = (
            f"Error querying experiments: {e}  "
            f"Assuming experiment was hard-deleted. Creating fresh experiment..."
        )
        print(f"⚠ Warning: {msg}")
        log.info(f"[mlflow_logger] WARNING: {msg}")
        mlflow.create_experiment(config.MLFLOW_EXPERIMENT_NAME)
        experiment = None

    if experiment is None:
        msg = f"Creating new MLflow experiment: {config.MLFLOW_EXPERIMENT_NAME}"
        print(msg)
        log.info(f"[mlflow_logger] {msg}")
        mlflow.create_experiment(config.MLFLOW_EXPERIMENT_NAME)

    elif experiment.lifecycle_stage == LifecycleStage.DELETED:
        client = MlflowClient()
        client.restore_experiment(experiment.experiment_id)
        msg = f"Restoring deleted MLflow experiment: {config.MLFLOW_EXPERIMENT_NAME}"
        print(msg)
        log.info(f"[mlflow_logger] {msg}")


def make_run_name(model: str, mode_str: str, logger: logging.Logger = None) -> str:
    """
    Build a versioned, dated run name from current config.

    Pattern : {model}_{mode}_v{N}_{YYYYMMDD}
    Example : random_forest_binary_v3_20250226

    Version N is auto-incremented by counting existing runs that share
    the same {model}_{mode} base — so each new run gets the next number.

    Parameters
    ----------
    model    : str   model name (key in model_registry)
    mode_str : str   mode name as string e.g. "binary" or "multiclass"
    logger   : logging.Logger, optional
    """
    log       = logger or get_null_logger()
    base_name = f"{model}_{mode_str}"
    date      = datetime.now().strftime("%Y%m%d")

    experiment = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        version = 1
    else:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f'tags."mlflow.runName" LIKE "{base_name}_v%"',
            output_format="list"
        )
        version = len(runs) + 1

    run_name = f"{base_name}_v{version}_{date}"
    log.info(f"[mlflow_logger] Run name generated: {run_name}")
    return run_name


def start_run(run_name: str):
    """
    Return an mlflow.start_run() context manager.

    Usage:
        with start_run("random_forest_binary"):
            ...
    """
    return mlflow.start_run(run_name=run_name)


def enable_mlflow_autolog(logger: logging.Logger = None) -> None:
    """
    Enable sklearn autologging.

    Auto-captures from GridSearchCV.fit():
        - best_params_ as params
        - best_score_ as metric
        - cv_results_ as artifact
        - best_estimator_ as model artifact

    Parameters
    ----------
    logger : logging.Logger, optional
    """
    log = logger or get_null_logger()

    mlflow.sklearn.autolog(
        log_models=False,
        log_datasets=False,
        log_input_examples=False,
        max_tuning_runs=5
    )

    log.info("[mlflow_logger] sklearn autologging enabled — log_models=False  max_tuning_runs=5")


# ── Manual logging ────────────────────────────────────────────────────────────

def log_mlflow_run_config(logger: logging.Logger = None) -> None:
    """
    Log pipeline configuration params that autolog does not capture.

    Parameters
    ----------
    logger : logging.Logger, optional
    """
    log      = logger or get_null_logger()
    mode_cfg = config.MODE_CONFIGS[config.MODE]

    params = {
        "mode":         config.MODE.value,
        "model":        config.MODEL,
        "train_size":   config.TRAIN_SIZE,
        "val_size":     config.VAL_SIZE,
        "random_state": config.RANDOM_STATE,
        "scoring":      mode_cfg["scoring"],
        "target_col":   mode_cfg["target_col"],
        "input_path":   str(config.INPUT_PATH),
    }

    mlflow.log_params(params)
    log.info(f"[mlflow_logger] Run config logged to MLflow params: {params}")


# ── GridSearch child run tagging ─────────────────────────────────────────────

def tag_gridsearch_child_runs(
    parent_run_id: str,
    logger: logging.Logger = None,
) -> None:
    """
    Rename every child run created by GridSearchCV.

    Runs are sorted by start time and numbered sequentially.
    Naming pattern : CVFold0_hyperparameterSet-{ID}
    Example        : CVFold0_hyperparameterSet-047

    CVFold0 is fixed — PredefinedSplit has exactly 1 fold.

    Parameters
    ----------
    parent_run_id : str   mlflow.active_run().info.run_id
    logger        : logging.Logger, optional
    """
    log    = logger or get_null_logger()
    client = MlflowClient()

    exp_id     = client.get_run(parent_run_id).info.experiment_id
    child_runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string=f'tags."mlflow.parentRunId" = "{parent_run_id}"',
        max_results=1000,
    )

    if not child_runs:
        msg = "MLflow: no child runs found — ensure enable_autolog() is called before fit()."
        print(msg)
        log.info(f"[mlflow_logger] WARNING: {msg}")
        return

    child_runs = sorted(child_runs, key=lambda r: r.info.start_time)

    for set_id, run in enumerate(child_runs):
        client.update_run(run.info.run_id, name=f"CVFold0_hyperparameterSet-{set_id:03d}")

    msg = (
        f"MLflow: renamed {len(child_runs)} child runs  →  "
        f"CVFold0_hyperparameterSet-000 ... "
        f"CVFold0_hyperparameterSet-{len(child_runs)-1:03d}"
    )
    print(msg)
    log.info(f"[mlflow_logger] {msg}")


def log_mlflow_evaluation(
    metrics: dict,
    logger: logging.Logger = None,
) -> None:
    """
    Log test-set evaluation metrics and artifact plots.

    Parameters
    ----------
    metrics : dict returned by evaluate()
              Must contain 'classification_report' (str).
              Binary     → 'roc_auc' (float), 'figure' (matplotlib Figure)
              Multiclass → 'per_class_roc_auc' (dict), 'figure' (matplotlib Figure)
    logger  : logging.Logger, optional
    """
    log = logger or get_null_logger()

    # Binary: log scalar roc_auc
    if "roc_auc" in metrics:
        mlflow.log_metric("test_roc_auc", metrics["roc_auc"])
        log.info(f"[mlflow_logger] Logged metric — test_roc_auc={metrics['roc_auc']:.4f}")

    # Multiclass: log per-class roc_auc values
    if "per_class_roc_auc" in metrics:
        for class_name, auc_val in metrics["per_class_roc_auc"].items():
            mlflow.log_metric(f"test_roc_auc_{class_name}", auc_val)
        log.info(f"[mlflow_logger] Logged per-class ROC-AUC: {metrics['per_class_roc_auc']}")

    # Log classification report as text artifact
    mlflow.log_text(metrics["classification_report"], "classification_report.txt")
    log.info("[mlflow_logger] Logged artifact — classification_report.txt")

    # Log evaluation figure (confusion matrix + ROC curve)
    if "figure" in metrics and metrics["figure"] is not None:
        mlflow.log_figure(metrics["figure"], "evaluation_plots.png")
        log.info("[mlflow_logger] Logged artifact — evaluation_plots.png")

    msg = "MLflow: evaluation metrics and artifacts logged."
    print(msg)
    log.info(f"[mlflow_logger] {msg}")


def log_mlflow_sklearn_model(
    grid_search,
    X_sample,
    logger: logging.Logger = None,
) -> str:
    """
    Log the best sklearn estimator as a proper MLflow model artifact.

    Packages:
        sklearn_model/
        ├── MLmodel
        ├── model.pkl
        ├── conda.yaml
        ├── requirements.txt
        └── python_env.yaml

    Parameters
    ----------
    grid_search : GridSearchCV  fitted grid search object
    X_sample    : pd.DataFrame  small sample of X_test for signature inference
    logger      : logging.Logger, optional

    Returns
    -------
    str  artifact URI of the logged model
    """
    log        = logger or get_null_logger()
    best_model = grid_search.best_estimator_

    log.info("[mlflow_logger] Logging sklearn model — inferring signature from X_sample")

    predictions = best_model.predict(X_sample)
    signature   = infer_signature(X_sample, predictions)

    model_info = mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="sklearn_model",
        signature=signature,
        input_example=X_sample[:3],
    )

    msg_uri      = f"MLflow: logged sklearn model → {model_info.model_uri}"
    msg_artifacts = "        Artifacts: model.pkl, conda.yaml, requirements.txt, python_env.yaml"
    print(msg_uri)
    print(msg_artifacts)
    log.info(f"[mlflow_logger] {msg_uri}")
    log.info(f"[mlflow_logger] {msg_artifacts.strip()}")

    return model_info.model_uri


def log_mlflow_model_artifact(
    model_path,
    logger: logging.Logger = None,
) -> None:
    """
    Log the saved joblib model (.pkl) and its companion metadata (.json)
    as MLflow artifacts. (Obsolete — log_mlflow_sklearn_model is preferred.)

    Parameters
    ----------
    model_path : Path  returned by model_saver.save_model()
    logger     : logging.Logger, optional
    """
    log        = logger or get_null_logger()
    model_path = Path(model_path)

    mlflow.log_artifact(str(model_path), artifact_path="model_joblib")
    msg = f"MLflow: logged artifact {model_path.name}"
    print(msg)
    log.info(f"[mlflow_logger] {msg}")

    metadata_path = model_path.with_suffix(".json")
    if metadata_path.exists():
        mlflow.log_artifact(str(metadata_path), artifact_path="model_joblib")
        msg = f"MLflow: logged artifact {metadata_path.name}"
        print(msg)
        log.info(f"[mlflow_logger] {msg}")


def log_mlflow_feature_importance(
    feat_imp_df,
    fig=None,
    logger: logging.Logger = None,
) -> None:
    """
    Log feature importance DataFrame and bar plot.

    Parameters
    ----------
    feat_imp_df : pd.DataFrame  (Feature, Importance)
    fig         : matplotlib Figure (optional)
    logger      : logging.Logger, optional
    """
    log = logger or get_null_logger()

    mlflow.log_table(feat_imp_df, artifact_file="feature_importance.json")
    log.info("[mlflow_logger] Logged artifact — feature_importance.json")

    if fig is not None:
        mlflow.log_figure(fig, "feature_importance.png")
        log.info("[mlflow_logger] Logged artifact — feature_importance.png")

    msg = "MLflow: feature importance logged."
    print(msg)
    log.info(f"[mlflow_logger] {msg}")


# ── Model Registry ────────────────────────────────────────────────────────────

def register_model(
    model_str: str,
    mode_str: str,
    logger: logging.Logger = None,
) -> str:
    """
    Register the best estimator from the active run into the MLflow Model Registry.

    Prototype — no stage is set (Staging / Production decisions happen later).
    Tags link the registered version back to the originating run for full
    traceability when a deployment decision is eventually made.

    Registry name pattern : churn_{mode}_{model}
    Example               : churn_binary_random_forest
                            churn_multiclass_xgboost

    Parameters
    ----------
    model_str : str  model name (key in model_registry.MODEL_REGISTRY)
    mode_str  : str  mode name as string e.g. "binary" or "multiclass"
    logger    : logging.Logger, optional

    Returns
    -------
    str — registered model version number
    """
    log = logger or get_null_logger()

    run = mlflow.active_run()
    if run is None:
        msg = "register_model() must be called inside an active MLflow run."
        log.error(f"[mlflow_logger] RuntimeError: {msg}")
        raise RuntimeError(msg)

    run_id        = run.info.run_id
    run_name      = run.data.tags.get("mlflow.runName", run_id)
    registry_name = f"churn_{mode_str}_{model_str}"
    model_uri     = f"runs:/{run_id}/sklearn_model"

    log.info(
        f"[mlflow_logger] Registering model — "
        f"registry='{registry_name}'  run_id={run_id}  uri={model_uri}"
    )

    client = MlflowClient()
    try:
        client.create_registered_model(registry_name)
    except mlflow.exceptions.MlflowException:
        pass  # already exists

    mv = client.create_model_version(
        name=registry_name,
        source=model_uri,
        run_id=run_id
    )

    client.set_model_version_tag(registry_name, mv.version, "mode",     mode_str)
    client.set_model_version_tag(registry_name, mv.version, "model",    model_str)
    client.set_model_version_tag(registry_name, mv.version, "run_name", run_name)
    client.set_model_version_tag(registry_name, mv.version, "run_id",   run_id)

    msg_reg   = f"MLflow: registered  '{registry_name}'  version {mv.version}"
    msg_run   = f"        run         : {run_name}"
    msg_id    = f"        run_id      : {run_id}"
    msg_stage = f"        stage       : None  — promote manually when ready"

    print(msg_reg)
    print(msg_run)
    print(msg_id)
    print(msg_stage)

    log.info(f"[mlflow_logger] {msg_reg}")
    log.info(f"[mlflow_logger] run={run_name}  run_id={run_id}  version={mv.version}")
    log.info(f"[mlflow_logger] stage=None — promote manually when ready")

    return mv.version


# ── Pipeline log upload ───────────────────────────────────────────────────────

def upload_pipeline_log_file(
    log_path: Path,
    logger: logging.Logger = None,
) -> None:
    """
    Upload the pipeline run log file as an MLflow artifact.

    Saves the .log file under the 'logs/' subfolder in the MLflow run artifacts,
    so it appears in the MLflow UI alongside evaluation plots and model files.

    Call this once at the very end of the  with start_run(...)  block,
    after register_model() — the log file is complete at that point.

    Parameters
    ----------
    log_path : Path   path to the .log file from pipeline_logger.get_logger()
    logger   : logging.Logger, optional

    MLflow artifact path
    --------------------
        <run>/artifacts/logs/pipeline_binary_random_forest_20260318_141032.log

    Raises
    ------
    FileNotFoundError                 — log_path does not exist
    mlflow.exceptions.MlflowException — no active run
    """
    log      = logger or get_null_logger()
    log_path = Path(log_path)

    if not log_path.exists():
        msg = (
            f"Log file not found: {log_path}\n"
            "Ensure the logger wrote to this path before calling upload_pipeline_log_file()."
        )
        log.error(f"[mlflow_logger] FileNotFoundError: {msg}")
        raise FileNotFoundError(msg)

    mlflow.log_artifact(str(log_path), artifact_path="logs")

    msg = f"MLflow: pipeline log uploaded → logs/{log_path.name}"
    print(msg)
    log.info(f"[mlflow_logger] {msg}")
    log.info(f"[mlflow_logger] Log file upload complete — run artifact trail is closed.")
