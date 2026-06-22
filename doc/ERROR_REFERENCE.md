# Error Reference — Churn Prediction Pipeline

This document lists all explicitly raised errors across `src/` modules, their origin function, and what triggers them. Useful for debugging and understanding pipeline failure modes. For a pipeline overview, see [`README.md`](../README.md).

---

## How Errors Surface

- Every error is logged via `log.error(...)` before being raised, so it always appears in `logs/pipeline_<mode>_<model>_YYYYMMDD_HHMMSS.log`.
- On pipeline failure (`run_churn_pipeline.py`), the exception type and message are also tagged to the MLflow run (`failure_exception`, `failure_message` tags) and the exit code is `1`.

---

## Error Catalogue

| # | Error Type | Trigger Condition | Module | Function | Lines |
|---|-----------|-------------------|--------|----------|-------|
| 1 | `FileNotFoundError` | Input file not found: `{input_path}` — Ensure `raw_churndata_00.csv` exists in the `data/` folder | `src/data_loader.py` | `load_data()` | L56–62 |
| 2 | `KeyError` | Expected columns missing from data: `{missing}` | `src/data_loader.py` | `load_data()` | L71–74 |
| 3 | `ValueError` | Unknown labels in target column `'{target_col}'`: `{unknown_labels}` — Expected: `{...}` | `src/data_loader.py` | `load_data()` | L94–100 |
| 4 | `FileNotFoundError` | Source data file not found: `{raw_path}` — Update `config.SOURCE_DATA_PATH` to point to `trajectory_snapshots.csv` | `src/preprocessor.py` | `load_source_data()` | L40–46 |
| 5 | `KeyError` | Source column `'{config.MCLASS_TARGET_COL}'` not found in source data — Available columns: `{...}` | `src/preprocessor.py` | `validate_source_columns()` | L69–75 |
| 6 | `KeyError` | `fillna_cols` references columns not found in source data: `{missing_cols}` — Verify column names match actual data | `src/preprocessor.py` | `apply_missing_value_imputation()` | L136–142 |
| 7 | `AssertionError` | Data leakage detected — `{N}` customers appear in multiple sets. Train∩Val, Train∩Test, Val∩Test overlap details | `src/splitter.py` | `verify_split()` | L133–139 |
| 8 | `KeyError` | `config.MODEL` not found in `MODEL_REGISTRY` — propagated from `get_model_config()` | `src/pipeline_builder.py` | `build_pipeline()` | L54–58 |
| 9 | `RuntimeError` | Model package not installed (e.g. xgboost) — propagated from `get_model_config()` | `src/pipeline_builder.py` | `build_pipeline()` | L54–58 |
| 10 | `ValueError` | `X_train` is empty — cannot build search set | `src/pipeline_builder.py` | `build_search_set()` | L119–122 |
| 11 | `ValueError` | `X_val` is empty — cannot build search set | `src/pipeline_builder.py` | `build_search_set()` | L123–126 |
| 12 | `KeyError` | `config.MODEL` not found in registry — propagated from `get_model_config()` | `src/tuner.py` | `run_grid_search()` | L38 |
| 13 | `ValueError` | `n` must be >= 1, got `{n}` | `src/tuner.py` | `get_top_results()` | L81–84 |
| 14 | `ValueError` | Unknown `MODE`: `{config.MODE}` | `src/evaluation.py` | `evaluate()` | L64–67 |
| 15 | `ValueError` | `top_n` must be >= 1, got `{top_n}` | `src/feature_importance.py` | `get_feature_importance()` | L52–55 |
| 16 | `AttributeError` | Model `'{config.MODEL}'` does not have `feature_importances_` — Only tree-based models are supported | `src/feature_importance.py` | `get_feature_importance()` | L67–73 |
| 17 | `OSError` | `model_dir` not writable — raised when saving `.pkl` | `src/model_saver.py` | `save_model()` | L67–72 |
| 18 | `OSError` | `model_dir` not writable — raised when saving `.json` metadata | `src/model_saver.py` | `save_model()` | L89–95 |
| 19 | `KeyError` | `model_name` not found in `MODEL_REGISTRY` — Available: `{list(...)}` | `src/model_registry.py` | `get_model_config()` | L64–68 |
| 20 | `RuntimeError` | Model `'{model_name}'` is registered but its package is not installed — Run: `pip install {model_name}` | `src/model_registry.py` | `get_model_config()` | L72–76 |
| 21 | `RuntimeError` | `register_model()` must be called inside an active MLflow run | `src/mlflow_logger.py` | `register_model()` | L435–438 |
| 22 | `FileNotFoundError` | Log file not found: `{log_path}` — Ensure the logger wrote to this path before calling `upload_pipeline_log_file()` | `src/mlflow_logger.py` | `upload_pipeline_log_file()` | L516–522 |

---

## Summary by Module

| Module | Errors Raised |
|--------|---------------|
| `src/data_loader.py` | `FileNotFoundError`, `KeyError`, `ValueError` |
| `src/preprocessor.py` | `FileNotFoundError`, `KeyError` (×2) |
| `src/splitter.py` | `AssertionError` |
| `src/pipeline_builder.py` | `KeyError`, `RuntimeError`, `ValueError` (×2) |
| `src/tuner.py` | `KeyError`, `ValueError` |
| `src/evaluation.py` | `ValueError` |
| `src/feature_importance.py` | `ValueError`, `AttributeError` |
| `src/model_saver.py` | `OSError` (×2) |
| `src/model_registry.py` | `KeyError`, `RuntimeError` |
| `src/mlflow_logger.py` | `RuntimeError`, `FileNotFoundError` |

---

## Notes

- Errors in `pipeline_builder.py` (rows 8–9) and `tuner.py` (row 12) are **re-raised** — they originate from `model_registry.py → get_model_config()`.
- `mlflow_logger.py → init_mlflow()` catches a generic `Exception` from `mlflow.get_experiment_by_name()` and handles it with a warning (no re-raise).
- `run_churn_pipeline.py` catches all exceptions at the top level and surfaces them via stderr and exit code `1`. It does not raise new error types itself.
- All error messages are simultaneously written to the pipeline `.log` file and tagged to the MLflow run for full traceability.
