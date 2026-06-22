# =============================================================================
# CHURN PREDICTION PIPELINE — OUTLINE
# =============================================================================
# Reflects final orchestrator notebook (churn_model_pipeline.ipynb)
# =============================================================================


# ── PROJECT STRUCTURE ─────────────────────────────────────────────────────────
#
#   churn_pipeline/
#   ├── src/
#   │   ├── config.py               
#   │   ├── model_registry.py       
#   │   ├── preprocessor.py         
#   │   ├── data_loader.py          
#   │   ├── splitter.py             
#   │   ├── pipeline_builder.py     
#   │   ├── tuner.py                
#   │   ├── evaluation.py           
#   │   ├── feature_importance.py   
#   │   └── model_saver.py          
#   ├── notebooks/
#   │   └── churn_model_pipeline.ipynb   
#   └── data/
#   └── model/



# =============================================================================
# ORCHESTRATOR FLOW  (churn_model_pipeline.ipynb)
# =============================================================================
#
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │  config.py                                                              │
#  │  cfg.MODEL = "random_forest" (default) | 'xgboost'                      │                                                   │
#  │  cfg.MODE = "binary" (default| "multiclass"                             │
#  │  force = True | False (default)                                         │         │
#  │                                                                         │
#  └───────────────────────────┬─────────────────────────────────────────────┘
#                              │
#  ┌───────────────────────────▼──────────────────────────────────────────────┐
#  │  PREPROCESSING STEP                                                      │
#  │  if (force and INPUT_PATH.exists()) :                                    │
#  │      run: load -> validate -> create_target -> impute -> save            │
#  └───────────────────────────────┬──────────────────────────────────────────┘
#                                  │
#  ┌───────────────────────────────▼──────────────────────────────────────────┐
#  │  MODEL PIPELINE (Single Mode: binary or multiclass)                      │
#  └───────────────────────────────┬──────────────────────────────────────────┘
#                                  │
#             load_data()          → X, y, cat_features
#             split_data(X, y, cfg.KEY_ID_COLS)  → X_train, X_val, X_test ...
#             verify_split()
#             summarise_target_distribution()
#             [drop KEY_ID_COLS]
#             build_pipeline(cat_features) → pipeline
#             build_search_set(…)          → X_search, y_search, pds
#             run_grid_search(…)           → grid_search
#             get_top_results(…)
#             evaluate(grid_search, …)     → metrics
#             get_feature_importance(…)    → feat_imp_df
#             save_model(grid_search, get_model_config(cfg.MODEL))
#
#
