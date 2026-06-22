
"""
cleanup_mlflow.py — Utility to clear MLflow runs, artifacts, traces, and registry.

This tool deletes specified MLflow experiments and all associated data (runs, artifacts,
metrics, traces, and registered models) to allow for fresh testing/training cycles.

The cleanup performs:
    1. Deletes registered models from MLflow model registry
    2. Deletes experiments via MLflow API
    3. Physically removes experiment directories, run artifacts, and model outputs from filesystem
    4. Clears .trash directory and models registry folder

Usage:
    python cleanup_mlflow.py

The script will:
    1. Connect to MLflow tracking URI (configured in src/config.py)
    2. Search for registered models matching churn model patterns
    3. Delete registered models from the registry
    4. Search for experiments by name: "churn-prediction" and "MLflow Demo"
    5. Delete each matching experiment (all runs, artifacts, and registry models)
    6. Physically remove experiment directories and artifacts from filesystem
    7. Print confirmation of deleted experiments and their IDs

Warning:
    This operation is destructive and cannot be undone. The script prompts for
    confirmation before proceeding. All model outputs, plots, and run data will
    be permanently deleted.
"""

import mlflow
from mlflow.tracking import MlflowClient
import sys
import shutil
from pathlib import Path

# Add src directory to path for importing config
sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import MLFLOW_TRACKING_URI


def cleanup_mlflow(experiment_names=None, model_name_patterns=None):
    """
    Delete specified MLflow experiments and all associated data, including registered models.
    
    Parameters
    ----------
    experiment_names : list of str, optional
        Names of experiments to delete. Defaults to ["churn-prediction", "MLflow Demo"].
        If an experiment name doesn't exist, it is silently skipped.
    
    model_name_patterns : list of str, optional
        Substring patterns to match registered model names for deletion.
        Defaults to ["churn_binary_random_forest", "churn_multiclass_random_forest"].
        Models containing any of these patterns will be deleted.
    
    Returns
    -------
    bool
        True if cleanup completed successfully, False if any errors occurred.
    
    Examples
    --------
    >>> from cleanup_mlflow import cleanup_mlflow
    >>> cleanup_mlflow()  # Deletes experiments and churn models
    ✓ MLflow tracking URI: mlruns
    ✓ Found 3 total experiments in tracking backend
    Searching for experiments: ['churn-prediction', 'MLflow Demo']
    Found experiment 'churn-prediction' (ID: 585269539571180371)
    Found experiment 'MLflow Demo' (ID: 120350087844263486)
    Searching for registered models...
    Found registered model 'churn_binary_random_forest'
    Found registered model 'churn_multiclass_random_forest'
    Deleting registered models...
      ✓ Deleted 'churn_binary_random_forest'
      ✓ Deleted 'churn_multiclass_random_forest'
    Deleting experiment 'churn-prediction' (ID: 585269539571180371)...
    Deleting experiment 'MLflow Demo' (ID: 120350087844263486)...
    ✓ Successfully deleted 2 registered models and 2 experiments.
    True
    """
    if experiment_names is None:
        experiment_names = ["churn-prediction", "MLflow Demo"]
    
    if model_name_patterns is None:
        model_name_patterns = ["churn_binary_random_forest", "churn_multiclass_random_forest"]
    
    try:
        # Set tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"✓ MLflow tracking URI: {MLFLOW_TRACKING_URI}")
        
        # Initialize MLflow Client for registry operations
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        
        # Search for all experiments
        all_experiments = mlflow.search_experiments()
        print(f"✓ Found {len(all_experiments)} total experiments in tracking backend")
        
        # Find experiments by name and collect their IDs
        experiments_to_delete = []
        print(f"\nSearching for experiments: {experiment_names}")
        
        for exp in all_experiments:
            if exp.name in experiment_names:
                experiments_to_delete.append((exp.name, exp.experiment_id))
                print(f"  Found experiment '{exp.name}' (ID: {exp.experiment_id})")
        
        # Search for and delete registered models
        registered_models_deleted = 0
        try:
            print(f"\nSearching for registered models...")
            all_models = client.search_registered_models()
            models_to_delete = []
            
            for model in all_models:
                for pattern in model_name_patterns:
                    if pattern in model.name:
                        models_to_delete.append(model.name)
                        print(f"  Found registered model '{model.name}'")
                        break
            
            if models_to_delete:
                print(f"\nDeleting registered models...")
                for model_name in models_to_delete:
                    try:
                        client.delete_registered_model(model_name)
                        print(f"  ✓ Deleted '{model_name}'")
                        registered_models_deleted += 1
                    except Exception as e:
                        print(f"  ✗ Error deleting '{model_name}': {e}")
                        return False
        except Exception as e:
            print(f"⚠ Error searching/deleting registered models: {e}")
            # Continue with experiment deletion even if model deletion fails
        
        if not experiments_to_delete:
            print(f"\n⚠ No experiments found matching: {experiment_names}")
            if registered_models_deleted == 0:
                return False
        
        # Delete each experiment
        if experiments_to_delete:
            print(f"\nDeleting {len(experiments_to_delete)} experiment(s)...")
            for exp_name, exp_id in experiments_to_delete:
                try:
                    mlflow.delete_experiment(exp_id)
                    print(f"  ✓ Deleted '{exp_name}' (ID: {exp_id})")
                except Exception as e:
                    print(f"  ✗ Error deleting '{exp_name}' (ID: {exp_id}): {e}")
                    return False
        
        # Physically delete experiment directories and artifacts from filesystem
        mlruns_path = Path(MLFLOW_TRACKING_URI)
        if mlruns_path.exists():
            print(f"\nCleaning up filesystem artifacts...")
            
            # Delete experiment directories
            for exp_name, exp_id in experiments_to_delete:
                exp_dir = mlruns_path / exp_id
                if exp_dir.exists():
                    try:
                        shutil.rmtree(exp_dir)
                        print(f"  ✓ Removed directory: {exp_id}/")
                    except Exception as e:
                        print(f"  ⚠ Error removing directory {exp_id}/: {e}")
            
            # Clear .trash directory but keep it (MLflow requires it to exist)
            trash_dir = mlruns_path / ".trash"
            if trash_dir.exists():
                try:
                    # Remove all contents inside .trash but keep the directory
                    for item in trash_dir.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                    print(f"  ✓ Cleared .trash directory")
                except Exception as e:
                    print(f"  ⚠ Error clearing .trash: {e}")
            else:
                # Ensure .trash exists even if it was deleted
                try:
                    trash_dir.mkdir(exist_ok=True)
                    print(f"  ✓ Recreated .trash directory")
                except Exception as e:
                    print(f"  ⚠ Error creating .trash: {e}")
            
            # Clear models directory if we deleted registered models
            if registered_models_deleted > 0:
                models_dir = mlruns_path / "models"
                if models_dir.exists():
                    try:
                        shutil.rmtree(models_dir)
                        models_dir.mkdir()  # Recreate empty directory
                        print(f"  ✓ Cleared models directory")
                    except Exception as e:
                        print(f"  ⚠ Error clearing models directory: {e}")
        
        total_deleted = registered_models_deleted + len(experiments_to_delete)
        print(f"\n✓ Successfully deleted {registered_models_deleted} registered model(s) and {len(experiments_to_delete)} experiment(s)")
        return True
    
    except Exception as e:
        print(f"✗ Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
        return False


def confirm_cleanup():
    """
    Prompt user for confirmation before running cleanup.
    
    Returns
    -------
    bool
        True if user confirms (y/yes), False otherwise.
    """
    print("\n" + "=" * 70)
    print("MLflow Cleanup Tool")
    print("=" * 70)
    print("\nWARNING: This will delete the following experiments and registered models:")
    print("  Experiments:")
    print("    • churn-prediction")
    print("    • MLflow Demo")
    print("\n  Associated Registered Models:")
    print("    • churn_binary_random_forest")
    print("    • churn_multiclass_random_forest")
    print("\nThis includes:")
    print("  • All runs and artifacts (model outputs, plots, reports)")
    print("  • All traces and metrics")
    print("  • All registered model versions")
    print("  • All experiment directories and files from filesystem")
    print("\nThis operation CANNOT be undone.")
    print("=" * 70)
    
    response = input("\nContinue with cleanup? (y/n): ").strip().lower()
    return response in ["y", "yes"]


if __name__ == "__main__":
    if confirm_cleanup():
        print("\nStarting cleanup...\n")
        success = cleanup_mlflow()
        
        if success:
            print("\n✓ Cleanup completed successfully!")
            print("  MLflow is ready for fresh training runs.")
            sys.exit(0)
        else:
            print("\n✗ Cleanup encountered errors. Please review the output above.")
            sys.exit(1)
    else:
        print("\n✗ Cleanup cancelled.")
        sys.exit(0)
