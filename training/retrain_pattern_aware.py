"""
Convenience script to run the full pattern-aware retraining pipeline.

This script executes:
1. step4b_add_pattern_features.py - Adds temporal pattern features
2. step5_3b_train_lightgbm_wide.py - Trains the model with class balancing

Usage:
    cd "D:\te connectivity 3"
    python training/retrain_pattern_aware.py

Output:
    - processed/features/rolling_features_pattern_aware.parquet
    - models/lightgbm_scrap_risk_wide_v2.pkl
    - models/feature_importance.csv
    - models/feature_importance.png
    - models/training_metrics_v2.json
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_step(script_name: str, description: str) -> bool:
    """Run a training step and return success status."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*70}\n")
    
    script_path = PROJECT_ROOT / "training" / script_name
    
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"\n[ERROR] {script_name} failed with exit code {result.returncode}")
            print(f"[ERROR] Elapsed time: {elapsed:.1f}s")
            return False
        
        print(f"\n[SUCCESS] {description} completed in {elapsed:.1f}s")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Exception running {script_name}: {e}")
        return False


def verify_outputs() -> dict:
    """Check which output files were created."""
    outputs = {
        "model_v2": PROJECT_ROOT / "models" / "lightgbm_scrap_risk_wide_v2.pkl",
        "feature_importance_csv": PROJECT_ROOT / "models" / "feature_importance.csv",
        "feature_importance_png": PROJECT_ROOT / "models" / "feature_importance.png",
        "training_metrics": PROJECT_ROOT / "models" / "training_metrics_v2.json",
    }
    
    results = {}
    for name, path in outputs.items():
        results[name] = path.exists()
    
    return results


def print_summary(outputs: dict):
    """Print final summary of the retraining pipeline."""
    print("\n" + "="*70)
    print("PATTERN-AWARE RETRAINING PIPELINE COMPLETE")
    print("="*70)
    
    print("\nOutput Files:")
    for name, exists in outputs.items():
        status = "✅ Created" if exists else "❌ Missing"
        print(f"  {status} - {name}")
    
    if all(outputs.values()):
        print("\n" + "-"*70)
        print("SUCCESS! All outputs generated.")
        print("-"*70)
        print("\nNext steps:")
        print("1. Review models/feature_importance.png to see top predictors")
        print("2. Check models/training_metrics_v2.json for accuracy metrics")
        print("3. To use the new model, update backend/data_access.py line 26:")
        print('   CONTROL_MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_scrap_risk_wide_v2.pkl"')
        print("4. Restart FastAPI server to load the new model")
        print("\n[NOTE] The model leverages existing rolling features (e.g., __std_15m, __mean_30m)")
        print("[NOTE] These distinguish stable oscillations from true anomalies")
    else:
        print("\n[WARNING] Some outputs are missing. Check the logs above for errors.")
    
    print("="*70 + "\n")


def main():
    print("="*70)
    print("PATTERN-AWARE MODEL RETRAINING PIPELINE")
    print("="*70)
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")
    print("\n[INFO] Skipping step4b — rolling features already exist in source data")
    print("[INFO] Features like Cycle_time__std_15m, Cushion__mean_30m already computed")
    
    total_start = time.time()
    
    # Skip Step 1 — pattern features already exist
    # Directly run Step 2 — train model
    step2_ok = run_step(
        "step5_3b_train_lightgbm_wide.py",
        "Training LightGBM with existing rolling features + class balancing"
    )
    
    if not step2_ok:
        print("\n[FATAL] Training failed. Check error messages above.")
        sys.exit(1)
    
    # Verify outputs
    outputs = verify_outputs()
    
    total_elapsed = time.time() - total_start
    print(f"\nTotal pipeline time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    
    # Print summary
    print_summary(outputs)
    
    # Exit with appropriate code
    if all(outputs.values()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
