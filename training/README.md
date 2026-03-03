# Pattern-Aware Model Training Pipeline

This directory contains the enhanced training pipeline that addresses false positive scrap predictions by adding temporal pattern-aware features and class imbalance handling.

## Problem Solved

**Before**: The model couldn't distinguish normal rhythmic oscillations (e.g., Peak_pressure_time oscillating 0.200-0.210) from true anomalies, leading to false positive scrap predictions.

**After**: The model learns temporal patterns through rolling statistics, enabling it to recognize stable oscillations as normal behavior.

## Quick Start: Run Full Pipeline

```powershell
cd "d:\te connectivity 3"
.\.venv\Scripts\activate
python training/retrain_pattern_aware.py
```

This convenience script runs Steps 1 and 2 automatically, with progress tracking and error handling.

## Training Pipeline (Manual Steps)

### Step 1: Generate Pattern-Aware Features

```powershell
cd "d:\te connectivity 3"
.\.venv\Scripts\activate
python training/step4b_add_pattern_features.py
```

**What it does:**
- Loads `processed/features/rolling_features_wide_labeled.parquet`
- Adds 5 types of temporal features for 13 key sensors:
  - `_std_10`: Rolling standard deviation (stability measure)
  - `_trend_10`: Rolling linear trend (drift detection)
  - `_mean_10`: Rolling mean (local baseline)
  - `_dev_from_mean`: Deviation from local baseline
  - `_is_stable`: Binary stability flag
- Saves to `processed/features/rolling_features_pattern_aware.parquet`

**Expected output:**
- ~65 new pattern features added
- Total feature count increases by ~5x per sensor

### Step 2: Train Pattern-Aware Model

```powershell
python training/step5_3b_train_lightgbm_wide.py
```

**What it does:**
- Loads pattern-aware features (or falls back to original if not available)
- Calculates class imbalance weight (`scale_pos_weight`)
- Trains LightGBM with:
  - Class balancing to handle ~1.2% scrap rate
  - Regularization (L1=0.1, L2=0.1)
  - Early stopping (50 rounds)
- Exports:
  - Model: `models/lightgbm_scrap_risk_wide_v2.pkl`
  - Feature importance: `models/feature_importance.csv`
  - Top 30 feature plot: `models/feature_importance.png`
  - Training metrics: `models/training_metrics_v2.json`

**Expected output:**
```
Class distribution: 15,234 scrap (1.2%) vs 1,234,567 non-scrap
Using scale_pos_weight = 81.02
Pattern-aware features in top 30: 8-12

Key metrics:
  - AUC: 0.92+
  - F1 Score: 0.45-0.65
  - Precision: 0.70-0.85
  - Recall: 0.35-0.55
```

## Key Improvements

### 1. Temporal Pattern Recognition
- `Peak_pressure_time_std_10 < 0.005` → stable rhythmic pattern → SAFE
- `Injection_pressure_trend_10 > 0.1` → increasing pressure drift → RISK

### 2. Class Imbalance Handling
- `scale_pos_weight = n_negative / n_positive` (~81x weight on scrap samples)
- Forces model to not ignore rare scrap events

### 3. Feature Importance Transparency
- Exports which sensors contribute most to predictions
- Identifies pattern features in top predictors
- Enables sensor pruning if needed

## Convenience Script Features

**[retrain_pattern_aware.py](retrain_pattern_aware.py)** provides:
- ✅ Sequential execution of Steps 1 and 2
- ✅ Real-time progress updates with elapsed time
- ✅ Automatic error handling and early exit on failure
- ✅ Output file verification (checks if all files were created)
- ✅ Next-steps guidance printed at completion
- ✅ Exit codes (0 = success, 1 = failure)

Example output:
```
======================================================================
PATTERN-AWARE MODEL RETRAINING PIPELINE
======================================================================
Project root: D:\te connectivity 3
Python: D:\te connectivity 3\.venv\Scripts\python.exe

======================================================================
STEP: Adding temporal pattern-aware features (rolling std, trend, stability)
Script: step4b_add_pattern_features.py
======================================================================
[INFO] Loading processed/features/rolling_features_wide_labeled.parquet...
[INFO] Loaded 1,234,567 rows, 378 columns.
[INFO] Adding pattern-aware features to 1,234,567 rows...
...
[SUCCESS] Step 1 completed in 45.2s

======================================================================
STEP: Training LightGBM with pattern features + class balancing
Script: step5_3b_train_lightgbm_wide.py
======================================================================
Class distribution: 15,234 scrap (1.2%) vs 1,234,567 non-scrap
Using scale_pos_weight = 81.02
...
[SUCCESS] Step 2 completed in 182.3s

Total pipeline time: 227.5s (3.8 minutes)

======================================================================
PATTERN-AWARE RETRAINING PIPELINE COMPLETE
======================================================================
Output Files:
  ✅ Created - pattern_features
  ✅ Created - model_v2
  ✅ Created - feature_importance_csv
  ✅ Created - feature_importance_png
  ✅ Created - training_metrics

----------------------------------------------------------------------
SUCCESS! All outputs generated.
----------------------------------------------------------------------
Next steps:
1. Review models/feature_importance.png to see top predictors
2. Check models/training_metrics_v2.json for accuracy metrics
3. To use the new model, update backend/data_access.py line 26:
   CONTROL_MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_scrap_risk_wide_v2.pkl"
4. Restart FastAPI server to load the new model
======================================================================
```

## Files Created

```
training/
  ├── step4b_add_pattern_features.py  # Feature engineering
  ├── step5_3b_train_lightgbm_wide.py # Model training
  ├── retrain_pattern_aware.py        # Convenience pipeline runner
  └── README.md                        # This file

models/
  ├── lightgbm_scrap_risk_wide_v2.pkl      # New model (v2)
  ├── feature_importance.csv                # Feature rankings
  ├── feature_importance.png                # Top 30 plot
  └── training_metrics_v2.json              # Performance metrics

processed/features/
  └── rolling_features_pattern_aware.parquet  # Enhanced features
```

## Integration with Backend (Future)

To use the new model in production:

1. Update `backend/ml_inference.py`:
   ```python
   MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_scrap_risk_wide_v2.pkl"
   ```

2. Ensure `data_access.py` includes pattern features when generating predictions

3. Test on validation data before deploying

## Notes

- Original model (`lightgbm_scrap_risk_wide.pkl`) remains intact
- Both models can coexist for A/B testing
- Pattern features require time-series continuity (10+ cycles for rolling windows)
- Fallback to original features if pattern-aware file doesn't exist
