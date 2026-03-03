"""
Step 5.3b: Train LightGBM scrap classifier with class balancing.
Memory-efficient version for large datasets.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Lazy imports to save memory
def get_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WIDE_FILE = PROJECT_ROOT / "processed" / "features" / "rolling_features_wide_labeled.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_scrap_risk_wide_v2.pkl"

# Pattern feature suffixes (already exist in data)
PATTERN_SUFFIXES = ['__std_5m', '__std_15m', '__std_30m', '__mean_5m', '__mean_15m', '__mean_30m']

# Columns to exclude from features
EXCLUDE_COLS = [
    'is_scrap', 'is_scrap_actual', 'early_scrap_risk', 'scrap_probability',
    'timestamp', 'event_timestamp', 'Time', 'time',
    'machine_id', 'machine_id_normalized', 'machine_definition',
    'part_number', 'tool_number', 'tool_id'
]

# Maximum rows for training (to fit in memory)
MAX_TRAIN_ROWS = 500_000


def find_target_column(df: pd.DataFrame) -> str:
    """Find the target column in the dataframe."""
    candidates = ['early_scrap_risk', 'is_scrap', 'is_scrap_actual']
    for col in candidates:
        if col in df.columns:
            print(f"[INFO] Using target column: {col}")
            return col
    raise ValueError(f"No target column found. Available: {df.columns.tolist()[:20]}")


def downsample_stratified(df: pd.DataFrame, target_col: str, max_rows: int) -> pd.DataFrame:
    """Stratified downsampling to preserve class ratio."""
    if len(df) <= max_rows:
        return df
    
    print(f"[INFO] Downsampling from {len(df):,} to {max_rows:,} rows (stratified)")
    
    # Separate classes
    df_pos = df[df[target_col] == 1]
    df_neg = df[df[target_col] == 0]
    
    # Calculate sampling ratio
    ratio = max_rows / len(df)
    n_pos = max(int(len(df_pos) * ratio), min(len(df_pos), 1000))  # At least 1000 positive samples
    n_neg = max_rows - n_pos
    
    # Sample each class
    df_pos_sample = df_pos.sample(n=min(n_pos, len(df_pos)), random_state=42)
    df_neg_sample = df_neg.sample(n=min(n_neg, len(df_neg)), random_state=42)
    
    result = pd.concat([df_pos_sample, df_neg_sample], ignore_index=True).sample(frac=1, random_state=42)
    
    print(f"[INFO] Sampled: {len(df_pos_sample):,} scrap + {len(df_neg_sample):,} non-scrap = {len(result):,} total")
    
    return result


def main():
    print("=" * 60)
    print("TRAINING LIGHTGBM SCRAP CLASSIFIER (Memory-Efficient)")
    print("=" * 60)
    
    # Step 1: Load data with memory optimization
    print(f"\n[INFO] Loading {WIDE_FILE}...")
    df = pd.read_parquet(WIDE_FILE, engine="pyarrow")
    print(f"[INFO] Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Step 2: Find target column
    target_col = find_target_column(df)
    
    # Step 3: Convert to float32 to save memory
    print("[INFO] Converting to float32 to save memory...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col != target_col:
            df[col] = df[col].astype(np.float32)
    
    # Step 4: Define feature columns
    feature_columns = [
        c for c in df.columns 
        if c not in EXCLUDE_COLS 
        and c != target_col
        and df[c].dtype in [np.float32, np.float64, np.int64, np.int32]
    ]
    print(f"[INFO] Using {len(feature_columns)} features")
    
    # Count pattern features
    n_pattern = len([c for c in feature_columns if any(s in c for s in PATTERN_SUFFIXES)])
    print(f"[INFO] Pattern-aware features: {n_pattern}")
    
    # Step 5: Downsample if needed
    df = downsample_stratified(df, target_col, MAX_TRAIN_ROWS)
    
    # Step 6: Prepare X and y
    print("[INFO] Preparing training data...")
    X = df[feature_columns].fillna(0).values.astype(np.float32)
    y = df[target_col].fillna(0).astype(int).values
    
    # Free memory
    del df
    import gc
    gc.collect()
    
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
    print(f"[INFO] Class distribution: {(y == 1).sum():,} scrap ({100 * (y == 1).mean():.2f}%) vs {(y == 0).sum():,} non-scrap")
    
    # Step 7: Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Step 8: Calculate class weight
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"[INFO] scale_pos_weight = {scale_pos_weight:.2f}")
    
    # Step 9: Train LightGBM
    print("\n[INFO] Training LightGBM...")
    import lightgbm as lgb
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_columns)
    valid_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_columns, reference=train_data)
    
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "num_leaves": 64,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "scale_pos_weight": scale_pos_weight,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # Step 10: Evaluate
    print("\n[INFO] Evaluating model...")
    from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
    
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    try:
        auc_score = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc_score = 0.0
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Non-Scrap", "Scrap"], zero_division=0))
    
    # Step 11: Save model
    print(f"\n[INFO] Saving model to {MODEL_PATH}...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    
    # Step 12: Feature importance
    print("[INFO] Generating feature importance...")
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    importance_file = PROJECT_ROOT / "models" / "feature_importance.csv"
    importance_df.to_csv(importance_file, index=False)
    
    # Plot
    plt = get_matplotlib()
    plt.figure(figsize=(12, 10))
    top_30 = importance_df.head(30)
    plt.barh(range(len(top_30)), top_30['importance'].values)
    plt.yticks(range(len(top_30)), top_30['feature'].values, fontsize=8)
    plt.xlabel('Feature Importance (Gain)')
    plt.title('Top 30 Features - Pattern-Aware Scrap Classifier')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_file = PROJECT_ROOT / "models" / "feature_importance.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    
    # Step 13: Save metrics
    metrics = {
        "accuracy": float((y_pred == y_test).mean()),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "auc": float(auc_score),
        "scale_pos_weight": float(scale_pos_weight),
        "n_features": len(feature_columns),
        "n_pattern_features": n_pattern,
        "n_train_samples": len(y_train),
        "n_test_samples": len(y_test),
        "n_train_scrap": int((y_train == 1).sum()),
        "n_test_scrap": int((y_test == 1).sum()),
    }
    
    metrics_file = PROJECT_ROOT / "models" / "training_metrics_v2.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Feature importance: {importance_file}")
    print(f"Metrics: {metrics_file}")
    print(f"\nKey Results:")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
