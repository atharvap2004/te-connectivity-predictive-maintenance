import pandas as pd
import numpy as np
from pathlib import Path

# Load the physics rules from the CSV
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INFO_FILE = PROJECT_ROOT / "processed" / "safe" / "AI_cup_parameter_info_cleaned.csv"


def load_physics_rules():
    """Loads and cleans the tolerance rules from the CSV."""
    if not INFO_FILE.exists():
        print(f"Warning: Physics file not found at {INFO_FILE}. Using empty ruleset.")
        return {}

    df = pd.read_csv(INFO_FILE)
    rules = {}
    for _, row in df.iterrows():
        sensor = str(row['variable_name']).strip()

        # Safely extract tolerances, default to NaN if missing or 'NA'
        try:
            tol_plus = float(row['tolerance_plus']) if pd.notna(row['tolerance_plus']) else np.nan
            tol_minus = float(row['tolerance_minus']) if pd.notna(row['tolerance_minus']) else np.nan
        except ValueError:
            tol_plus, tol_minus = np.nan, np.nan

        rules[sensor] = {
            'plus': tol_plus,
            'minus': tol_minus,
            'remark': str(row.get('Remark', ''))
        }
    return rules


# Cache the rules globally so we don't read the CSV every second
PHYSICS_RULES = load_physics_rules()


def calculate_dynamic_limits(recent_history: pd.DataFrame) -> dict:
    """
    Strict 2-Bucket dynamic limits calibrated to the official TE Connectivity
    AI_cup_parameter_info CSV. No relational rules. No fabricated percentages.

    Bucket A: Hard CSV tolerances  →  local_median ± abs(tol)
    Bucket C: Statistical (3σ)     →  local_median ± max(3σ, 2% of median)
    """
    dynamic_limits = {}
    if recent_history is None or recent_history.empty:
        return dynamic_limits

    # ── Local context from the past_window ONLY ──
    local_median = recent_history.median(numeric_only=True)
    local_std = recent_history.std(numeric_only=True)

    # Exclude non-sensor / metadata columns
    NON_SENSOR_COLUMNS = {
        "Machine_status", "Scrap_counter", "Shot_counter", "Shot_size",
        "Time_on_machine", "Alrams_array", "scrap_probability",
        "is_scrap_actual", "predicted_scrap", "machine_id_normalized",
    }
    sensor_columns = [
        col for col in local_median.index
        if col in PHYSICS_RULES and col not in NON_SENSOR_COLUMNS
    ]

    for sensor in sensor_columns:
        setpoint = local_median[sensor]
        std_val = local_std[sensor]

        if pd.isna(setpoint):
            continue

        rule = PHYSICS_RULES[sensor]
        tol_plus = rule['plus']
        tol_minus = rule['minus']

        # --- BUCKET A: Hard Tolerances from CSV ---
        if pd.notna(tol_plus) and pd.notna(tol_minus):
            min_limit = setpoint - abs(tol_minus)
            max_limit = setpoint + abs(tol_plus)
        else:
            # --- BUCKET C: Statistical Process Control (3σ with 5% floor) ---
            sigma = std_val if not pd.isna(std_val) else 0.0
            safe_margin = max(sigma * 3, abs(setpoint) * 0.05)
            min_limit = setpoint - safe_margin
            max_limit = setpoint + safe_margin

        # ── Sanity Clamps ──
        # Clamp min to 0 for physically-positive quantities.
        # Allow negative for deviation, offset, and temperature (tmp) sensors.
        sensor_lower = sensor.lower()
        allow_negative = any(kw in sensor_lower for kw in ("deviation", "offset", "tmp"))
        if not allow_negative:
            min_limit = max(0, min_limit)

        dynamic_limits[sensor] = {
            "min": min_limit,
            "max": max_limit,
        }

    return dynamic_limits
