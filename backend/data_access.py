import re
import time
import warnings
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

# Silence pandas fragmentation warnings and sklearn feature-name warnings
warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from config_limits import ML_THRESHOLDS
from dynamic_limits import calculate_dynamic_limits

# Robust project root detection
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WIDE_FILE = PROJECT_ROOT / "processed" / "features" / "rolling_features_demo.parquet"
WIDE_FILE_FALLBACK = PROJECT_ROOT / "processed" / "features" / "rolling_features_wide.parquet"
FEB_RESULTS_FILE = PROJECT_ROOT / "new_processed_data" / "FEB_TEST_RESULTS.parquet"
MACHINE_TESTS_DIR = PROJECT_ROOT / "new_processed_data"
CONTROL_MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_scrap_risk_wide_v2.pkl"
MODEL_FEATURES_PATH = PROJECT_ROOT / "processed" / "features" / "rolling_feature_columns.txt"
FORECASTER_MODEL_PATH = PROJECT_ROOT / "models" / "sensor_forecaster_lagged.pkl"
FUTURE_RISK_THRESHOLD = float(ML_THRESHOLDS.get("MEDIUM", 0.60))



# --------------- TTL Cache Infrastructure ---------------
_ttl_cache: dict = {}   # key -> (expiry_timestamp, result)
_TTL_SECONDS = 15

def _get_cached(key):
    """Return cached result if still fresh, else None."""
    entry = _ttl_cache.get(key)
    if entry and time.monotonic() < entry[0]:
        return entry[1]
    return None

def _set_cached(key, value):
    """Store a result with a 15-second TTL."""
    _ttl_cache[key] = (time.monotonic() + _TTL_SECONDS, value)


def _normalize_machine_id(machine_id: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9]", "", str(machine_id or "")).upper()
    if compact.startswith("M"):
        return compact
    return f"M{compact}"


def _display_machine_id(machine_norm: str) -> str:
    match = re.match(r"^M(\d+)$", machine_norm)
    if match:
        return f"M-{match.group(1)}"
    return machine_norm


def _safe_float(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _downsample(df: pd.DataFrame, max_points: int = 360) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    sampled = df.iloc[::step].copy()
    if sampled.iloc[-1]["timestamp"] != df.iloc[-1]["timestamp"]:
        sampled = pd.concat([sampled, df.tail(1)], ignore_index=True)
    return sampled.drop_duplicates(subset=["timestamp"], keep="last")


def _clean_limit_payload(current_safe_limits: dict):
    cleaned = {}
    for sensor, limits in current_safe_limits.items():
        cleaned[sensor] = {
            "min": _safe_float(limits.get("min")) if "min" in limits else None,
            "max": _safe_float(limits.get("max")) if "max" in limits else None,
        }
    return cleaned


@lru_cache(maxsize=1)
def _load_control_model_and_features():
    if not CONTROL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {CONTROL_MODEL_PATH}")

    model = joblib.load(CONTROL_MODEL_PATH)
    
    if hasattr(model, "feature_name_"):
        features = model.feature_name_
    elif hasattr(model, "booster_"):
        features = model.booster_.feature_name()
    else:
        with open(MODEL_FEATURES_PATH, "r") as f:
            features = [
                line.strip() for line in f.readlines() 
                if line.strip() and line.strip() not in ["machine_id_normalized", "event_timestamp", "timestamp", "machine_id", "is_scrap"]
            ]
            
    return model, tuple(features)


@lru_cache(maxsize=1)
def _load_sensor_forecaster():
    if not FORECASTER_MODEL_PATH.exists():
        raise FileNotFoundError(f"Sensor forecaster not found: {FORECASTER_MODEL_PATH}")
    artifact = joblib.load(FORECASTER_MODEL_PATH)
    return (
        artifact["model"],
        list(artifact["sensor_columns"]),
        list(artifact["input_features"]),
        int(artifact["num_lags"]),
        list(artifact.get("hydra_features", []))
    )


@lru_cache(maxsize=1)
def _load_feb_results():
    if not FEB_RESULTS_FILE.exists():
        raise FileNotFoundError(f"FEB results file not found: {FEB_RESULTS_FILE}")

    # Column-filtered load: only the columns we actually need
    _feb_cols = [
        "timestamp", "Injection_pressure", "Cycle_time",
        "scrap_probability", "is_scrap_actual",
    ]
    try:
        feb = pd.read_parquet(FEB_RESULTS_FILE, columns=_feb_cols, engine="pyarrow")
    except Exception:
        # Fallback: load all columns if filtering fails (schema drift)
        feb = pd.read_parquet(FEB_RESULTS_FILE, engine="pyarrow")

    if "timestamp" not in feb.columns:
        raise ValueError("FEB_TEST_RESULTS.parquet must include a 'timestamp' column.")

    feb["timestamp"] = pd.to_datetime(feb["timestamp"], utc=True, errors="coerce")
    feb = feb.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    for key_col in ("Injection_pressure", "Cycle_time"):
        if key_col in feb.columns:
            feb[key_col] = pd.to_numeric(feb[key_col], errors="coerce").round(4)

    return feb


@lru_cache(maxsize=16)
def _load_machine_pivot(machine_norm: str):
    machine_path = MACHINE_TESTS_DIR / f"{machine_norm}_TEST.parquet"
    if not machine_path.exists():
        raise FileNotFoundError(f"Machine test parquet not found: {machine_path}")

    raw = pd.read_parquet(machine_path, columns=["timestamp", "variable_name", "value", "machine_definition"], engine="pyarrow")
    machine_definition = "UNKNOWN"
    defs = raw["machine_definition"].dropna().astype(str).unique()
    if len(defs) > 0:
        machine_definition = defs[0]

    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["value"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp"])

    pivot = raw.pivot_table(
        index="timestamp",
        columns="variable_name",
        values="value",
        aggfunc="mean",
    ).reset_index()
    pivot = pivot.sort_values("timestamp").reset_index(drop=True)

    _, model_features = _load_control_model_and_features()
    for feature in model_features:
        if feature not in pivot.columns:
            pivot[feature] = 0.0

    for key_col in ("Injection_pressure", "Cycle_time"):
        if key_col in pivot.columns:
            pivot[key_col] = pd.to_numeric(pivot[key_col], errors="coerce").round(4)

    return pivot, machine_definition


def _build_machine_feb_history(machine_norm: str):
    """Build machine history with 15-second TTL cache."""
    cache_key = ("feb_history", machine_norm)
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached
    feb = _load_feb_results()
    pivot, machine_definition = _load_machine_pivot(machine_norm)

    join_cols = ["timestamp", "Injection_pressure", "Cycle_time"]
    missing_join = [c for c in join_cols if c not in pivot.columns or c not in feb.columns]
    if missing_join:
        raise ValueError(f"Cannot map machine rows to FEB results. Missing join columns: {missing_join}")

    feb_unique = feb.drop_duplicates(subset=join_cols, keep="first")
    # Merge using the FULL pivot so that all sensor columns are carried through
    history = pivot.merge(feb_unique, on=join_cols, how="left")

    if history.empty:
        raise ValueError(f"No FEB history matched machine {machine_norm}.")

    if "scrap_probability" not in history.columns:
        history["scrap_probability"] = 0.0
    history["scrap_probability"] = pd.to_numeric(history["scrap_probability"], errors="coerce")

    if "is_scrap_actual" not in history.columns:
        history["is_scrap_actual"] = 0
    history["is_scrap_actual"] = pd.to_numeric(history["is_scrap_actual"], errors="coerce").fillna(0)

    missing_prob_mask = history["scrap_probability"].isna()
    if missing_prob_mask.any():
        model, model_features = _load_control_model_and_features()
        feature_frame = history.loc[missing_prob_mask].copy()
        for feature in model_features:
            if feature not in feature_frame.columns:
                feature_frame[feature] = 0.0
        X_missing = feature_frame[list(model_features)].fillna(0.0)
        if hasattr(model, "predict_proba"):
            missing_probs = model.predict_proba(X_missing)[:, 1]
        else:
            missing_probs = model.predict(X_missing)
        history.loc[missing_prob_mask, "scrap_probability"] = missing_probs

    history["scrap_probability"] = history["scrap_probability"].fillna(0.0).clip(0, 1)

    # --- IDLE MACHINE FILTER: Override reality ---
    # If the cycle time is excessively low (<0.5s), the machine is essentially idle/offline. 
    # It cannot produce parts, therefore it cannot produce scrap. 
    if "Cycle_time" in history.columns:
        idle_mask = history["Cycle_time"].fillna(0) < 0.5
        history.loc[idle_mask, "scrap_probability"] = 0.0
        history.loc[idle_mask, "is_scrap_actual"] = 0

    history["machine_id_normalized"] = machine_norm
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
    history = history.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    tool_match = re.search(r"-([A-Za-z0-9]+)$", str(machine_definition))
    tool_id = tool_match.group(1) if tool_match else "UNKNOWN"

    machine_info = {
        "id": _display_machine_id(machine_norm),
        "tool_id": tool_id,
        "part_number": "UNKNOWN",
    }
    result = (history, machine_info)
    _set_cached(cache_key, result)
    return result


def _compute_root_causes(current_sensors: dict, current_safe_limits: dict):
    exceeded = []
    nearby = []
    for sensor, limits in current_safe_limits.items():
        sensor_value = _safe_float(current_sensors.get(sensor))
        if sensor_value is None:
            continue

        lower = _safe_float(limits.get("min")) if "min" in limits else None
        upper = _safe_float(limits.get("max")) if "max" in limits else None
        span_candidates = []
        if lower is not None and upper is not None:
            span_candidates.append(abs(upper - lower))
        if upper is not None:
            span_candidates.append(abs(upper))
        if lower is not None:
            span_candidates.append(abs(lower))
        span = max(max(span_candidates) if span_candidates else 1.0, 1.0)

        if upper is not None and sensor_value > upper:
            breach_magnitude = (sensor_value - upper) / span
            if breach_magnitude >= 0.01:  # Filter out noise breaches (<1% overshoot)
                exceeded.append((sensor, breach_magnitude))
            continue
        if lower is not None and sensor_value < lower:
            breach_magnitude = (lower - sensor_value) / span
            if breach_magnitude >= 0.01:  # Filter out noise breaches (<1% undershoot)
                exceeded.append((sensor, breach_magnitude))
            continue

        distances = []
        if lower is not None:
            distances.append(abs(sensor_value - lower))
        if upper is not None:
            distances.append(abs(upper - sensor_value))
        if distances:
            normalized_margin = min(distances) / span
            nearby.append((sensor, 1.0 - min(normalized_margin, 1.0)))

    if exceeded:
        exceeded_sorted = sorted(exceeded, key=lambda item: item[1], reverse=True)
        return [sensor for sensor, _ in exceeded_sorted[:3]], [sensor for sensor, _ in exceeded_sorted]

    nearby_sorted = sorted(nearby, key=lambda item: item[1], reverse=True)
    return [sensor for sensor, _ in nearby_sorted[:3]], []


def _infer_step_seconds(history: pd.DataFrame) -> int:
    if len(history) < 2:
        return 60
    diffs = history["timestamp"].diff().dropna().dt.total_seconds()
    if diffs.empty:
        return 60
    median_step = float(diffs.median())
    if not np.isfinite(median_step) or median_step <= 0:
        return 60
    return int(np.clip(round(median_step), 10, 120))


def _generate_future_horizon(past_window: pd.DataFrame, future_minutes: int, current_safe_limits: dict = None):
    model, model_features = _load_control_model_and_features()
    model_features = list(model_features)
    
    if past_window.empty:
        return pd.DataFrame(columns=["timestamp", "scrap_probability", "is_scrap_actual"])

    recent = past_window.sort_values("timestamp").tail(min(240, len(past_window))).copy()
    
    for feature in model_features:
        if feature not in recent.columns:
            recent[feature] = 0.0
        recent[feature] = pd.to_numeric(recent[feature], errors="coerce").ffill().fillna(0.0)

    step_seconds = 60  # 1-minute resolution (temporal downsampling)
    steps = max(6, future_minutes)  # e.g. 35 min → 35 steps
    last_ts = recent["timestamp"].iloc[-1]
    
    future_timestamps = [last_ts + pd.Timedelta(minutes=i) for i in range(1, steps + 1)]
    future = pd.DataFrame({"timestamp": future_timestamps})

    try:
        forecaster_data = _load_sensor_forecaster()
        if forecaster_data:
            forecaster = forecaster_data[0] if isinstance(forecaster_data, tuple) else forecaster_data["model"]
            raw_sensors = forecaster_data[1] if isinstance(forecaster_data, tuple) else forecaster_data["sensor_columns"]
            input_features = forecaster_data[2] if isinstance(forecaster_data, tuple) else forecaster_data["input_features"]
            num_lags = forecaster_data[3] if isinstance(forecaster_data, tuple) else forecaster_data["num_lags"]
            hydra_features = forecaster_data[4] if isinstance(forecaster_data, tuple) and len(forecaster_data) > 4 else forecaster_data.get("hydra_features", [])
            
            history_needed = num_lags + 1
            if len(recent) < history_needed:
                pad_df = pd.DataFrame([recent.iloc[0]] * (history_needed - len(recent)))
                history_df = pd.concat([pad_df, recent], ignore_index=True)
            else:
                history_df = recent.tail(history_needed).copy()
                
            # Bring any available hydra features into the recent window if they aren't there explicitly
            for h in hydra_features:
                if h not in history_df.columns:
                    history_df[h] = 0.0
                
            # Validate sensor columns are present in the data
            missing_sensors = [s for s in raw_sensors if s not in recent.columns]
            if missing_sensors:
                print(f"[DEBUG] Forecaster sensors MISSING from past_window: {missing_sensors}")
                print(f"[DEBUG] Available columns: {sorted(recent.columns.tolist())}")
                raise KeyError(f"Forecaster sensors missing: {missing_sensors}")

            t_forecast = time.perf_counter()
            # Capture the last hydra feature values so we can carry them forward
            last_hydra = {h: history_df[h].iloc[-1] for h in hydra_features}

            state_buffer = history_df[raw_sensors].to_dict('records')
            future_raw_predictions = []

            # Pre-allocate numpy array for prediction input (avoids DataFrame overhead)
            n_features = len(input_features)
            X_step_arr = np.zeros((1, n_features))
            
            for _ in range(steps):
                current_input = {}
                # Add constant hydra context
                current_input.update(last_hydra)

                for s in raw_sensors:
                    current_input[s] = state_buffer[-1].get(s, 0.0)
                for lag in range(1, num_lags + 1):
                    lag_idx = -(lag + 1)
                    for s in raw_sensors:
                        current_input[f"{s}_lag_{lag}"] = state_buffer[lag_idx].get(s, 0.0)
                        
                # Fill pre-allocated array (fast, no object creation)
                for fi, f in enumerate(input_features):
                    X_step_arr[0, fi] = current_input.get(f, 0.0)
                pred_raw = forecaster.predict(X_step_arr)[0]
                
                next_state = {}
                for i, sensor in enumerate(raw_sensors):
                    final_val = float(pred_raw[i])
                    
                    # Clamp within context-aware safe limits
                    if current_safe_limits and sensor in current_safe_limits:
                        if "min" in current_safe_limits[sensor]:
                            final_val = max(final_val, _safe_float(current_safe_limits[sensor]["min"]) or final_val)
                        if "max" in current_safe_limits[sensor]:
                            final_val = min(final_val, _safe_float(current_safe_limits[sensor]["max"]) or final_val)
                            
                    next_state[sensor] = final_val
                    
                future_raw_predictions.append(next_state)
                state_buffer.pop(0)
                state_buffer.append(next_state)
                
            pred_df = pd.DataFrame(future_raw_predictions)
            for col in raw_sensors:
                future[col] = pred_df[col]
            print(f"[HEARTBEAT]   forecast_loop ({steps} steps): {time.perf_counter() - t_forecast:.3f}s")
        else:
            raise ValueError("Lagged forecaster model missing.")
            
    except Exception as e:
        print(f"Warning: Lagged forecaster bypassed ({e}). Falling back to EMA.")
        # EMA fallback: smoothly decay from last known value instead of a flat line
        ema_alpha = 0.05  # Small alpha = slow decay, keeps line close to last known
        fallback_keys = current_safe_limits.keys() if current_safe_limits else recent.select_dtypes(include=[np.number]).columns
        for col in fallback_keys:
            if col in recent.columns:
                last_val = float(recent.iloc[-1][col])
                col_mean = float(recent[col].mean())
                ema_values = []
                current_ema = last_val
                for _ in range(len(future)):
                    current_ema = ema_alpha * col_mean + (1 - ema_alpha) * current_ema
                    ema_values.append(round(current_ema, 6))
                future[col] = ema_values

    last_known_features = recent.iloc[-1]
    for feature in model_features:
        if feature not in future.columns:
            future[feature] = last_known_features.get(feature, 0.0)

    # ── Carry forward ALL safe_limits sensors into future (not just model features) ──
    # Without this, sensors the forecaster doesn't predict (e.g. Ejector_fix_deviation_torque)
    # are absent from future timeline points → charts for those sensors appear blank.
    if current_safe_limits:
        for sensor in current_safe_limits:
            if sensor not in future.columns:
                last_val = recent[sensor].dropna().iloc[-1] if sensor in recent.columns and recent[sensor].notna().any() else 0.0
                future[sensor] = float(last_val)

    safe_features = [f for f in model_features if f in future.columns and f not in ["timestamp", "machine_id", "machine_id_normalized", "event_timestamp", "is_scrap"]]
    X_future = future[safe_features].fillna(0.0)
    
    if hasattr(model, "predict_proba"):
        risk_values = model.predict_proba(X_future)[:, 1]
    else:
        risk_values = model.predict(X_future)

    future["scrap_probability"] = pd.Series(risk_values).clip(0, 1)

    # --- IDLE MACHINE FILTER (FUTURE) ---
    if "Cycle_time" in future.columns:
        future_idle_mask = future["Cycle_time"].fillna(0) < 0.5
        future.loc[future_idle_mask, "scrap_probability"] = 0.0

    future["is_scrap_actual"] = 0
    future["predicted_scrap"] = (future["scrap_probability"] >= FUTURE_RISK_THRESHOLD).astype(int)
    
    return future


def _row_to_timeline_point(row, is_future: bool, current_safe_limits: dict = None):
    sensors = {}
    sensor_keys = current_safe_limits.keys() if current_safe_limits else []
    for sensor in sensor_keys:
        if sensor in row and pd.notna(row[sensor]):
            sensors[sensor] = round(float(row[sensor]), 2)

    ts = pd.to_datetime(row["timestamp"])
    # Strip any residual tz-info so strftime never shifts the clock value
    if hasattr(ts, "tz") and ts.tz is not None:
        ts = ts.tz_localize(None)
    return {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "risk_score": round(float(row.get("scrap_probability", 0.0)), 2),
        "is_future": bool(is_future),
        "is_scrap_actual": int(float(row.get("is_scrap_actual", 0) or 0)),
        "sensors": sensors,
    }


def build_control_room_payload(machine_id: str, time_window: int = 240, future_window: int = 35):
    # Check payload-level TTL cache first
    payload_key = ("payload", machine_id, time_window, future_window)
    cached_payload = _get_cached(payload_key)
    if cached_payload is not None:
        print(f"[CACHE HIT] Returning saved payload for {machine_id} (0.00s)", flush=True)
        return cached_payload

    t0 = time.perf_counter()
    print(f"[CACHE MISS] Fetching fresh data from Parquet for {machine_id}...", flush=True)

    machine_norm = _normalize_machine_id(machine_id)

    t_io = time.perf_counter()
    history, machine_info = _build_machine_feb_history(machine_norm)
    print(f"[HEARTBEAT]   _build_machine_feb_history: {time.perf_counter() - t_io:.3f}s", flush=True)

    if history.empty:
        raise ValueError(f"No history found for machine {machine_id}")

    history = history.sort_values("timestamp").reset_index(drop=True)

    # ── Ensure timestamps are clean datetime ──
    history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")

    # ── AUTO-STEP-BACK LIVE MODE ──
    # Find the absolute end of the file and step back 1 hour
    # so the final hour of data can be cross-checked against the forecast
    max_time = history["timestamp"].max()
    anchor = max_time - pd.Timedelta(hours=1)

    # Cutoff uses the time_window (e.g., 120 min = 2 hours before anchor)
    cutoff = anchor - pd.Timedelta(minutes=time_window)

    past_window = history[
        (history["timestamp"] >= cutoff) & (history["timestamp"] <= anchor)
    ].copy()
    print(f"[HEARTBEAT]   max_time={max_time}  anchor={anchor}  cutoff={cutoff}  past_rows={len(past_window)}", flush=True)
    # --- SAFETY NET FOR EMPTY DATA ---
    if past_window.empty:
        print(f"[WARNING] No data found for {machine_id} window. Returning OFFLINE fallback.", flush=True)
        return {
            "machine_info": {"id": machine_id, "tool_id": "UNKNOWN", "part_number": "UNKNOWN"},
            "summary_stats": {"past_scrap_detected": 0, "future_scrap_predicted": 0},
            "current_health": {"status": "OFFLINE", "risk_score": 0.0, "root_causes": []},
            "timeline": [],
            "safe_limits": {}
        }
    if past_window.empty:
        past_window = history[history["timestamp"] <= anchor].tail(1).copy()

    # ── Forward-fill sensor data to handle sparse machines (M-612, M-607 etc.) ──
    numeric_cols = past_window.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        past_window[numeric_cols] = past_window[numeric_cols].ffill().bfill()

    # --- Dynamic Setpoint Engine: compute limits from recent data ---
    t_limits = time.perf_counter()
    current_safe_limits = calculate_dynamic_limits(past_window)
    print(f"[HEARTBEAT]   calculate_dynamic_limits: {time.perf_counter() - t_limits:.3f}s", flush=True)

    current_row = past_window.iloc[-1]
    current_sensors = {}
    for sensor in current_safe_limits:
        if sensor in current_row and pd.notna(current_row[sensor]):
            current_sensors[sensor] = float(current_row[sensor])

    root_causes, breached_sensors = _compute_root_causes(current_sensors, current_safe_limits)
    base_risk = float(current_row.get("scrap_probability", 0.0) or 0.0)
    risk_penalty = min(0.25, 0.06 * len(breached_sensors))
    current_risk = min(1.0, base_risk + risk_penalty)

    if breached_sensors or current_risk >= float(ML_THRESHOLDS.get("MEDIUM", 0.60)):
        status = "HIGH"
    elif current_risk >= float(ML_THRESHOLDS.get("LOW", 0.30)):
        status = "MEDIUM"
    else:
        status = "LOW"

    future_minutes = max(5, min(60, future_window))  # clamp to [5, 60]
    future_horizon = _generate_future_horizon(past_window, future_minutes=future_minutes, current_safe_limits=current_safe_limits)

    past_scrap_detected = int((past_window["is_scrap_actual"].fillna(0) >= 1).sum())
    future_scrap_predicted = int((future_horizon["scrap_probability"] >= FUTURE_RISK_THRESHOLD).sum())

    past_timeline = _downsample(past_window, max_points=320)
    future_timeline = _downsample(future_horizon, max_points=max(len(future_horizon), 210))

    timeline = []
    for _, row in past_timeline.iterrows():
        timeline.append(_row_to_timeline_point(row, is_future=False, current_safe_limits=current_safe_limits))
    for _, row in future_timeline.iterrows():
        timeline.append(_row_to_timeline_point(row, is_future=True, current_safe_limits=current_safe_limits))

    payload = {
        "machine_info": machine_info,
        "summary_stats": {
            "past_scrap_detected": past_scrap_detected,
            "future_scrap_predicted": future_scrap_predicted,
        },
        "current_health": {
            "status": status,
            "risk_score": round(current_risk, 2),
            "root_causes": root_causes or ["Injection_pressure"],
        },
        "timeline": timeline,
        "safe_limits": _clean_limit_payload(current_safe_limits),
    }
    print(f"[HEARTBEAT] build_control_room_payload END   | total: {time.perf_counter() - t0:.3f}s", flush=True)
    _set_cached(payload_key, payload)
    return payload


def get_recent_window(machine_id, minutes=60):
    """
    Synchronized fetch: Pulls the recent window from the same February history 
    timeline as build_control_room_payload, ensuring Section A and Section B match.
    """
    machine_norm = _normalize_machine_id(machine_id)
    history, _ = _build_machine_feb_history(machine_norm)
    
    if history.empty:
        return pd.DataFrame()

    history = history.sort_values("timestamp").reset_index(drop=True)
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
    history = history.dropna(subset=["timestamp"])

    if history.empty:
        return pd.DataFrame()

    max_time = history["timestamp"].max()
    anchor = max_time - pd.Timedelta(hours=1)
    
    cutoff = anchor - pd.Timedelta(minutes=minutes)

    past_window = history[
        (history["timestamp"] >= cutoff) & (history["timestamp"] <= anchor)
    ].copy()

    # Backwards compatibility for forecasting.py
    past_window["event_timestamp"] = past_window["timestamp"]
    
    return past_window