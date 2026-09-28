"""Deterministic dataset validation; never invent or alter measured targets."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from ml.config import DATA_DIR, ALL_FEATURES, TARGETS, RESULTS_DIR, RAW_DATA_DIR

TARGET_COLUMNS = TARGETS + ["cooling_power_kw"]
DERIVED_MARGINS = {"server_inlet_safety_margin_c": ("server_inlet_temperature_c", 27.0),
                   "hotspot_safety_margin_c": ("hotspot_temperature_c", 35.0)}

def validate_frame(df, name="dataset"):
    required = ["timestamp", *ALL_FEATURES, *TARGET_COLUMNS, *DERIVED_MARGINS]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")
    if df.empty or df.isna().any().any():
        raise ValueError(f"{name}: empty dataset or missing values")
    numeric = df.select_dtypes("number")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{name}: non-finite numeric values")
    dates = pd.to_datetime(df.timestamp, errors="raise")
    for col, expected in {"hour":dates.dt.hour,"minute":dates.dt.minute,"month":dates.dt.month,"day_of_month":dates.dt.day,"week_of_year":dates.dt.isocalendar().week,"day_of_week":dates.dt.day_name()}.items():
        if not (df[col].to_numpy()==expected.to_numpy()).all():
            raise ValueError(f"{name}: timestamp-derived column mismatch: {col}")
    if df.duplicated(["timestamp", "data_center_id"]).any():
        raise ValueError(f"{name}: duplicate timestamp/data-center keys")
    for col in df.columns:
        if col.endswith("_pct") and not df[col].between(0, 100).all():
            raise ValueError(f"{name}: percentage outside [0,100]: {col}")
    for col in ["it_power_kw", "airflow_cfm", "server_count", "cooling_power_kw", "cooling_load_kw"]:
        if not (df[col] > 0).all():
            raise ValueError(f"{name}: nonpositive {col}")
    derived = {
        "combined_compute_utilization_pct": .7*df.cpu_utilization_pct + .3*df.gpu_utilization_pct,
        "average_compute_memory_utilization_pct": (df.cpu_utilization_pct + df.gpu_utilization_pct + df.memory_utilization_pct)/3,
        "airflow_per_it_kw": df.airflow_cfm/df.it_power_kw,
        **{col: limit-df[target] for col,(target,limit) in DERIVED_MARGINS.items()},
        "inlet_temperature_within_target": df.server_inlet_temperature_c.between(18,27).astype(int),
    }
    for col, expected in derived.items():
        if not np.allclose(df[col], expected, rtol=1e-8, atol=1e-8):
            raise ValueError(f"{name}: inconsistent derived column {col}")
    return df

def audit_data(data_dir=DATA_DIR, raw_dir=RAW_DATA_DIR):
    frames = {name: validate_frame(pd.read_csv(data_dir/f"{name}.csv"), name)
              for name in ["train", "validation", "test"]}
    for name in ["train","validation","test"]:
        if not frames[name].timestamp.is_monotonic_increasing:
            raise ValueError(f"{name}: rows must be chronological")
    if not (frames["train"].timestamp.max() < frames["validation"].timestamp.min()
            and frames["validation"].timestamp.max() < frames["test"].timestamp.min()):
        raise ValueError("Temporal leakage between train, validation and test")
    combined = pd.concat([frames[k] for k in ["train","validation","test"]],ignore_index=True)
    validate_frame(combined, "combined")
    # Rebuild the full corrected mirror in memory instead of storing duplicate CSVs.
    full = pd.read_csv(raw_dir/"cooling_observations.csv")
    for column, (target, limit) in DERIVED_MARGINS.items():
        full[column] = limit-full[target]
    validate_frame(full, "raw observations with corrected margins")
    full = full.sort_values(["timestamp","data_center_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(combined, full, check_exact=False, rtol=1e-8, atol=1e-8)
    reference = pd.read_csv(raw_dir/"optimization_reference.csv")
    if reference.isna().any().any() or not np.isfinite(reference.select_dtypes("number").to_numpy()).all():
        raise ValueError("Invalid optimization reference data")
    if reference.duplicated(["timestamp","data_center_id"]).any():
        raise ValueError("Duplicate optimization reference keys")
    joined = combined.merge(reference,on=["timestamp","data_center_id"],suffixes=("", "_ref"),validate="one_to_one")
    if len(joined) != len(combined):
        raise ValueError("Optimization reference key mismatch")
    for col in ["cooling_setpoint_c","fan_speed_pct","pump_speed_pct","cooling_power_kw","cooling_load_kw"]:
        if not np.allclose(joined[col], joined[col+"_ref"]):
            raise ValueError(f"Optimization reference measurements differ: {col}")
    result = {"status":"passed", "total_rows":len(combined), "features":len(ALL_FEATURES),
              "data_centers":sorted(combined.data_center_id.unique().tolist()),
              "provenance":"Provided research dataset; source instrumentation and generation process were not supplied.",
              "mode":"historical_replay",
              "splits":{name:{"rows":len(df),"from":df.timestamp.min(),"to":df.timestamp.max(),
                        "sha256":hashlib.sha256((data_dir/f"{name}.csv").read_bytes()).hexdigest()}
                        for name,df in frames.items()},
              "unknown_test_categories":{c:sorted(set(frames["test"][c].astype(str))-set(frames["train"][c].astype(str)))
                        for c in ["data_center_id","season","time_of_day","day_of_week"]},
              "corrections":"Safety margins use 27 C inlet / 35 C hotspot. No measurement or target values changed.",
              "reference_rows":len(reference)}
    result["splits"]["research_ready_dataset"] = {
        "rows":len(full), "from":full.timestamp.min(), "to":full.timestamp.max(),
        "sha256":hashlib.sha256((raw_dir/"cooling_observations.csv").read_bytes()).hexdigest(),
        "source":"datasets/raw/cooling_observations.csv (margins corrected in memory)"}
    return result

if __name__ == "__main__":
    result = audit_data()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR/"data_quality.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

