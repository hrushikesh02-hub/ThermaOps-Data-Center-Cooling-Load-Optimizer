from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "datasets" / "raw"
DATA_DIR = PROJECT_ROOT / "datasets" / "processed"
MODELS_DIR = PROJECT_ROOT / "ml" / "models" / "artifacts"
PREPROCESSORS_DIR = PROJECT_ROOT / "outputs" / "preprocessors"
RESULTS_DIR = PROJECT_ROOT / "outputs"
FIG_EDA_DIR = RESULTS_DIR / "figures" / "eda"
FIG_EVAL_DIR = RESULTS_DIR / "figures" / "model_evaluation"
FIG_IMP_DIR = RESULTS_DIR / "figures" / "feature_importance"
TABLES_DIR = RESULTS_DIR / "tables"

# Targets
TARGETS = [
    "cooling_load_kw",
    "server_inlet_temperature_c",
    "hotspot_temperature_c"
]

# Excluded model inputs (targets, downstream outcomes and derived quantities)
UNIVERSAL_LEAKAGE = [
    "optimal_cooling_setpoint_c",
    "optimal_fan_speed_pct",
    "optimal_pump_speed_pct",
    "optimized_cooling_power_kw",
    "energy_savings_pct",
    "pue",
    "energy_consumption_kwh",
    "thermal_risk_score",
    "thermal_status",
    "cooling_to_it_power_ratio",
    "cooling_power_to_it_power_ratio",
    "hotspot_safety_margin_c",
    "server_inlet_safety_margin_c",
    "inlet_temperature_within_target"
]

# Available Safe Base Features
CATEGORICAL_FEATURES = [
    "data_center_id",
    "time_of_day",
    "day_of_week",
    "season",
    "peak_hour"
]

NUMERICAL_BASE_FEATURES = [
    "server_count",
    "cpu_utilization_pct",
    "gpu_utilization_pct",
    "memory_utilization_pct",
    "network_load_gbps",
    "storage_io_load_pct",
    "it_power_kw",
    "rack_density_kw",
    "indoor_temperature_c",
    "outdoor_temperature_c",
    "outdoor_humidity_pct",
    "indoor_humidity_pct",
    "airflow_cfm",
    "combined_compute_utilization_pct",
    "average_compute_memory_utilization_pct",
    "airflow_per_it_kw",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "hour",
    "minute",
    "month",
    "day_of_month",
    "week_of_year"
]

# Control variables - safe for thermal responses, maybe not for load if they are responses
CONTROL_VARIABLES = [
    "fan_speed_pct",
    "pump_speed_pct",
    "cooling_setpoint_c",
    "supply_air_temperature_c" 
]

# All four models use the same independent features. Targets and downstream
# measurements are excluded to prevent target leakage.
NUMERICAL_FEATURES = NUMERICAL_BASE_FEATURES + CONTROL_VARIABLES
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Model Settings
RANDOM_SEED = 42

# Hyperparameter search space
PARAM_GRIDS = {
    "RandomForest": {
        "n_estimators": [50, 100],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    },
    "XGBoost": {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
    },
    "LightGBM": {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
        "num_leaves": [31, 63]
    }
}


def ensure_directories():
    """Create output folders used by optional evaluation commands."""
    for path in [RESULTS_DIR, TABLES_DIR, FIG_EDA_DIR, FIG_EVAL_DIR, FIG_IMP_DIR,
                 RESULTS_DIR / "ablation" / "plots" / "convergence",
                 RESULTS_DIR / "optimization" / "convergence",
                 RESULTS_DIR / "optimization" / "plots"]:
        path.mkdir(parents=True, exist_ok=True)
