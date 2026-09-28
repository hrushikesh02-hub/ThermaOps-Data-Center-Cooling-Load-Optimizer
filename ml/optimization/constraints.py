import sys
from pathlib import Path
import yaml


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# Decision Variable Bounds
BOUNDS = CONFIG["optimization"]["bounds"]
X_LOWER = [
    BOUNDS["cooling_setpoint_c"][0],
    BOUNDS["supply_air_temperature_c"][0],
    BOUNDS["fan_speed_pct"][0],
    BOUNDS["pump_speed_pct"][0]
]
X_UPPER = [
    BOUNDS["cooling_setpoint_c"][1],
    BOUNDS["supply_air_temperature_c"][1],
    BOUNDS["fan_speed_pct"][1],
    BOUNDS["pump_speed_pct"][1]
]

# Thermal Constraints
MAX_INLET = CONFIG["optimization"]["thermal_constraints"]["max_server_inlet_c"]
MAX_HOTSPOT = CONFIG["optimization"]["thermal_constraints"]["max_hotspot_c"]

# Penalty Weights
PENALTY_THERMAL = float(CONFIG["optimization"]["penalties"]["thermal_violation_weight"])
PENALTY_OOD = float(CONFIG["optimization"]["penalties"]["ood_violation_weight"])

def calculate_thermal_penalty(inlet_preds, hotspot_preds):
    """
    Vectorized calculation of thermal penalties.
    1e6 * ( max(0, predicted_server_inlet - 27)^2 + max(0, predicted_hotspot - 35)^2 )
    """
    import numpy as np
    
    inlet_violations = np.maximum(0, inlet_preds - MAX_INLET)
    hotspot_violations = np.maximum(0, hotspot_preds - MAX_HOTSPOT)
    
    penalty = PENALTY_THERMAL * (inlet_violations**2 + hotspot_violations**2)
    return penalty
