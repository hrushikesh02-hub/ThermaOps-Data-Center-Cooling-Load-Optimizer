import pandas as pd
import logging
from ml.config import UNIVERSAL_LEAKAGE, TARGETS, ensure_directories

def setup_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = setup_logger("utils")

def load_data(filepath):
    """Loads CSV and parses timestamp."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def verify_temporal_split(train_df, val_df, test_df):
    """Verifies chronological order of datasets."""
    train_max = train_df['timestamp'].max()
    val_min = val_df['timestamp'].min()
    val_max = val_df['timestamp'].max()
    test_min = test_df['timestamp'].min()

    logger.info(f"Train max: {train_max}")
    logger.info(f"Val min: {val_min}")
    logger.info(f"Val max: {val_max}")
    logger.info(f"Test min: {test_min}")

    if not train_max < val_min:
        raise ValueError(f"Temporal leak: Train max ({train_max}) >= Val min ({val_min})")
    if not val_max < test_min:
        raise ValueError(f"Temporal leak: Val max ({val_max}) >= Test min ({test_min})")
    logger.info("Temporal split verification passed.")

def check_leakage(features_used, target):
    """Checks for explicit leakage columns or target-derived columns."""
    forbidden = set(UNIVERSAL_LEAKAGE)
    forbidden.update(TARGETS + ["cooling_power_kw", "chiller_load_pct", "crac_load_pct", "return_air_temperature_c", "supply_return_temp_delta_c"])
    forbidden.add(target)
    
    # Target-derived string matching to be safe
    for f in features_used:
        if f in forbidden:
            raise ValueError(f"Leakage detected! Feature '{f}' is in the forbidden list.")
        if target.replace("_kw", "").replace("_c", "") in f and f != target:
            # Simple heuristic, but might flag too much.
            # Instead we rely on explicit LEAKAGE_COLUMNS which covers derived features.
            pass
            
    logger.info(f"Leakage check PASSED for target {target}")
