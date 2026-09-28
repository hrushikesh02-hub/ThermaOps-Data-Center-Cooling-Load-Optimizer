import sys
import pandas as pd
from pathlib import Path
import joblib

from ml.config import MODELS_DIR, CATEGORICAL_FEATURES, NUMERICAL_BASE_FEATURES, CONTROL_VARIABLES

def calculate_baseline(scenario_df):
    """
    Calculates the baseline cooling power by running the exact original scenario row
    through the frozen cooling power surrogate model.
    """
    from ml.optimization.fitness import get_surrogate_models
    model = get_surrogate_models()[0]
    
    allowed_features = CATEGORICAL_FEATURES + NUMERICAL_BASE_FEATURES + CONTROL_VARIABLES
    X = scenario_df[allowed_features]
    
    baseline_power = model.predict(X)
    
    # Return as series/array
    return baseline_power
