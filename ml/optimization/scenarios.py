import pandas as pd
from pathlib import Path
import sys

from ml.config import DATA_DIR, CATEGORICAL_FEATURES, NUMERICAL_BASE_FEATURES, CONTROL_VARIABLES

def get_scenarios():
    """
    Extracts 7 representative scenarios from the test set for validation.
    """
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    scenarios = {}
    
    # 1. Low Workload (approx 10th percentile of it_power_kw)
    p10 = test_df['it_power_kw'].quantile(0.10)
    scenarios['Low Workload'] = test_df.iloc[(test_df['it_power_kw'] - p10).abs().argsort()[:1]].copy()
    
    # 2. Medium Workload (approx 50th percentile)
    p50 = test_df['it_power_kw'].quantile(0.50)
    scenarios['Medium Workload'] = test_df.iloc[(test_df['it_power_kw'] - p50).abs().argsort()[:1]].copy()
    
    # 3. High Workload (approx 90th percentile)
    p90 = test_df['it_power_kw'].quantile(0.90)
    scenarios['High Workload'] = test_df.iloc[(test_df['it_power_kw'] - p90).abs().argsort()[:1]].copy()
    
    # 4. Uniform Workload (low variance across compute types)
    test_df['util_var'] = test_df[['cpu_utilization_pct', 'memory_utilization_pct', 'gpu_utilization_pct']].var(axis=1)
    idx_min = test_df['util_var'].idxmin()
    idx_max = test_df['util_var'].idxmax()
    scenarios['Uniform Workload'] = test_df.loc[[idx_min]].drop(columns=['util_var']).copy()
    
    # 5. Skewed Workload (high variance across compute types)
    scenarios['Skewed Workload'] = test_df.loc[[idx_max]].drop(columns=['util_var']).copy()
    test_df = test_df.drop(columns=['util_var'])
    
    # 6. Extreme Weather (highest outdoor temperature)
    scenarios['Extreme Weather'] = test_df.loc[test_df['outdoor_temperature_c'].idxmax():test_df['outdoor_temperature_c'].idxmax()].copy()
    
    # 7. Another DC (pick a different data_center_id from Medium Workload if possible)
    base_dc = scenarios['Medium Workload']['data_center_id'].iloc[0]
    other_dcs = test_df[test_df['data_center_id'] != base_dc]
    if not other_dcs.empty:
        scenarios['Alternate Data Center'] = other_dcs.iloc[0:1].copy()
    else:
        scenarios['Alternate Data Center'] = scenarios['Medium Workload'].copy()
        
    # Ensure they only retain the 34 features needed for prediction
    allowed_features = CATEGORICAL_FEATURES + NUMERICAL_BASE_FEATURES + CONTROL_VARIABLES
    
    # Keep original dataframe structure but just return the scenarios dict
    return scenarios
