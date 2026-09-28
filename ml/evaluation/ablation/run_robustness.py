from ml.config import RESULTS_DIR
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging


from ml.optimization.scenarios import get_scenarios
from ml.optimization.ga_optimizer import run_ga_optimization
from ml.optimization.fitness import CoolingOptimizationProblem

# Setup logger
logger = logging.getLogger("robustness")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

import warnings


SEEDS = [42, 7, 21, 100, 123]

def run_robustness():
    from ml.config import ensure_directories
    ensure_directories()
    logger.info("Starting Robustness Testing...")
    
    scenarios = get_scenarios()
    
    # Dynamically inject 2 synthetic scenarios using 'Medium Workload' as base
    base_row = scenarios['Medium Workload'].copy()
    
    # 1. High Outdoor Temp
    high_temp_row = base_row.copy()
    high_temp_row['outdoor_temperature_c'] = 35.0  # valid summer temp
    scenarios['Synthetic: High Outdoor Temp'] = high_temp_row
    
    # 2. High CPU Utilization
    high_cpu_row = base_row.copy()
    high_cpu_row['cpu_utilization_pct'] = 98.0
    high_cpu_row['combined_compute_utilization_pct'] = .7 * 98.0 + .3 * high_cpu_row['gpu_utilization_pct']
    high_cpu_row['average_compute_memory_utilization_pct'] = (98.0 + high_cpu_row['gpu_utilization_pct'] + high_cpu_row['memory_utilization_pct']) / 3
    scenarios['Synthetic: High CPU'] = high_cpu_row
    
    results = []
    
    for name, row in scenarios.items():
        logger.info(f"--- Scenario: {name} ---")
        
        # Eval baseline
        problem = CoolingOptimizationProblem(row, apply_thermal=True, apply_ood=True)
        X_baseline = row[['cooling_setpoint_c', 'supply_air_temperature_c', 'fan_speed_pct', 'pump_speed_pct']].values
        out_base = {}
        problem._evaluate(X_baseline, out_base)
        baseline_power = out_base['power'][0]
        
        for seed in SEEDS:
            res = run_ga_optimization(row, apply_thermal=True, apply_ood=True, seed=seed)
            runtime = res['runtime_s']
            hist = res['history']
            
            pwr = res['power']
            results.append({
                'Scenario': name,
                'Seed': seed,
                'Optimized_Power': pwr,
                'Savings_Pct': (baseline_power - pwr) / baseline_power * 100 if baseline_power > 0 else 0.0,
                'Server_Inlet': res['inlet'],
                'Hotspot': res['hotspot'],
                'Thermal_Penalty': res['thermal_penalty'],
                'OOD_Penalty': res['ood_penalty'],
                'Runtime_s': runtime,
                'Feasible': res['feasible'],
                'Objective_Evaluations': res['evaluations']
            })
            
    df_results = pd.DataFrame(results)
    
    out_dir = (RESULTS_DIR / 'ablation')
    df_results.to_csv(out_dir / 'robustness_results.csv', index=False)
    
    # Aggregate for scenario_robustness.csv
    df_scenario = df_results.groupby('Scenario').agg({
        'Optimized_Power': ['mean', 'std', 'min', 'max'],
        'Savings_Pct': ['mean', 'std', 'min', 'max'],
    })
    # Flatten multi-index columns
    df_scenario.columns = ['_'.join(col).strip() for col in df_scenario.columns.values]
    df_scenario.reset_index(inplace=True)
    df_scenario.to_csv(out_dir / 'scenario_robustness.csv', index=False)
    
    logger.info("Saved robustness_results.csv and scenario_robustness.csv")
    
    generate_robustness_plots(df_results, df_scenario, out_dir / 'plots')

def generate_robustness_plots(df_full, df_agg, plot_dir):
    sns.set_theme(style="whitegrid")
    
    # Robustness Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_full, x='Scenario', y='Optimized_Power', hue='Scenario')
    plt.title('Candidate power across seeds (includes infeasible diagnostics)')
    plt.ylabel('Optimized Power (kW)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plot_dir / 'robustness_boxplot.png')
    plt.close()
    
    # Scenario Savings
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_agg, x='Scenario', y='Savings_Pct_mean', color='steelblue')
    # Add error bars for std
    plt.errorbar(x=df_agg['Scenario'], y=df_agg['Savings_Pct_mean'], yerr=df_agg['Savings_Pct_std'], fmt='none', c='black', capsize=5)
    plt.title('Candidate power change across seeds (includes infeasible diagnostics)')
    plt.ylabel('Mean Power Savings (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plot_dir / 'scenario_savings.png')
    plt.close()

if __name__ == "__main__":
    run_robustness()
