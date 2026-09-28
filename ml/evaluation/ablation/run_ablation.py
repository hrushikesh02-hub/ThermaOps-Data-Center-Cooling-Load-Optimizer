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
logger = logging.getLogger("ablation")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

import warnings


def evaluate_baseline(scenario_row):
    """Evaluate current unmodified variables."""
    problem = CoolingOptimizationProblem(scenario_row, apply_thermal=True, apply_ood=True)
    X_baseline = scenario_row[['cooling_setpoint_c', 'supply_air_temperature_c', 'fan_speed_pct', 'pump_speed_pct']].values
    
    out = {}
    problem._evaluate(X_baseline, out)
    
    return {
        'power': out['power'][0],
        'thermal_penalty': out['thermal_penalty'][0],
        'ood_penalty': out['ood_penalty'][0],
        'inlet': out['inlet'][0],
        'hotspot': out['hotspot'][0]
    }

def run_ablation():
    from ml.config import ensure_directories
    ensure_directories()
    logger.info("Starting Ablation Study...")
    
    scenarios = get_scenarios()
    
    results = []
    
    for name, row in scenarios.items():
        logger.info(f"--- Scenario: {name} ---")
        
        # 1. Baseline (No Optimization)
        base_res = evaluate_baseline(row)
        baseline_power = base_res['power']
        
        # 2. Full System (Thermal=True, OOD=True)
        res_full = run_ga_optimization(row, apply_thermal=True, apply_ood=True, seed=42)
        t_full = res_full['runtime_s']
        
        # 3. Without OOD (Thermal=True, OOD=False)
        res_no_ood = run_ga_optimization(row, apply_thermal=True, apply_ood=False, seed=42)
        t_no_ood = res_no_ood['runtime_s']
        
        # 4. Without Thermal Constraints (Thermal=False, OOD=True)
        res_no_therm = run_ga_optimization(row, apply_thermal=False, apply_ood=True, seed=42)
        t_no_therm = res_no_therm['runtime_s']
        
        # Helper to construct row
        def format_res(exp_name, res, runtime, base_power):
            pwr = res['power']
            return {
                'Scenario': name,
                'Experiment': exp_name,
                'Baseline_Power': base_power,
                'Optimized_Power': pwr,
                'Savings_Pct': (base_power - pwr) / base_power * 100 if base_power > 0 else 0.0,
                'Thermal_Violations': 1 if res['thermal_penalty'] > 0 else 0,
                'OOD_Violations': 1 if res['ood_penalty'] > 0 else 0,
                'Server_Inlet': res['inlet'],
                'Hotspot': res['hotspot'],
                'Runtime_s': runtime,
                'Feasible': res['feasible']
            }
        
        results.append({
            'Scenario': name,
            'Experiment': 'No Optimization',
            'Baseline_Power': baseline_power,
            'Optimized_Power': baseline_power,
            'Savings_Pct': 0.0,
            'Thermal_Violations': 1 if base_res['thermal_penalty'] > 0 else 0,
            'OOD_Violations': 1 if base_res['ood_penalty'] > 0 else 0,
            'Server_Inlet': base_res['inlet'],
            'Hotspot': base_res['hotspot'],
            'Feasible': base_res['thermal_penalty'] == 0 and base_res['ood_penalty'] == 0,
            'Runtime_s': 0.0
        })
        
        results.append(format_res('Full System', res_full, t_full, baseline_power))
        results.append(format_res('Without OOD', res_no_ood, t_no_ood, baseline_power))
        results.append(format_res('Without Thermal', res_no_therm, t_no_therm, baseline_power))
        
    df_results = pd.DataFrame(results)
    
    # Save CSV
    out_dir = (RESULTS_DIR / 'ablation')
    df_results.to_csv(out_dir / 'ablation_results.csv', index=False)
    
    logger.info("Saved ablation_results.csv")
    
    # Generate Plots
    generate_ablation_plots(df_results, out_dir / 'plots')

def generate_ablation_plots(df, plot_dir):
    sns.set_theme(style="whitegrid")
    
    # Filter out 'No Optimization' for savings plot (it's always 0)
    df_plot = df[df['Experiment'] != 'No Optimization']
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x='Scenario', y='Savings_Pct', hue='Experiment')
    plt.title('Ablation candidate power change (includes infeasible diagnostics)')
    plt.ylabel('Power Savings (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plot_dir / 'ablation_savings.png')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df_plot, x='Server_Inlet', y='Hotspot', hue='Experiment', style='Scenario', s=150)
    plt.axvline(27.0, color='red', linestyle='--', label='Inlet Limit (27C)')
    plt.axhline(35.0, color='darkred', linestyle=':', label='Hotspot Limit (35C)')
    plt.title('Ablation Study: Thermal Safety Space')
    plt.xlabel('Server Inlet Temperature (C)')
    plt.ylabel('Hotspot Temperature (C)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(plot_dir / 'ablation_thermal_safety.png')
    plt.close()

if __name__ == "__main__":
    run_ablation()
