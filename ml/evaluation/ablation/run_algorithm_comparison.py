from ml.config import RESULTS_DIR
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging


from ml.optimization.scenarios import get_scenarios
from ml.optimization.ga_optimizer import run_ga_optimization
from ml.optimization.de_optimizer import run_de_optimization
from ml.optimization.fitness import CoolingOptimizationProblem

# Setup logger
logger = logging.getLogger("algo_comparison")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

import warnings


def evaluate_baseline(scenario_row):
    problem = CoolingOptimizationProblem(scenario_row, apply_thermal=True, apply_ood=True)
    X_baseline = scenario_row[['cooling_setpoint_c', 'supply_air_temperature_c', 'fan_speed_pct', 'pump_speed_pct']].values
    out = {}
    problem._evaluate(X_baseline, out)
    return out['power'][0]

def run_comparison():
    from ml.config import ensure_directories
    ensure_directories()
    logger.info("Starting GA vs DE Comparison...")
    scenarios = get_scenarios()
    
    results = []
    
    for name, row in scenarios.items():
        logger.info(f"--- Scenario: {name} ---")
        baseline_power = evaluate_baseline(row)
        
        # GA
        res_ga = run_ga_optimization(row, apply_thermal=True, apply_ood=True, seed=42)
        t_ga = res_ga['runtime_s']
        hist_ga = res_ga['history']
        
        # DE
        res_de = run_de_optimization(row, apply_thermal=True, apply_ood=True, seed=42)
        t_de = res_de['runtime_s']
        hist_de = res_de['history']
        
        def format_res(algo, res, runtime, hist):
            pwr = res['power']
            return {
                'Scenario': name,
                'Algorithm': algo,
                'Baseline_Power': baseline_power,
                'Optimized_Power': pwr,
                'Power_Reduction_Pct': (baseline_power - pwr) / baseline_power * 100 if baseline_power > 0 else 0.0,
                'Thermal_Violation_Rate': 1 if res['thermal_penalty'] > 0 else 0,
                'OOD_Violation_Rate': 1 if res['ood_penalty'] > 0 else 0,
                'Runtime_s': runtime,
                'Feasible': res['feasible'],
                'Objective_Evaluations': res['evaluations']
            }
        
        results.append(format_res('GA', res_ga, t_ga, hist_ga))
        results.append(format_res('DE', res_de, t_de, hist_de))
        
        # Plot convergence for this scenario
        plot_convergence(hist_ga, hist_de, name)
        
    df_results = pd.DataFrame(results)
    
    out_dir = (RESULTS_DIR / 'ablation')
    df_results.to_csv(out_dir / 'ga_vs_de_comparison.csv', index=False)
    logger.info("Saved ga_vs_de_comparison.csv")
    
    generate_comparison_plots(df_results, out_dir / 'plots')

def plot_convergence(hist_ga, hist_de, scenario_name):
    # hist_ga and hist_de are lists of best fitness per generation
    plt.figure(figsize=(8, 5))
    plt.plot(hist_ga, label='GA')
    plt.plot(hist_de, label='DE')
    plt.title(f'Convergence Comparison: {scenario_name}')
    plt.xlabel('Generation')
    plt.ylabel('Best candidate power (kW); feasibility prioritized')
    plt.legend()
    plt.tight_layout()
    # Save in convergence folder if we want, or just plots
    plot_path = RESULTS_DIR/'ablation'/'plots'/'convergence'
    plot_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path / f'{scenario_name.replace(" ", "_")}_convergence.png')
    plt.close()

def generate_comparison_plots(df, plot_dir):
    sns.set_theme(style="whitegrid")
    
    # Savings Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Scenario', y='Power_Reduction_Pct', hue='Algorithm')
    plt.title('GA/DE diagnostic candidate power change (check feasibility)')
    plt.ylabel('Power Reduction (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plot_dir / 'ga_vs_de_savings.png')
    plt.close()
    
    # Runtime Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Scenario', y='Runtime_s', hue='Algorithm')
    plt.title('Algorithm Comparison: Runtime (Seconds)')
    plt.ylabel('Runtime (s)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plot_dir / 'ga_vs_de_runtime.png')
    plt.close()

if __name__ == "__main__":
    run_comparison()
