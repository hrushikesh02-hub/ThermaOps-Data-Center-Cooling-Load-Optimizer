import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def calculate_savings(baseline_power, optimized_power):
    """
    estimated_savings_pct = (baseline - optimized) / baseline * 100
    """
    if baseline_power <= 0:
        return 0.0
    return ((baseline_power - optimized_power) / baseline_power) * 100

def generate_comparison_plots(ga_history, de_history, scenario_name, output_dir):
    """
    Plots convergence history for both algorithms.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    if ga_history:
        plt.plot(range(1, len(ga_history)+1), ga_history, label='Genetic algorithm', marker='o', markersize=3, alpha=0.7)
    if de_history:
        plt.plot(range(1, len(de_history)+1), de_history, label='Differential Evolution', marker='x', markersize=3, alpha=0.7)
        
    plt.xlabel('Generations')
    plt.ylabel('Best candidate power (kW); feasibility prioritized')
    plt.title(f'Convergence History: {scenario_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    safe_name = scenario_name.replace(" ", "_").lower()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{safe_name}_convergence.png", dpi=150)
    plt.close()

def generate_summary_plots(scenario_df, output_dir):
    """
    Generates comparison bar plots across all scenarios for Power, Savings %, and Runtime.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Cooling Power Comparison Plot
    plt.figure(figsize=(12, 6))
    scenarios = scenario_df['Scenario']
    x = range(len(scenarios))
    width = 0.25
    
    plt.bar([i - width for i in x], scenario_df['Baseline_Power'], width=width, label='Baseline', color='#7f8c8d')
    plt.bar(x, scenario_df['GA_Power'], width=width, label='GA (Optimized)', color='#2980b9')
    plt.bar([i + width for i in x], scenario_df['DE_Power'], width=width, label='DE (Optimized)', color='#27ae60')
    
    plt.xlabel('Validation Scenario', fontweight='bold')
    plt.ylabel('Predicted Cooling Power (kW)', fontweight='bold')
    plt.title('Baseline vs GA vs DE Optimized Cooling Power Across Scenarios', fontweight='bold')
    plt.xticks(x, scenarios, rotation=25, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "cooling_power_comparison.png", dpi=150)
    plt.close()
    
    # 2. Energy Savings % Comparison Plot
    plt.figure(figsize=(10, 5))
    plt.bar([i - width/2 for i in x], scenario_df['GA_Savings_Pct'], width=width, label='GA Savings %', color='#3498db')
    plt.bar([i + width/2 for i in x], scenario_df['DE_Savings_Pct'], width=width, label='DE Savings %', color='#2ecc71')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xlabel('Validation Scenario', fontweight='bold')
    plt.ylabel('Instantaneous Cooling Power Savings (%)', fontweight='bold')
    plt.title('GA vs DE Cooling Power Reduction Across Scenarios', fontweight='bold')
    plt.xticks(x, scenarios, rotation=25, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "energy_savings_comparison.png", dpi=150)
    plt.close()
    
    # 3. Runtime Comparison Plot
    plt.figure(figsize=(10, 5))
    plt.bar([i - width/2 for i in x], scenario_df['GA_Runtime_s'], width=width, label='GA Runtime (s)', color='#e67e22')
    plt.bar([i + width/2 for i in x], scenario_df['DE_Runtime_s'], width=width, label='DE Runtime (s)', color='#e74c3c')
    plt.xlabel('Validation Scenario', fontweight='bold')
    plt.ylabel('Execution Time (seconds)', fontweight='bold')
    plt.title('Optimizer Computational Efficiency (Runtime in Seconds)', fontweight='bold')
    plt.xticks(x, scenarios, rotation=25, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "runtime_comparison.png", dpi=150)
    plt.close()

def save_results(results_list, output_file):
    df = pd.DataFrame(results_list)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
