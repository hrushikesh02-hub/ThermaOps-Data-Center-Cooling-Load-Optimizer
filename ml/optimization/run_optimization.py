from ml.config import RESULTS_DIR
import sys
import warnings


import matplotlib
matplotlib.use("Agg")

import pandas as pd
from pathlib import Path


from ml.optimization.scenarios import get_scenarios
from ml.optimization.baseline import calculate_baseline
from ml.optimization.ga_optimizer import run_ga_optimization
from ml.optimization.de_optimizer import run_de_optimization
from ml.evaluation.optimization import calculate_savings, generate_comparison_plots, generate_summary_plots, save_results
from ml.utils import setup_logger

logger = setup_logger("run_optimization")
RESULTS_OPT_DIR = RESULTS_DIR / "optimization"

def main():
    logger.info("=======================================")
    logger.info("STARTING EVOLUTIONARY OPTIMIZATION")
    logger.info("=======================================")
    
    scenarios = get_scenarios()
    logger.info(f"Loaded {len(scenarios)} validation scenarios.")
    
    all_ga_results = []
    all_de_results = []
    comparison_summary = []
    
    for name, scenario_row in scenarios.items():
        logger.info(f"\n--- Processing Scenario: {name} ---")
        
        # 1. Baseline
        baseline_power = calculate_baseline(scenario_row)[0]
        logger.info(f"Baseline Predicted Power: {baseline_power:.2f} kW")
        
        # 2. Run GA
        logger.info("Running Genetic Algorithm (GA)...")
        ga_res = run_ga_optimization(scenario_row)
        ga_savings = calculate_savings(baseline_power, ga_res['power'])
        
        # 3. Run DE
        logger.info("Running Differential Evolution (DE)...")
        de_res = run_de_optimization(scenario_row)
        de_savings = calculate_savings(baseline_power, de_res['power'])
        
        # 4. Reporting
        logger.info(f"GA -> Power: {ga_res['power']:.2f} kW, Savings: {ga_savings:.2f}%, "
                    f"Thermal Penalty: {ga_res['thermal_penalty']:.0f}, OOD Penalty: {ga_res['ood_penalty']:.0f}")
        logger.info(f"DE -> Power: {de_res['power']:.2f} kW, Savings: {de_savings:.2f}%, "
                    f"Thermal Penalty: {de_res['thermal_penalty']:.0f}, OOD Penalty: {de_res['ood_penalty']:.0f}")
                    
        # Append GA row
        ga_dict = {"Scenario": name, "Algorithm": "GA", "Baseline_Power": baseline_power, "Savings_Pct": ga_savings}
        ga_dict.update({k: v for k, v in ga_res.items() if k != "history" and k != "best_x"})
        # Unpack best_x
        ga_dict.update({"opt_setpoint": ga_res["best_x"][0], "opt_supply_temp": ga_res["best_x"][1], 
                        "opt_fan_pct": ga_res["best_x"][2], "opt_pump_pct": ga_res["best_x"][3]})
        all_ga_results.append(ga_dict)
        
        # Append DE row
        de_dict = {"Scenario": name, "Algorithm": "DE", "Baseline_Power": baseline_power, "Savings_Pct": de_savings}
        de_dict.update({k: v for k, v in de_res.items() if k != "history" and k != "best_x"})
        de_dict.update({"opt_setpoint": de_res["best_x"][0], "opt_supply_temp": de_res["best_x"][1], 
                        "opt_fan_pct": de_res["best_x"][2], "opt_pump_pct": de_res["best_x"][3]})
        all_de_results.append(de_dict)
        
        # Scenario summary
        comparison_summary.append({
            "Scenario": name,
            "Baseline_Power": baseline_power,
            "GA_Power": ga_res['power'] if ga_res["feasible"] else float("nan"),
            "DE_Power": de_res['power'] if de_res["feasible"] else float("nan"),
            "GA_Savings_Pct": ga_savings if ga_res["feasible"] else float("nan"),
            "DE_Savings_Pct": de_savings if de_res["feasible"] else float("nan"),
            "GA_Valid": ga_res["feasible"],
            "DE_Valid": de_res["feasible"],
            "GA_Runtime_s": ga_res['runtime_s'],
            "DE_Runtime_s": de_res['runtime_s']
        })
        
        # 5. Plot Convergence
        generate_comparison_plots(ga_res["history"], de_res["history"], name, RESULTS_OPT_DIR / "convergence")
        
    # Save CSVs
    save_results(all_ga_results, RESULTS_OPT_DIR / "ga_results.csv")
    save_results(all_de_results, RESULTS_OPT_DIR / "de_results.csv")
    save_results(comparison_summary, RESULTS_OPT_DIR / "scenario_results.csv")
    
    # Generate overall summary plots
    summary_df = pd.DataFrame(comparison_summary)
    generate_summary_plots(summary_df, RESULTS_OPT_DIR / "plots")
    
    logger.info("\n=======================================")
    logger.info("OPTIMIZATION COMPLETED SUCCESSFULLY")
    logger.info("Results saved to outputs/optimization/")
    logger.info("=======================================")

if __name__ == "__main__":
    main()
