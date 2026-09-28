"""Generate measured summaries without preset scientific conclusions."""
import sys
from pathlib import Path
import pandas as pd
from ml.config import RESULTS_DIR, ensure_directories

def generate_reports():
    ensure_directories()
    out=RESULTS_DIR/"ablation"
    ab=pd.read_csv(out/"ablation_results.csv")
    alg=pd.read_csv(out/"ga_vs_de_comparison.csv")
    rob=pd.read_csv(out/"robustness_results.csv")
    valid=alg[alg.Feasible.astype(bool)]
    stats={
        "Overall_Average_Savings_Pct":valid.Power_Reduction_Pct.mean(),
        "Median_Savings_Pct":valid.Power_Reduction_Pct.median(),
        "Std_Savings_Pct":valid.Power_Reduction_Pct.std(),
        "Min_Savings_Pct":valid.Power_Reduction_Pct.min(),
        "Max_Savings_Pct":valid.Power_Reduction_Pct.max(),
        "Thermal_Feasibility_Rate":1-alg.Thermal_Violation_Rate.mean(),
        "OOD_Feasibility_Rate":1-alg.OOD_Violation_Rate.mean(),
        "Full_Feasibility_Rate":alg.Feasible.mean(),
        "Feasible_Runs":len(valid),"Total_Runs":len(alg),
        "Average_GA_Runtime_s":alg[alg.Algorithm=="GA"].Runtime_s.mean(),
        "Average_DE_Runtime_s":alg[alg.Algorithm=="DE"].Runtime_s.mean()}
    pd.DataFrame([stats]).to_csv(out/"summary_statistics.csv",index=False)
    def percent(value):
        return f"{value:.4f}%" if pd.notna(value) else "N/A (no feasible runs)"
    lines=["ABLATION AND ALGORITHM REPORT",
        "All values below are computed from current CSVs.",
        "Research ablations use point thermal predictions; the operator API additionally uses validation uncertainty allowances.",
        "Infeasible candidate metrics are diagnostic only and must not be applied.",
        f"Feasible algorithm runs: {len(valid)}/{len(alg)}",
        f"Mean savings among feasible runs: {percent(stats['Overall_Average_Savings_Pct'])}",
        "Experimental summaries (negative savings retained):"]
    for name,rows in ab.groupby("Experiment"):
        accepted=rows[rows.Feasible.astype(bool)]
        lines.append(f"{name}: {len(accepted)}/{len(rows)} feasible; mean feasible savings={percent(accepted.Savings_Pct.mean())}; thermal violation rate={rows.Thermal_Violations.mean():.4f}; OOD violation rate={rows.OOD_Violations.mean():.4f}")
    (out/"ablation_report.txt").write_text("\n".join(lines),encoding="utf-8")
    grouped=rob.groupby("Scenario").agg(runs=("Seed","count"),feasible=("Feasible","sum"),
        seed_savings_std=("Savings_Pct","std"),mean_savings=("Savings_Pct","mean"))
    text=["ROBUSTNESS REPORT",f"Seeds: {sorted(rob.Seed.unique().tolist())}",
        f"Feasible runs: {int(rob.Feasible.sum())}/{len(rob)}",
        "Standard deviations are calculated within each scenario across seeds, not across unrelated workloads.",
        "Candidate savings below include infeasible diagnostics; see feasibility counts.",
        grouped.to_string(),
        "No claim of a global optimum, real energy savings, or physical hardware safety follows from these surrogate experiments."]
    (out/"robustness_report.txt").write_text("\n".join(text),encoding="utf-8")
    print("\n".join(lines))
if __name__=="__main__": generate_reports()

