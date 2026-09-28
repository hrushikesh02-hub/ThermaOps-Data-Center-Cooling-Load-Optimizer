"""Optional reproducible research reports using the active seasonal bundle."""
from ml.preprocessing.prepare import prepare_datasets
from ml.evaluation.evaluate import evaluate_models
from ml.evaluation.eda import run_eda
from ml.optimization.run_optimization import main as run_optimization
from ml.evaluation.ablation.run_ablation import run_ablation
from ml.evaluation.ablation.run_algorithm_comparison import run_comparison
from ml.evaluation.ablation.run_robustness import run_robustness
from ml.evaluation.ablation.generate_reports import generate_reports
from ml.config import ensure_directories


def run_research():
    ensure_directories()
    prepare_datasets()
    evaluate_models()
    run_eda()
    run_optimization()
    run_ablation()
    run_comparison()
    run_robustness()
    generate_reports()
    print('Research reports saved under outputs/.')
