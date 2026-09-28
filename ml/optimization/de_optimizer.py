from ml.optimization.runner import run_optimizer

def run_de_optimization(scenario_row, apply_thermal=True, apply_ood=True, seed=None, **kwargs):
    return run_optimizer(scenario_row,"DE",apply_thermal,apply_ood,seed,**kwargs)

