"""Shared bounded GA/DE runner with rounded-output revalidation."""
import time
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from ml.optimization.fitness import CoolingOptimizationProblem, DECISIONS
from ml.optimization.constraints import CONFIG

class History(Callback):
    def __init__(self):
        super().__init__()
        self.values=[]
        self.candidates=[]
    def notify(self,algorithm):
        self.values.append(float(np.min(algorithm.opt.get("F"))))
        self.candidates.append(algorithm.pop.get("X").copy())

def run_optimizer(scenario_row, algorithm_name="GA", apply_thermal=True, apply_ood=True, seed=None,
                  limits=None, margins=None, population_size=None, max_generations=None):
    if algorithm_name not in ("GA","DE"):
        raise ValueError("algorithm_name must be GA or DE")
    config = CONFIG["optimization"]
    seed = config["random_seed"] if seed is None else seed
    pop = population_size or config["algorithm"]["population_size"]
    gens = max_generations or config["algorithm"]["max_generations"]
    if type(pop) is not int or pop<4 or type(gens) is not int or gens<1:
        raise ValueError("Population must be an integer >=4 and generations an integer >=1")
    problem = CoolingOptimizationProblem(scenario_row,apply_thermal,apply_ood,limits,margins)
    start = time.perf_counter()
    region_bounds = dict(zip(DECISIONS, zip(problem.xl, problem.xu)))
    nearest, minimum_distance, reachable = problem.ood_detector.closest_controls(scenario_row, region_bounds)
    rng = np.random.default_rng(seed)
    sampling = rng.uniform(problem.xl,problem.xu,size=(pop,4))
    original = scenario_row[DECISIONS].to_numpy(dtype=float)[0]
    sampling[0] = np.clip(original,problem.xl,problem.xu)
    sampling[1] = np.clip(nearest[0],problem.xl,problem.xu)
    algorithm = GA(pop_size=pop,sampling=sampling,eliminate_duplicates=True) if algorithm_name=="GA" else DE(pop_size=pop,sampling=sampling,variant="DE/rand/1/bin",CR=.9,F=.8)
    history = History()
    blocked = apply_ood and not bool(reachable[0]) and minimum_distance[0] > problem.ood_detector.d_max + 1e-8
    if blocked:
        # The continuous minimum already exceeds the threshold. More generations
        # or different thermal limits cannot make this fixed state supported.
        candidates = np.vstack([sampling[:2], original])
        evaluations = 0
    else:
        result = minimize(problem,algorithm,("n_gen",gens),seed=seed,callback=history,
                          save_history=False,verbose=False,return_least_infeasible=True)
        # Retain earlier candidates: final-population rounding can lose a valid
        # solution that the search visited before its last generation.
        candidates = np.vstack([*history.candidates, sampling[:2], original])
        evaluations = int(result.algorithm.evaluator.n_eval)
    # The operator receives these exact 0.1-unit values; recheck after rounding.
    candidates = np.unique(np.round(candidates,1),axis=0)
    out={}; problem._evaluate(candidates,out)
    within = np.all((candidates>=problem.xl)&(candidates<=problem.xu),axis=1)
    feasible = within & np.all(out["G"]<=0,axis=1)
    if feasible.any():
        ids=np.flatnonzero(feasible); best=ids[np.argmin(out["power"][ids])]
    else:
        # Diagnostic candidate only; callers must honor feasible=False.
        violation=np.maximum(out["G"],0).sum(axis=1)+np.maximum(problem.xl-candidates,0).sum(axis=1)+np.maximum(candidates-problem.xu,0).sum(axis=1)
        best=int(np.argmin(violation))
    metrics={key:float(out[key][best]) for key in ["power","inlet","hotspot","load","thermal_penalty","ood_penalty","ood_distance"]}
    return dict(best_x=candidates[best],fitness=metrics["power"],**{k:v for k,v in metrics.items() if k!="power"},
        power=metrics["power"],ood_valid=bool(out["ood_valid"][best]),feasible=bool(feasible[best]),
        runtime_s=time.perf_counter()-start,evaluations=evaluations+len(candidates),
        history=history.values,algorithm=algorithm_name,region_blocked=bool(blocked),
        minimum_region_distance=float(minimum_distance[0]))

