"""Frozen surrogate predictions with explicit feasibility constraints."""
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from ml.config import ALL_FEATURES
from ml.optimization.constraints import X_LOWER, X_UPPER, MAX_INLET, MAX_HOTSPOT, PENALTY_THERMAL, PENALTY_OOD
from ml.optimization.ood_detector import OODDetector

DECISIONS = ["cooling_setpoint_c","supply_air_temperature_c","fan_speed_pct","pump_speed_pct"]
from ml.models.predict import get_surrogate_models

class CoolingOptimizationProblem(Problem):
    def __init__(self, scenario_row, apply_thermal=True, apply_ood=True, limits=None, margins=None):
        if len(scenario_row)!=1:
            raise ValueError("Exactly one operating-state row is required")
        self.limits = limits or {}
        bounds = self.limits.get("bounds",dict(zip(DECISIONS,zip(X_LOWER,X_UPPER))))
        self.max_inlet = self.limits.get("max_inlet_c",MAX_INLET)
        self.max_hotspot = self.limits.get("max_hotspot_c",MAX_HOTSPOT)
        self.power_cap = self.limits.get("max_power_kw")
        self.margins = margins or {"inlet":0.0,"hotspot":0.0}
        count = (2 if apply_thermal else 0) + (1 if apply_ood else 0) + (1 if self.power_cap is not None else 0)
        super().__init__(n_var=4,n_obj=1,n_ieq_constr=count,
            xl=np.array([bounds[k][0] for k in DECISIONS]),xu=np.array([bounds[k][1] for k in DECISIONS]))
        self.scenario_row = scenario_row.copy()
        self.apply_thermal, self.apply_ood = apply_thermal, apply_ood
        self.model_power,self.model_inlet,self.model_hotspot,self.model_load = get_surrogate_models()
        self.ood_detector = OODDetector()
        self.feature_order = ALL_FEATURES
        self.allowed_features = ALL_FEATURES

    def candidates(self, X):
        X = np.asarray(X,dtype=float)
        if X.ndim != 2 or X.shape[1] != 4 or not np.isfinite(X).all():
            raise ValueError("Controls must be a finite N x 4 array")
        frame = pd.concat([self.scenario_row]*len(X),ignore_index=True)
        frame[DECISIONS] = X
        return frame

    def _evaluate(self, X, out, *args, **kwargs):
        frame = self.candidates(X)
        features = frame[self.feature_order]
        values = {key: np.asarray(model.predict(features),dtype=float) for key,model in
            [("power",self.model_power),("inlet",self.model_inlet),("hotspot",self.model_hotspot),("load",self.model_load)]}
        if not all(np.isfinite(v).all() for v in values.values()) or (values["power"]<=0).any() or (values["load"]<=0).any():
            raise ValueError("Surrogate returned invalid predictions")
        distance, valid = self.ood_detector.inspect(frame)
        inlet_v = values["inlet"]+self.margins["inlet"]-self.max_inlet
        hotspot_v = values["hotspot"]+self.margins["hotspot"]-self.max_hotspot
        thermal = PENALTY_THERMAL*(np.maximum(0,inlet_v)**2+np.maximum(0,hotspot_v)**2)
        ood = PENALTY_OOD*np.maximum(0,distance-self.ood_detector.d_max)**2
        constraints = []
        if self.apply_thermal: constraints.extend([inlet_v,hotspot_v])
        if self.apply_ood: constraints.append(np.where(valid,distance-self.ood_detector.d_max,np.maximum(distance-self.ood_detector.d_max,1e-6)))
        if self.power_cap is not None: constraints.append(values["power"]-self.power_cap)
        out.update(values)
        out["F"] = values["power"][:,None]
        out["G"] = np.column_stack(constraints) if constraints else np.empty((len(frame),0))
        out.update(ood_valid=valid,ood_distance=distance,thermal_penalty=thermal,ood_penalty=ood)

