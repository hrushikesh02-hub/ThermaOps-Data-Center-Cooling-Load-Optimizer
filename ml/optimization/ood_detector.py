"""Training-only physical operating-region check, independent of calendar drift."""
from functools import lru_cache
from itertools import product
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from ml.config import DATA_DIR, CONTROL_VARIABLES
from ml.optimization.constraints import PENALTY_OOD, CONFIG
from ml.models.profile import model_frames

# Calendar fields are deliberately excluded: a later date is not a physical anomaly.
# Derived copies are excluded so duplicated information does not dominate distance.
PHYSICAL_FEATURES = ["server_count","cpu_utilization_pct","gpu_utilization_pct",
    "memory_utilization_pct","network_load_gbps","storage_io_load_pct","it_power_kw",
    "rack_density_kw","indoor_temperature_c","outdoor_temperature_c",
    "outdoor_humidity_pct","indoor_humidity_pct","airflow_cfm"] + CONTROL_VARIABLES

@lru_cache(maxsize=1)
def fitted_region():
    train = model_frames()['train']
    scaler = StandardScaler().fit(train[PHYSICAL_FEATURES])
    x = scaler.transform(train[PHYSICAL_FEATURES])
    covariance = LedoitWolf().fit(x)
    distances = np.sqrt(np.maximum(covariance.mahalanobis(x),0))
    return scaler, covariance.location_, covariance.precision_, float(np.percentile(distances,CONFIG["optimization"]["ood"]["threshold_percentile"])), set(train.data_center_id)

class OODDetector:
    def __init__(self):
        self.preprocessor, self.mu, self.inv_cov, self.d_max, self.centers = fitted_region()
        self.allowed_features = PHYSICAL_FEATURES

    def distances(self, transformed):
        x = np.asarray(transformed,dtype=float)-self.mu
        d2 = np.einsum("ij,jk,ik->i",x,self.inv_cov,x)
        return np.sqrt(np.maximum(d2,0))

    def inspect(self, frame):
        transformed = self.preprocessor.transform(frame[PHYSICAL_FEATURES])
        distance = self.distances(transformed)
        known = frame.data_center_id.isin(self.centers).to_numpy()
        valid = np.isfinite(distance) & (distance <= self.d_max) & known
        return distance, valid

    def calculate_penalty(self, transformed):
        distance = self.distances(transformed)
        violation = np.maximum(0,distance-self.d_max)
        return PENALTY_OOD*violation**2, np.isfinite(distance)&(distance<=self.d_max)

    def closest_controls(self, frame, bounds):
        """Global minimum physical distance at fixed state within control bounds.

        Mahalanobis distance squared is a convex quadratic. With four controls,
        enumerate the 81 free/lower/upper active sets and solve each free block.
        This continuous minimum is a lower bound for rounded operator controls;
        being reachable is necessary, not sufficient, for a recommendation.
        """
        names = list(bounds)
        if set(names) != set(CONTROL_VARIABLES):
            raise ValueError("Bounds must contain all four controls")
        indices = [PHYSICAL_FEATURES.index(k) for k in names]
        fixed = [i for i in range(len(PHYSICAL_FEATURES)) if i not in indices]
        z = self.preprocessor.transform(frame[PHYSICAL_FEATURES]) - self.mu
        scale = self.preprocessor.scale_[indices]
        offset = self.preprocessor.mean_[indices] + self.mu[indices]*scale
        lower = (np.array([bounds[k][0] for k in names])-offset)/scale
        upper = (np.array([bounds[k][1] for k in names])-offset)/scale
        precision = self.inv_cov[np.ix_(indices, indices)]
        linear = z[:, fixed] @ self.inv_cov[np.ix_(fixed, indices)]
        best_value = np.full(len(frame), np.inf)
        best = np.empty((len(frame), len(names)))
        for status in product((-1, 0, 1), repeat=len(names)):
            free = [i for i, s in enumerate(status) if s == 0]
            active = [i for i, s in enumerate(status) if s != 0]
            candidate = np.zeros_like(best)
            for i in active:
                candidate[:, i] = lower[i] if status[i] == -1 else upper[i]
            if free:
                rhs = linear[:, free] + candidate[:, active] @ precision[np.ix_(active, free)]
                candidate[:, free] = np.linalg.solve(precision[np.ix_(free, free)], -rhs.T).T
            valid = np.all((candidate >= lower-1e-10) & (candidate <= upper+1e-10), axis=1)
            candidate = np.clip(candidate, lower, upper)
            value = np.einsum('ij,jk,ik->i', candidate, precision, candidate) + 2*np.sum(candidate*linear, axis=1)
            improved = valid & (value < best_value)
            best[improved] = candidate[improved]
            best_value[improved] = value[improved]
        z[:, indices] = best
        distance = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', z, self.inv_cov, z), 0))
        controls = best*scale+offset
        reachable = (distance <= self.d_max) & frame.data_center_id.isin(self.centers).to_numpy()
        return controls, distance, reachable

