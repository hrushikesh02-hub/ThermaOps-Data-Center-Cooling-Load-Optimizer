"""Load the four validated, frozen prediction pipelines."""
from functools import lru_cache
import joblib
from ml.config import MODELS_DIR
from ml.models.profile import load_model_profile


@lru_cache(maxsize=1)
def get_surrogate_models():
    profile = load_model_profile()
    result = []
    for key in ['power','inlet','hotspot','load']:
        pipeline = joblib.load(MODELS_DIR/profile['selected_models'][key])
        estimator = pipeline.named_steps['model']
        if 'n_jobs' in estimator.get_params():
            estimator.set_params(n_jobs=1)
        result.append(pipeline)
    return tuple(result)
