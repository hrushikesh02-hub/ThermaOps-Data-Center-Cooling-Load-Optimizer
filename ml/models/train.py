"""Reproduce seasonal training from readable settings, without old fitted models."""
import json
import hashlib
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from ml.config import ALL_FEATURES, DATA_DIR, PARAM_GRIDS, RANDOM_SEED
from ml.preprocessing.pipeline import build_preprocessor
from ml.preprocessing.validate import audit_data
from ml.utils import check_leakage

TARGET_KEYS = {'power':'cooling_power_kw', 'inlet':'server_inlet_temperature_c',
               'hotspot':'hotspot_temperature_c', 'load':'cooling_load_kw'}
FACTORIES = {'dummy':DummyRegressor, 'linearregression':LinearRegression,
             'randomforest':RandomForestRegressor, 'xgboost':XGBRegressor,
             'lightgbm':LGBMRegressor}
PREFIX_ROWS = 1000


def load_training_frames():
    """Retain the original chronological 8000/500 seasonal training boundary."""
    original_train = pd.read_csv(DATA_DIR/'train.csv')
    original_validation = pd.read_csv(DATA_DIR/'validation.csv')
    train = pd.concat([original_train, original_validation.iloc[:PREFIX_ROWS]], ignore_index=True)
    validation = original_validation.iloc[PREFIX_ROWS:].copy()
    if validation.empty or not train.timestamp.max() < validation.timestamp.min():
        raise ValueError('Training and validation must use separate chronological periods')
    return train, validation


def load_recipes():
    return json.loads(Path(__file__).with_name('training_config.json').read_text(encoding='utf-8'))


def build_pipeline(recipe, features):
    estimator = FACTORIES[recipe['family']](**recipe['parameters'])
    if 'n_jobs' in estimator.get_params():
        estimator.set_params(n_jobs=1)
    if 'verbosity' in estimator.get_params():
        estimator.set_params(verbosity=-1 if recipe['family']=='lightgbm' else 0)
    return Pipeline([('preprocessor', build_preprocessor(features, recipe['scale_numeric'])),
                     ('model', estimator)])


def metrics(actual, prediction):
    if not np.isfinite(prediction).all():
        raise ValueError('Non-finite model predictions')
    return {'mae':float(mean_absolute_error(actual,prediction)),
            'rmse':float(mean_squared_error(actual,prediction)**.5),
            'r2':float(r2_score(actual,prediction))}


def train_bundle(output_dir, tune=False):
    """Train 20 candidates and save only the four validation-selected pipelines.

    Dummy and linear regression provide the same benchmark comparisons as before.
    Default settings reproduce seasonal_v1; optional tuning uses training-only CV.
    The caller chooses a NEW output folder, so active models are never overwritten.
    """
    audit_data()
    train, validation = load_training_frames()
    recipes = load_recipes()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifacts = output/'seasonal_v1'
    artifacts.mkdir()
    report = {'training_rows':len(train), 'validation_rows':len(validation),
              'candidates':[], 'selected':{}, 'tuned':tune}
    selected = {}
    for key, target in TARGET_KEYS.items():
        check_leakage(ALL_FEATURES, target)
        best, best_model = None, None
        for family in FACTORIES:
            recipe = recipes[f'{family}_{target}']
            pipeline = build_pipeline(recipe, train[ALL_FEATURES])
            start = time.perf_counter()
            grid_name = {'randomforest':'RandomForest','xgboost':'XGBoost','lightgbm':'LightGBM'}.get(family)
            if tune and grid_name:
                search = RandomizedSearchCV(pipeline,
                    {f'model__{k}':v for k,v in PARAM_GRIDS[grid_name].items()},
                    n_iter=5, cv=TimeSeriesSplit(3), scoring='neg_mean_absolute_error',
                    random_state=RANDOM_SEED, n_jobs=1, error_score='raise')
                search.fit(train[ALL_FEATURES], train[target])
                pipeline = search.best_estimator_
            else:
                pipeline.fit(train[ALL_FEATURES], train[target])
            prediction = pipeline.predict(validation[ALL_FEATURES])
            item = {'file':f'{family}_{target}.joblib','target':target,'family':family,
                    **{'validation_'+k:v for k,v in metrics(validation[target],prediction).items()},
                    'training_seconds':time.perf_counter()-start}
            report['candidates'].append(item)
            print(json.dumps(item), flush=True)
            if best is None or item['validation_mae'] < best['validation_mae']:
                best, best_model = item, pipeline
        selected[key] = 'seasonal_v1/'+best['file']
        joblib.dump(best_model, output/selected[key])
        report['selected'][key] = best.copy()

    # Selection is finished before test targets are used for accuracy evaluation.
    test = pd.read_csv(DATA_DIR/'test.csv')
    if not validation.timestamp.max() < test.timestamp.min():
        raise ValueError('Test observations must follow validation')
    for key, target in TARGET_KEYS.items():
        model = joblib.load(output/selected[key])
        prediction = model.predict(test[ALL_FEATURES])
        residual = validation[target].to_numpy()-model.predict(validation[ALL_FEATURES])
        report['selected'][key].update(
            **{'test_'+k:v for k,v in metrics(test[target],prediction).items()},
            validation_upper_residual_95=float(max(0,np.quantile(residual,.95))))
    profile = {'version':1, 'name':'seasonal_v1',
        'description':'Chronological seasonal coverage: 8000 training / 500 validation / 1500 held-out replay rows. Original CSV files retained.',
        'training_validation_prefix_rows':PREFIX_ROWS,
        'dataset_sha256':{n:hashlib.sha256((DATA_DIR/f'{n}.csv').read_bytes()).hexdigest()
                          for n in ['train','validation','test']},
        'selected_models':selected,
        'model_sha256':{name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in selected.values()}}
    for filename,value in [('model_profile.json',profile),('selected_models.json',selected),('training_report.json',report)]:
        (output/filename).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report
