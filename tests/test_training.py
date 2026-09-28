"""Exercise training, preprocessing, persistence, and small CV fits without replacing shipped models."""
import sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from ml.preprocessing.pipeline import build_preprocessor, get_fitted_preprocessor
from ml.models import train as train_models
from ml.config import DATA_DIR, ALL_FEATURES, TARGETS

class TrainingTests(unittest.TestCase):
    def test_training_entrypoint_compares_all_families_and_saves_four_selected(self):
        frame=pd.read_csv(DATA_DIR/"train.csv").head(180)
        train,validation=frame.iloc[:150],frame.iloc[150:]
        recipes=train_models.load_recipes()
        for recipe in recipes.values():
            if recipe['family'] in ['randomforest','xgboost','lightgbm']:
                recipe['parameters'].update(n_estimators=4,max_depth=3)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'bundle'
            with patch.object(train_models,'load_training_frames',return_value=(train,validation)),patch.object(train_models,'load_recipes',return_value=recipes):
                report=train_models.train_bundle(out)
            self.assertEqual(len(report['candidates']),20)
            self.assertEqual(len(list(out.rglob('*.joblib'))),4)
            self.assertTrue((out/'model_profile.json').exists())
            for key,target in train_models.TARGET_KEYS.items():
                best=report['selected'][key]
                candidates=[r for r in report['candidates'] if r['target']==target]
                self.assertEqual(best['validation_mae'],min(r['validation_mae'] for r in candidates))
                model=joblib.load(out/'seasonal_v1'/best['file'])
                pred=model.predict(train[ALL_FEATURES].head(3))
                self.assertTrue(np.isfinite(pred).all())

    def test_preprocessor_persistence_and_unknown_category(self):
        from ml.preprocessing import pipeline as preprocessing
        data=pd.read_csv(DATA_DIR/"train.csv").head(80)[ALL_FEATURES]
        with tempfile.TemporaryDirectory() as directory,patch.object(preprocessing,"PREPROCESSORS_DIR",Path(directory)/"nested"):
            preprocessor=get_fitted_preprocessor(data,True,"smoke")
            future=data.iloc[[0]].copy();future["season"]="Monsoon"
            result=preprocessor.transform(future)
            self.assertTrue(np.isfinite(result).all())
            self.assertNotIn("server_inlet_temperature_c",preprocessor.feature_names_in_)

    def test_pipeline_cross_validation(self):
        train=pd.read_csv(DATA_DIR/"train.csv").head(120)
        pipeline=Pipeline([("preprocessor",build_preprocessor(train[ALL_FEATURES],True)),
                           ("model",RandomForestRegressor(n_estimators=4,random_state=42,n_jobs=1))])
        search=RandomizedSearchCV(pipeline,{"model__max_depth":[2,3]},n_iter=2,
            cv=TimeSeriesSplit(3),scoring="neg_mean_absolute_error",random_state=42,error_score="raise")
        search.fit(train[ALL_FEATURES],train.cooling_power_kw)
        self.assertTrue(np.isfinite(search.best_score_))

if __name__=="__main__":unittest.main(verbosity=2)

