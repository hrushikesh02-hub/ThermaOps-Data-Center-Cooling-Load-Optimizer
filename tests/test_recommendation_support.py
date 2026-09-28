"""Regressions for fixed-state OOD blockers and saved-policy snapshot selection."""
import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import minimize
from api.app import create_app
from api.service import default_limits
from ml.optimization.fitness import DECISIONS
from ml.optimization.ood_detector import OODDetector, PHYSICAL_FEATURES
from ml.optimization.runner import run_optimizer
from ml.models.profile import model_frames, load_model_profile


class RecommendationSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.app=create_app(cls.temp.name)
        cls.service=cls.app.extensions['cooling']
        cls.client=cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_distance_minimum_matches_independent_convex_solver(self):
        detector=OODDetector()
        frame=self.service.frame.iloc[::199]
        for bounds in [default_limits()['bounds'],dict(zip(DECISIONS,[[24,25],[18,19],[30.2,31],[90,100]]))]:
            with self.subTest(bounds=bounds):
                controls,distances,reachable=detector.closest_controls(frame,bounds)
                idx=[PHYSICAL_FEATURES.index(k) for k in DECISIONS]
                z=detector.preprocessor.transform(frame[PHYSICAL_FEATURES])-detector.mu
                offset=detector.preprocessor.mean_[idx]+detector.mu[idx]*detector.preprocessor.scale_[idx]
                scale=detector.preprocessor.scale_[idx]
                lo=(np.array([bounds[k][0] for k in DECISIONS])-offset)/scale
                hi=(np.array([bounds[k][1] for k in DECISIONS])-offset)/scale
                for i,row in enumerate(z):
                    def objective(x):
                        candidate=row.copy();candidate[idx]=x
                        return candidate@detector.inv_cov@candidate,2*(detector.inv_cov@candidate)[idx]
                    result=minimize(objective,np.clip(row[idx],lo,hi),jac=True,bounds=list(zip(lo,hi)),
                        method='L-BFGS-B',options={'ftol':1e-13,'gtol':1e-9})
                    self.assertAlmostEqual(distances[i]**2,result.fun,places=7)
                candidate=frame.copy();candidate[DECISIONS]=controls
                actual,valid=detector.inspect(candidate)
                np.testing.assert_allclose(actual,distances,rtol=1e-10)
                np.testing.assert_array_equal(valid,reachable)

    def test_all_snapshots_reproduce_region_coverage(self):
        support=self.service.snapshot_support(self.service.get_limits('DC_1'))
        self.assertEqual(int(support.baseline_region_valid.sum()),1444)
        self.assertEqual(int(support.region_reachable.sum()),1448)

    def test_unsupported_state_skips_search_and_explains_fixed_state(self):
        support=self.service.snapshot_support(self.service.get_limits('DC_1'))
        row=self.service.state(support.minimum_region_distance.idxmax())
        for algorithm in ['GA','DE']:
            with self.subTest(algorithm=algorithm),patch('ml.optimization.runner.minimize',side_effect=AssertionError('Impossible OOD state must skip GA/DE')):
                response=self.client.post('/api/optimize',json={'state_id':row.index[0],'algorithm':algorithm})
                self.assertEqual(response.status_code,201,response.text)
                run=response.json
                self.assertEqual(run['diagnostic']['code'],'unsupported_operating_state')
                self.assertIsNone(run['recommendation'])
                self.assertIsNone(run['savings_pct'])
                region=run['diagnostic']['region']
                self.assertGreater(region['minimum_distance'],region['threshold'])
                self.assertTrue(region['largest_state_shifts'])
                self.assertLessEqual(run['evaluations'],3)

    def test_state_list_support_uses_saved_limits(self):
        before=self.client.get('/api/states?dc=DC_3&limit=500').json
        passing=[s for s in before['items'] if s['baseline_feasible']]
        self.assertTrue(passing)
        for state in passing:
            prediction=self.client.post('/api/predict',json={'state_id':state['state_id']}).json
            self.assertTrue(prediction['feasible'])
        policy=self.service.get_limits('DC_3')
        limits=default_limits();limits['max_power_kw']=1
        try:
            saved=self.client.put('/api/limits/DC_3',json={'expected_revision':policy['revision'],'limits':limits}).json
            after=self.client.get('/api/states?dc=DC_3&limit=500').json
            self.assertEqual(after['policy_revision'],saved['revision'])
            self.assertFalse(any(s['baseline_feasible'] for s in after['items']))
            self.assertEqual(before['support_summary']['region_reachable'],after['support_summary']['region_reachable'])
        finally:
            current=self.service.get_limits('DC_3')
            self.service.set_limits('DC_3',{'expected_revision':current['revision'],'limits':default_limits()})

    def test_nearest_region_seed_and_thermal_checks_are_revalidated(self):
        support=self.service.snapshot_support(self.service.get_limits('DC_1'))
        state_id=support.index[support.baseline_feasible][0]
        row=self.service.state(state_id)
        for algorithm in ['GA','DE']:
            result=run_optimizer(row,algorithm,limits=default_limits(),margins=self.service.base_margins,
                population_size=8,max_generations=2)
            self.assertFalse(result['region_blocked'])
            self.assertTrue(result['feasible'])
            controls=dict(zip(DECISIONS,result['best_x']))
            self.assertTrue(self.service.prediction(row,controls,self.service.get_limits('DC_1'))['feasible'])

    def test_unknown_center_cannot_be_supported(self):
        row=self.service.frame.iloc[[0]].copy();row['data_center_id']='unknown'
        _,_,reachable=OODDetector().closest_controls(row,default_limits()['bounds'])
        self.assertFalse(reachable[0])

    def test_reported_dc2_failure_is_now_supported(self):
        frame=self.service.frame
        row=frame[(frame.data_center_id=='DC_2') & (frame.timestamp=='2024-07-26 22:30:00')]
        for algorithm in ['GA','DE']:
            response=self.client.post('/api/optimize',json={'state_id':row.index[0],'algorithm':algorithm})
            self.assertEqual(response.status_code,201,response.text)
            self.assertEqual(response.json['status'],'feasible',response.json['diagnostic'])

    def test_active_split_has_monsoon_and_no_test_leakage(self):
        frames=model_frames()
        self.assertEqual([len(frames[k]) for k in ['train','validation','test']],[8000,500,1500])
        self.assertIn('Monsoon',set(frames['train'].season))
        self.assertLess(frames['train'].timestamp.max(),frames['validation'].timestamp.min())
        self.assertLess(frames['validation'].timestamp.max(),frames['test'].timestamp.min())
        self.assertEqual(self.service.data_quality['unknown_test_categories']['season'],[])
        self.assertEqual(self.service.model_report['inlet']['validation']['rows'],500)
        self.assertEqual(len(load_model_profile()['model_sha256']),4)

    def test_model_change_invalidates_old_runs_without_policy_change(self):
        support=self.service.snapshot_support(self.service.get_limits('DC_1'))
        sid=support.index[support.baseline_feasible & self.service.frame.data_center_id.eq('DC_1')][0]
        run=self.service.optimize({'state_id':sid})
        self.assertFalse(self.service.run(run['id'])['stale'])
        legacy=dict(run);legacy.pop('model_revision')
        with self.service.connect() as db:
            db.execute('UPDATE runs SET body=? WHERE id=?',(json.dumps(legacy),run['id']))
        self.assertTrue(self.service.run(run['id'])['stale'])
        response=self.client.post('/api/recommendations/'+run['id']+'/apply',json={'state_id':sid,'expected_revision':run['policy']['revision']})
        self.assertEqual(response.status_code,409)

    def test_profile_rejects_changed_data_or_model_artifacts(self):
        original=Path.read_bytes
        for filename in ['test.csv','randomforest_cooling_power_kw.joblib']:
            def changed(path):
                return b'changed artifact' if path.name==filename else original(path)
            with self.subTest(filename=filename),patch.object(Path,'read_bytes',changed):
                with self.assertRaises(ValueError):
                    load_model_profile.__wrapped__()


if __name__=='__main__':
    unittest.main()
