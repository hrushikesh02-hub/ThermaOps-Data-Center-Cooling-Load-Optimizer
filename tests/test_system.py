"""End-to-end API and scientific regressions against real supplied models."""
import copy, csv, io, json, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from api.app import create_app
from api.service import default_limits, validate_limits, InputError
from ml.optimization.fitness import CoolingOptimizationProblem, DECISIONS
from ml.optimization.ood_detector import OODDetector
from ml.optimization.runner import run_optimizer
from ml.config import DATA_DIR, ALL_FEATURES
from ml.preprocessing.validate import audit_data,validate_frame
from ml.utils import check_leakage

class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.app=create_app(cls.temp.name)
        cls.service=cls.app.extensions["cooling"]
        cls.client=cls.app.test_client()
        # Choose a genuinely in-region baseline from data; no mock model predictions.
        frame=cls.service.frame
        region=OODDetector()
        _,valid=region.inspect(frame)
        ids=frame.index[valid & (frame.data_center_id=="DC_1")].tolist()
        cls.state_id=next(i for i in ids if cls.service.predict({"state_id":i})["feasible"])
        cls.dc="DC_1"
        cls.optimized=cls.service.optimize({"state_id":cls.state_id,"algorithm":"GA"})
        assert cls.optimized["status"]=="feasible"

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def setUp(self):
        p=self.service.get_limits(self.dc)
        if p["limits"]!=default_limits():
            self.service.set_limits(self.dc,{"expected_revision":p["revision"],"limits":default_limits()})

    def post(self,path,body):
        return self.client.post(path,json=body)

    def test_01_all_read_endpoints_and_static_assets(self):
        paths=["/","/static/styles.css","/static/app.js","/api/health","/api/config",
            "/api/states?dc=DC_1","/api/states/"+self.state_id,
            "/api/trends?dc=DC_1&until="+self.state_id,"/api/limits/DC_1",
            "/api/history","/api/history/"+self.optimized["id"],"/api/history.csv",
            "/api/datasets","/api/models","/api/docs","/api/openapi.json"]
        for path in paths:
            with self.subTest(path=path):
                response=self.client.get(path)
                self.assertEqual(response.status_code,200,response.data[:300])
                self.assertIn("nosniff",response.headers["X-Content-Type-Options"])
        self.assertEqual(self.client.get("/api/health").json["models_loaded"],4)

    def test_02_predict_uses_models_and_validates_controls(self):
        base=self.post("/api/predict",{"state_id":self.state_id})
        self.assertEqual(base.status_code,200)
        self.assertTrue(base.json["feasible"])
        for value in base.json["predictions"].values(): self.assertTrue(np.isfinite(value))
        control=dict(base.json["controls"]);control["supply_air_temperature_c"]=30
        preview=self.post("/api/predict",{"state_id":self.state_id,"controls":control})
        self.assertEqual(preview.status_code,200)
        self.assertFalse(preview.json["feasible"])
        self.assertNotEqual(base.json["predictions"]["inlet"],preview.json["predictions"]["inlet"])

    def test_03_limits_persist_and_isolate_centers(self):
        original=self.service.get_limits(self.dc)
        other=self.service.get_limits("DC_2")
        changed=default_limits();changed["max_inlet_c"]=26.5
        response=self.client.put("/api/limits/DC_1",json={"expected_revision":original["revision"],"limits":changed})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json["revision"],original["revision"]+1)
        self.assertEqual(self.service.get_limits("DC_2"),other)
        with self.service.connect() as db:
            stored=json.loads(db.execute("SELECT body FROM limits WHERE dc='DC_1'").fetchone()[0])
        self.assertEqual(stored["max_inlet_c"],26.5)

    def test_04_stale_policy_write_conflicts(self):
        p=self.service.get_limits(self.dc)
        body={"expected_revision":p["revision"],"limits":default_limits()}
        self.assertEqual(self.client.put("/api/limits/DC_1",json=body).status_code,200)
        self.assertEqual(self.client.put("/api/limits/DC_1",json=body).status_code,409)

    def test_05_invalid_prediction_input_matrix(self):
        baseline=self.service.predict({"state_id":self.state_id})["controls"]
        cases=[{},[],None,{"state_id":[]},{"state_id":self.state_id,"extra":1},
               {"state_id":self.state_id,"controls":{}},{"state_id":self.state_id,"controls":[]}]
        for value in [True,"22",None,float("nan"),float("inf"),-1,101]:
            controls=baseline.copy();controls["fan_speed_pct"]=value
            cases.append({"state_id":self.state_id,"controls":controls})
        for body in cases:
            with self.subTest(body=str(body)):
                response=self.client.post("/api/predict",data=json.dumps(body),content_type="application/json")
                self.assertEqual(response.status_code,422,response.data)

    def test_06_invalid_limit_matrix(self):
        cases=[]
        for key,value in [("max_inlet_c",28),("max_hotspot_c",36),("reserve_c",-1),("max_power_kw",0),("max_power_kw",float("nan"))]:
            body=default_limits();body[key]=value;cases.append(body)
        for bounds in [[27,18],[18,18],[17,27],[18.01,27],[18],["18",27]]:
            body=default_limits();body["bounds"]["cooling_setpoint_c"]=bounds;cases.append(body)
        body=default_limits();body["bounds"]={};cases.append(body)
        body=default_limits();body["max_hotspot_c"]=20;cases.append(body)
        for limits in cases:
            with self.subTest(limits=limits):
                p=self.service.get_limits(self.dc)
                r=self.client.put("/api/limits/DC_1",json={"expected_revision":p["revision"],"limits":limits})
                self.assertEqual(r.status_code,422,r.data)

    def test_07_ga_endpoint_creates_valid_run(self):
        response=self.post("/api/optimize",{"state_id":self.state_id,"algorithm":"GA","seed":42})
        self.assertEqual(response.status_code,201,response.data)
        r=response.json;self.assertEqual(r["status"],"feasible")
        self.assertFalse(r["stale"])
        for k,value in r["recommendation"]["controls"].items():
            self.assertAlmostEqual(value*10,round(value*10))
        self.assertLessEqual(r["recommendation"]["upper_estimates_c"]["inlet"],r["policy"]["limits"]["max_inlet_c"])
        self.assertTrue(r["recommendation"]["ood"]["valid"])
        self.assertEqual(self.client.get("/api/history/"+r["id"]).json["id"],r["id"])

    def test_08_de_endpoint_creates_valid_run(self):
        response=self.post("/api/optimize",{"state_id":self.state_id,"algorithm":"DE"})
        self.assertEqual(response.status_code,201,response.data)
        self.assertEqual(response.json["status"],"feasible")
        self.assertTrue(response.json["recommendation"]["feasible"])

    def test_09_infeasible_limit_withholds_controls_and_savings(self):
        p=self.service.get_limits(self.dc);limits=default_limits();limits["max_power_kw"]=1
        self.service.set_limits(self.dc,{"expected_revision":p["revision"],"limits":limits})
        response=self.post("/api/optimize",{"state_id":self.state_id})
        self.assertEqual(response.status_code,201)
        self.assertEqual(response.json["status"],"infeasible")
        self.assertIsNone(response.json["recommendation"]);self.assertIsNone(response.json["savings_kw"])
        r=response.json
        self.assertEqual(self.post("/api/recommendations/"+r["id"]+"/apply",{"state_id":self.state_id,"expected_revision":r["policy"]["revision"]}).status_code,409)

    def test_10_apply_is_persistent_and_idempotent(self):
        run=self.service.optimize({"state_id":self.state_id})
        body={"state_id":self.state_id,"expected_revision":run["policy"]["revision"]}
        response=self.post("/api/recommendations/"+run["id"]+"/apply",body)
        self.assertEqual(response.status_code,200,response.data);self.assertIsNotNone(response.json["applied_at"])
        again=self.post("/api/recommendations/"+run["id"]+"/apply",body)
        self.assertEqual(again.json["applied_at"],response.json["applied_at"])
        state=self.client.get("/api/states/"+self.state_id).json
        self.assertEqual(state["simulated"]["controls"],run["recommendation"]["controls"])
        with self.service.connect() as db:
            rows=db.execute("SELECT body FROM events WHERE event='simulation_setpoints_applied'").fetchall()
        self.assertEqual(sum(json.loads(r[0])["run_id"]==run["id"] for r in rows),1)

    def test_11_changed_state_and_limits_block_apply(self):
        run=self.service.optimize({"state_id":self.state_id})
        path="/api/recommendations/"+run["id"]+"/apply"
        wrong={"state_id":"another-state","expected_revision":run["policy"]["revision"]}
        self.assertEqual(self.post(path,wrong).status_code,409)
        self.service.set_limits(self.dc,{"expected_revision":run["policy"]["revision"],"limits":default_limits()})
        self.assertEqual(self.post(path,{"state_id":self.state_id,"expected_revision":run["policy"]["revision"]}).status_code,409)
        self.assertTrue(self.client.get("/api/history/"+run["id"]).json["stale"])

    def test_12_busy_optimizer_returns_429_and_health_remains_ready(self):
        with self.service.optimize_lock:
            response=self.post("/api/optimize",{"state_id":self.state_id})
            self.assertEqual(response.status_code,429)
            self.assertTrue(self.client.get("/api/health").json["optimization_busy"])
        self.assertFalse(self.client.get("/api/health").json["optimization_busy"])

    def test_13_unknown_resources_and_methods(self):
        for path in ["/api/states/missing","/api/limits/unknown","/api/history/missing","/api/none","/static/missing"]:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code,404)
        self.assertEqual(self.client.delete("/api/history").status_code,405)
        self.assertEqual(self.post("/api/recommendations/missing/apply",{"state_id":self.state_id,"expected_revision":1}).status_code,404)

    def test_14_pagination_and_trend_scoping(self):
        for path in ["/api/states?limit=0","/api/states?offset=-1","/api/history?limit=501","/api/trends?limit=abc","/api/history?offset=1.2"]:
            with self.subTest(path=path): self.assertEqual(self.client.get(path).status_code,422)
        first=self.client.get("/api/states?dc=DC_1&limit=2").json
        next_page=self.client.get("/api/states?dc=DC_1&limit=2&offset=2").json
        self.assertFalse(set(i["state_id"] for i in first["items"])&set(i["state_id"] for i in next_page["items"]))
        self.assertEqual(first["total"],300)
        self.assertEqual(self.client.get("/api/trends?dc=DC_2&until="+self.state_id).status_code,422)
        trend=self.client.get("/api/trends?dc=DC_1&until="+self.state_id).json
        end=self.service.state(self.state_id).iloc[0].timestamp
        self.assertTrue(all(i["timestamp"]<=end for i in trend["items"]))

    def test_15_transport_errors_are_json(self):
        self.assertEqual(self.client.post("/api/predict",data="{}",content_type="text/plain").status_code,415)
        self.assertEqual(self.client.post("/api/predict",data="{broken",content_type="application/json").status_code,400)
        self.assertEqual(self.client.post("/api/predict",data="x"*20000,content_type="application/json").status_code,413)
        self.assertEqual(self.client.post("/api/predict",json={"state_id":self.state_id},headers={"Origin":"https://example.org"}).status_code,403)

    def test_16_csv_includes_all_rows(self):
        data=list(csv.DictReader(io.StringIO(self.client.get("/api/history.csv?dc=DC_1").text)))
        self.assertEqual(len(data),self.service.history(self.dc)["total"])
        self.assertTrue(all(row["data_center_id"]==self.dc for row in data))

    def test_17_dataset_and_split_integrity(self):
        quality=audit_data()
        self.assertEqual(quality["total_rows"],10000)
        self.assertEqual([quality["splits"][n]["rows"] for n in ["train","validation","test"]],[7000,1500,1500])
        self.assertEqual(quality["unknown_test_categories"]["season"],["Monsoon"])

    def test_18_corrupt_data_fails_closed(self):
        original=pd.read_csv(DATA_DIR/"test.csv").head(4)
        mutations=[lambda d:d.drop(columns="it_power_kw"),lambda d:pd.concat([d,d.iloc[[0]]]),
            lambda d:d.assign(it_power_kw=float("inf")),lambda d:d.assign(cpu_utilization_pct=-1),
            lambda d:d.assign(hotspot_safety_margin_c=0)]
        for mutate in mutations:
            with self.subTest(mutation=str(mutate)):
                with self.assertRaises(ValueError): validate_frame(mutate(original.copy()))

    def test_19_target_leakage_rejected(self):
        for target in ["cooling_power_kw","cooling_load_kw","server_inlet_temperature_c","hotspot_temperature_c"]:
            check_leakage(ALL_FEATURES,target)
            for leak in ["cooling_power_kw","server_inlet_temperature_c","hotspot_safety_margin_c","return_air_temperature_c"]:
                with self.subTest(target=target,leak=leak):
                    with self.assertRaises(ValueError):check_leakage(ALL_FEATURES+[leak],target)

    def test_20_ood_respects_physical_controls_and_ignores_calendar(self):
        row=self.service.state(self.state_id);detector=OODDetector()
        before,valid=detector.inspect(row)
        shifted=row.copy();shifted["month"]=12;shifted["week_of_year"]=52;shifted["season"]="Future"
        after,_=detector.inspect(shifted);np.testing.assert_allclose(before,after)
        shifted=row.copy();shifted["supply_air_temperature_c"]=100
        distance,valid=detector.inspect(shifted)
        self.assertFalse(valid[0]);self.assertGreater(distance[0],before[0])

    def test_21_rounded_optimizer_repeatability(self):
        row=self.service.state(self.state_id)
        a=run_optimizer(row,seed=42,population_size=20,max_generations=8)
        b=run_optimizer(row,seed=42,population_size=20,max_generations=8)
        np.testing.assert_array_equal(a["best_x"],b["best_x"])
        self.assertEqual(a["feasible"],b["feasible"]);self.assertEqual(a["power"],b["power"])

    def test_22_real_models_reproduce_baseline_accuracy(self):
        for name,minimum in [("power",.94),("inlet",.93),("hotspot",.90),("load",.99)]:
            with self.subTest(model=name):self.assertGreater(self.service.model_report[name]["test"]["r2"],minimum)

    def test_23_invalid_optimizer_requests(self):
        for body in [{"state_id":self.state_id,"algorithm":"bad"},{"state_id":self.state_id,"seed":True},
                     {"state_id":self.state_id,"seed":-1},{"state_id":self.state_id,"seed":1.5},
                     {"state_id":self.state_id,"limits":{}}]:
            with self.subTest(body=body):self.assertEqual(self.post("/api/optimize",body).status_code,422)

    def test_24_problem_rejects_missing_and_nonfinite_candidate_shapes(self):
        row=self.service.state(self.state_id)
        with self.assertRaises(ValueError):CoolingOptimizationProblem(pd.concat([row,row]))
        problem=CoolingOptimizationProblem(row)
        for x in [np.array([1,2,3,4]),np.ones((2,3)),np.array([[1,2,3,np.nan]])]:
            with self.assertRaises(ValueError):problem._evaluate(x,{})

    def test_25_concurrent_policy_updates_have_one_winner(self):
        rev=self.service.get_limits(self.dc)["revision"]
        def update():
            with self.app.test_client() as client:
                return client.put("/api/limits/DC_1",json={"expected_revision":rev,"limits":default_limits()}).status_code
        with ThreadPoolExecutor(max_workers=2) as pool: results=list(pool.map(lambda _:update(),range(2)))
        self.assertEqual(sorted(results),[200,409])


    def test_26_all_centers_and_algorithms_with_restart_persistence(self):
        _,valid=OODDetector().inspect(self.service.frame)
        evidence=[]
        for dc in self.service.centers:
            ids=self.service.frame.index[valid & (self.service.frame.data_center_id==dc)]
            state_id=next(i for i in ids if self.service.predict({"state_id":i})["feasible"])
            for algorithm in ["GA","DE"]:
                with self.subTest(dc=dc,algorithm=algorithm):
                    response=self.post("/api/optimize",{"state_id":state_id,"algorithm":algorithm})
                    self.assertEqual(response.status_code,201)
                    run=response.json
                    self.assertTrue(run["recommendation"]["feasible"])
                    evidence.append({"data_center_id":dc,"algorithm":algorithm,"state_id":state_id,
                        "status":run["status"],"savings_pct":run["savings_pct"],"runtime_s":run["runtime_s"]})
        restored=create_app(self.temp.name).extensions["cooling"]
        self.assertEqual(restored.history()["total"],self.service.history()["total"])
        self.assertEqual(restored.get_limits("DC_1"),self.service.get_limits("DC_1"))
        from ml.config import RESULTS_DIR
        out=RESULTS_DIR/"verification"
        out.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(evidence).to_csv(out/"operator_scenarios.csv",index=False)

if __name__=="__main__":unittest.main(verbosity=2)

