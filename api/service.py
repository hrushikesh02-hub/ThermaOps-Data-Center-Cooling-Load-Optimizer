"""Application services shared by HTTP routes and regression tests."""
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
from threading import Lock, RLock
from functools import lru_cache
import hashlib, json, sqlite3, uuid
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ml.config import DATA_DIR, MODELS_DIR, PROJECT_ROOT, ALL_FEATURES
from ml.optimization.fitness import CoolingOptimizationProblem, DECISIONS, get_surrogate_models
from ml.optimization.runner import run_optimizer
from ml.preprocessing.validate import audit_data
from ml.models.profile import model_frames, profile_summary

class InputError(ValueError):
    pass
class ConflictError(ValueError):
    pass

def number(value, name, low, high):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not low<=value<=high:
        raise InputError(f"{name} must be a finite number between {low} and {high}")
    return float(value)

def fields(body, allowed, required=()):
    if not isinstance(body,dict):
        raise InputError("Request body must be a JSON object")
    extra=set(body)-set(allowed)
    missing=set(required)-set(body)
    if extra: raise InputError(f"Unknown fields: {', '.join(sorted(extra))}")
    if missing: raise InputError(f"Missing fields: {', '.join(sorted(missing))}")
    return body

def default_limits():
    return {"max_inlet_c":27.0,"max_hotspot_c":35.0,"max_power_kw":None,"reserve_c":0.0,
        "bounds":{name:[18.0,27.0] if name.endswith("_c") else [30.0,100.0] for name in DECISIONS}}

def validate_limits(body):
    fields(body,default_limits(),default_limits())
    result={
        "max_inlet_c":number(body["max_inlet_c"],"max_inlet_c",18,27),
        "max_hotspot_c":number(body["max_hotspot_c"],"max_hotspot_c",20,35),
        "reserve_c":number(body["reserve_c"],"reserve_c",0,5),
        "max_power_kw":None if body["max_power_kw"] is None else number(body["max_power_kw"],"max_power_kw",1,10000)}
    if result["max_hotspot_c"]<result["max_inlet_c"]:
        raise InputError("Hotspot limit must be at least the inlet limit")
    fields(body["bounds"],DECISIONS,DECISIONS)
    result["bounds"]={}
    for name in DECISIONS:
        pair=body["bounds"][name]
        if not isinstance(pair,list) or len(pair)!=2: raise InputError(f"{name}: expected [minimum, maximum]")
        lo,hi=(18,27) if name.endswith("_c") else (30,100)
        pair=[number(v,name,lo,hi) for v in pair]
        if pair[0]>=pair[1]: raise InputError(f"{name}: minimum must be below maximum")
        if any(abs(v*10-round(v*10))>1e-7 for v in pair): raise InputError("Control bounds must use 0.1-unit increments")
        result["bounds"][name]=pair
    return result

def utcnow():
    return datetime.now(timezone.utc).isoformat()

class CoolingService:
    def __init__(self, state_dir=None):
        self.data_quality=audit_data()
        self.model_profile=profile_summary()
        self.model_revision=self.model_profile['revision']
        frames=model_frames()
        self.data_quality['original_splits']=self.data_quality['splits']
        self.data_quality['splits']={**self.data_quality['splits'],**self.model_profile['splits']}
        self.data_quality['model_profile']=self.model_profile
        self.data_quality['unknown_test_categories']={c:sorted(set(frames['test'][c].astype(str))-set(frames['train'][c].astype(str)))
            for c in ['data_center_id','season','time_of_day','day_of_week']}
        self.frame=pd.read_csv(DATA_DIR/"test.csv")
        self.frame["state_id"]=[hashlib.sha256(f"{r.data_center_id}|{r.timestamp}".encode()).hexdigest()[:16] for r in self.frame.itertuples()]
        self.frame=self.frame.set_index("state_id",drop=False)
        self.centers=sorted(self.frame.data_center_id.unique().tolist())
        self.db_path=Path(state_dir or PROJECT_ROOT/"outputs"/"runtime")/"operator.sqlite3"
        self.db_path.parent.mkdir(parents=True,exist_ok=True)
        self.db_lock=RLock()
        self.optimize_lock=Lock()
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS limits(dc TEXT PRIMARY KEY, revision INTEGER NOT NULL, body TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, dc TEXT NOT NULL, created_at TEXT NOT NULL, body TEXT NOT NULL, applied_at TEXT);
                CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, dc TEXT NOT NULL, event TEXT NOT NULL, created_at TEXT NOT NULL, body TEXT NOT NULL);
            """)
            for dc in self.centers:
                db.execute("INSERT OR IGNORE INTO limits VALUES(?,?,?,?)",(dc,1,json.dumps(default_limits()),utcnow()))
        self.models=get_surrogate_models()
        self.model_report=self.evaluate_models()
        self.base_margins={k:self.model_report[k]["validation_upper_residual_95"] for k in ["inlet","hotspot"]}
        from ml.optimization.ood_detector import OODDetector
        _, supported = OODDetector().inspect(self.frame)
        self.data_quality["test_baseline_region_support"]={"supported":int(supported.sum()),"total":len(supported),"percent":float(supported.mean()*100)}

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.db_path,timeout=10)
        db.row_factory=sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def evaluate_models(self):
        val=model_frames()['validation']
        test=model_frames()['test']
        report={}
        self.test_predictions={}
        for key,target,model in zip(["power","inlet","hotspot","load"],
            ["cooling_power_kw","server_inlet_temperature_c","hotspot_temperature_c","cooling_load_kw"],self.models):
            result={"target":target,"estimator":type(model.named_steps["model"]).__name__}
            for name,df in [("validation",val),("test",test)]:
                pred=model.predict(df[ALL_FEATURES])
                if name=="test": self.test_predictions[key]=np.asarray(pred,dtype=float)
                residual=df[target].to_numpy()-pred
                result[name]={"mae":float(mean_absolute_error(df[target],pred)),
                              "rmse":float(np.sqrt(mean_squared_error(df[target],pred))),
                              "r2":float(r2_score(df[target],pred)),"rows":len(df)}
                if name=="validation":
                    result["validation_upper_residual_95"]=float(max(0,np.quantile(residual,.95)))
                    result["validation_absolute_error_95"]=float(np.quantile(abs(residual),.95))
            report[key]=result
        return report

    @lru_cache(maxsize=32)
    def region_support(self, bound_pairs):
        from ml.optimization.ood_detector import OODDetector
        detector=OODDetector()
        _, distance, reachable=detector.closest_controls(self.frame,dict(zip(DECISIONS,bound_pairs)))
        baseline_distance, baseline_valid=detector.inspect(self.frame)
        return pd.DataFrame({"minimum_region_distance":distance,"region_reachable":reachable,
            "baseline_region_valid":baseline_valid},index=self.frame.index)

    def snapshot_support(self, policy):
        limits=policy["limits"]
        support=self.region_support(tuple(tuple(limits["bounds"][k]) for k in DECISIONS)).copy()
        feasible=support.baseline_region_valid.to_numpy().copy()
        for k in DECISIONS:
            lo,hi=limits["bounds"][k]
            feasible &= self.frame[k].between(lo,hi).to_numpy()
        for k in ["inlet","hotspot"]:
            feasible &= self.test_predictions[k]+self.base_margins[k]+limits["reserve_c"] <= limits["max_"+k+"_c"]
        if limits["max_power_kw"] is not None:
            feasible &= self.test_predictions["power"] <= limits["max_power_kw"]
        support["baseline_feasible"]=feasible
        return support

    def region_diagnostic(self, row, policy):
        from ml.optimization.ood_detector import OODDetector, PHYSICAL_FEATURES
        detector=OODDetector()
        support=self.snapshot_support(policy).loc[row.index[0]]
        z=detector.preprocessor.transform(row[PHYSICAL_FEATURES])[0]
        fixed=[i for i,k in enumerate(PHYSICAL_FEATURES) if k not in DECISIONS]
        largest=sorted(fixed,key=lambda i:abs(z[i]),reverse=True)[:3]
        return {"reachable":bool(support.region_reachable),
            "minimum_distance":float(support.minimum_region_distance),"threshold":detector.d_max,
            "largest_state_shifts":[{"feature":PHYSICAL_FEATURES[i],"value":float(row.iloc[0][PHYSICAL_FEATURES[i]]),
                "training_mean":float(detector.preprocessor.mean_[i]),"standard_deviations":float(z[i])} for i in largest]}

    def require_dc(self, dc):
        if dc not in self.centers: raise KeyError("Unknown data center")
        return dc

    def state(self,state_id):
        if not isinstance(state_id,str): raise InputError("state_id must be a string")
        if state_id not in self.frame.index: raise KeyError("Operating state not found")
        return self.frame.loc[[state_id]].copy()

    def get_limits(self, dc):
        self.require_dc(dc)
        with self.connect() as db:
            r=db.execute("SELECT * FROM limits WHERE dc=?",(dc,)).fetchone()
        return {"data_center_id":dc,"revision":r["revision"],"updated_at":r["updated_at"],"limits":json.loads(r["body"])}

    def set_limits(self,dc,body):
        fields(body,["expected_revision","limits"],["expected_revision","limits"])
        rev=body["expected_revision"]
        if type(rev) is not int or rev<1: raise InputError("expected_revision must be a positive integer")
        limits=validate_limits(body["limits"])
        self.require_dc(dc)
        with self.db_lock,self.connect() as db:
            cur=db.execute("UPDATE limits SET revision=revision+1, body=?,updated_at=? WHERE dc=? AND revision=?",
                (json.dumps(limits),utcnow(),dc,rev))
            if not cur.rowcount: raise ConflictError("Limits changed in another session. Reload and try again.")
            db.execute("INSERT INTO events(dc,event,created_at,body) VALUES(?,?,?,?)",
                (dc,"limits_updated",utcnow(),json.dumps({"revision":rev+1,"limits":limits})))
        return self.get_limits(dc)

    def problem(self,row,policy):
        reserve=policy["limits"]["reserve_c"]
        margins={k:v+reserve for k,v in self.base_margins.items()}
        return CoolingOptimizationProblem(row,limits=policy["limits"],margins=margins)

    def prediction(self,row,controls,policy):
        problem=self.problem(row,policy)
        x=np.array([[controls[k] for k in DECISIONS]],dtype=float)
        out={}; problem._evaluate(x,out)
        reasons=[]
        limits=policy["limits"]
        for k in DECISIONS:
            lo,hi=limits["bounds"][k]
            if not lo<=controls[k]<=hi: reasons.append(f"{k} is outside operator bounds")
        if out["inlet"][0]+problem.margins["inlet"]>limits["max_inlet_c"]: reasons.append("Inlet upper estimate exceeds the limit")
        if out["hotspot"][0]+problem.margins["hotspot"]>limits["max_hotspot_c"]: reasons.append("Hotspot upper estimate exceeds the limit")
        if limits["max_power_kw"] is not None and out["power"][0]>limits["max_power_kw"]: reasons.append("Predicted cooling power exceeds the cap")
        if not bool(out["ood_valid"][0]): reasons.append("Outside the learned physical operating region")
        values={k:float(out[k][0]) for k in ["power","inlet","hotspot","load"]}
        return {"controls":controls,"predictions":values,
            "upper_estimates_c":{k:values[k]+problem.margins[k] for k in ["inlet","hotspot"]},
            "uncertainty_margin_c":problem.margins,
            "ood":{"distance":float(out["ood_distance"][0]),"threshold":problem.ood_detector.d_max,"valid":bool(out["ood_valid"][0])},
            "feasible":not reasons,"violations":reasons,"policy_revision":policy["revision"],
            "unknown_season":str(row.iloc[0].season) in self.data_quality["unknown_test_categories"]["season"]}

    def predict(self,body):
        fields(body,["state_id","controls"],["state_id"])
        row=self.state(body["state_id"])
        policy=self.get_limits(str(row.iloc[0].data_center_id))
        controls={k:float(row.iloc[0][k]) for k in DECISIONS}
        if "controls" in body:
            fields(body["controls"],DECISIONS,DECISIONS)
            for k in DECISIONS:
                # Historical baseline can lie outside the configured recommendation envelope.
                lo,hi=(10,40) if k.endswith("_c") else (0,100)
                controls[k]=number(body["controls"][k],k,lo,hi)
        return self.prediction(row,controls,policy)

    def optimize(self,body):
        fields(body,["state_id","algorithm","seed"],["state_id"])
        algorithm=body.get("algorithm","GA")
        if algorithm not in ["GA","DE"]: raise InputError("algorithm must be GA or DE")
        seed=body.get("seed",42)
        if type(seed) is not int or not 0<=seed<=2147483647: raise InputError("seed must be an integer between 0 and 2147483647")
        row=self.state(body["state_id"])
        dc=str(row.iloc[0].data_center_id)
        policy=self.get_limits(dc)
        baseline_controls={k:float(row.iloc[0][k]) for k in DECISIONS}
        baseline=self.prediction(row,baseline_controls,policy)
        result=run_optimizer(row,algorithm,seed=seed,limits=policy["limits"],
            margins={k:v+policy["limits"]["reserve_c"] for k,v in self.base_margins.items()},
            population_size=40,max_generations=35)
        controls={k:float(v) for k,v in zip(DECISIONS,result["best_x"])}
        diagnostic=self.prediction(row,controls,policy)
        feasible=result["feasible"] and diagnostic["feasible"]
        region=self.region_diagnostic(row,policy)
        if result["region_blocked"]:
            message=(f"This snapshot cannot enter the learned operating region within the saved control bounds: "
                f"minimum possible distance {region['minimum_distance']:.2f}, allowed {region['threshold']:.2f}. "
                "Changing temperature limits or running a longer search cannot resolve this operating-region blocker. "
                "Choose a snapshot that passes checks. Broader coverage requires representative training data and revalidation.")
        else:
            message="No feasible recommendation found within this search budget. Review the reported thermal, power and control limits."
        savings=baseline["predictions"]["power"]-diagnostic["predictions"]["power"]
        record={"id":str(uuid.uuid4()),"created_at":utcnow(),"data_center_id":dc,"state_id":body["state_id"],
            "timestamp":str(row.iloc[0].timestamp),"mode":"simulation","algorithm":algorithm,"seed":seed,
            "policy":policy,"model_revision":self.model_revision,"model_profile":self.model_profile['name'],
            "baseline":baseline,"status":"feasible" if feasible else "infeasible",
            "recommendation":diagnostic if feasible else None,
            "diagnostic":None if feasible else {"violations":diagnostic["violations"],"message":message,
                "code":"unsupported_operating_state" if result["region_blocked"] else "search_exhausted",
                "region":region,"candidate_assessment":diagnostic},
            "savings_kw":savings if feasible else None,
            "savings_pct":100*savings/baseline["predictions"]["power"] if feasible else None,
            "runtime_s":result["runtime_s"],"evaluations":result["evaluations"],"convergence_kw":result["history"],
            "applied_at":None,"stale":False}
        with self.db_lock,self.connect() as db:
            # Policy may have changed during a long optimization.
            revision=db.execute("SELECT revision FROM limits WHERE dc=?",(dc,)).fetchone()[0]
            record["stale"]=revision!=policy["revision"]
            db.execute("INSERT INTO runs VALUES(?,?,?,?,?)",(record["id"],dc,record["created_at"],json.dumps(record,allow_nan=False),None))
        return record

    def history(self,dc=None,limit=50,offset=0):
        if dc: self.require_dc(dc)
        with self.connect() as db:
            where=" WHERE dc=?" if dc else ""
            args=[dc] if dc else []
            count=db.execute("SELECT count(*) FROM runs"+where,args).fetchone()[0]
            rows=db.execute("SELECT body,applied_at FROM runs"+where+" ORDER BY created_at DESC LIMIT ? OFFSET ?",args+[limit,offset]).fetchall()
        items=[]
        for row in rows:
            record=json.loads(row["body"])
            record["applied_at"]=row["applied_at"]
            record["stale"]=(self.get_limits(record["data_center_id"])["revision"]!=record["policy"]["revision"]
                or record.get('model_revision','legacy')!=self.model_revision)
            items.append(record)
        return {"items":items,"total":count,"limit":limit,"offset":offset}

    def run(self,run_id):
        with self.connect() as db:
            row=db.execute("SELECT body,applied_at FROM runs WHERE id=?",(run_id,)).fetchone()
        if row is None: raise KeyError("Optimization run not found")
        record=json.loads(row["body"])
        record["applied_at"]=row["applied_at"]
        record["stale"]=(self.get_limits(record["data_center_id"])["revision"]!=record["policy"]["revision"]
            or record.get('model_revision','legacy')!=self.model_revision)
        return record

    def apply(self,run_id,body):
        fields(body,["state_id","expected_revision"],["state_id","expected_revision"])
        if not isinstance(body["state_id"],str) or type(body["expected_revision"]) is not int:
            raise InputError("state_id must be a string and expected_revision an integer")
        with self.db_lock:
            record=self.run(run_id)
            policy=self.get_limits(record["data_center_id"])
            if body["state_id"]!=record["state_id"]: raise ConflictError("Operating state changed. Generate a new recommendation.")
            if body["expected_revision"]!=policy["revision"] or record["stale"]: raise ConflictError("Limits or model version changed. Generate a new recommendation.")
            if record["recommendation"] is None: raise ConflictError("This run has no feasible recommendation")
            if record["applied_at"]: return record
            check=self.prediction(self.state(record["state_id"]),record["recommendation"]["controls"],policy)
            if not check["feasible"]: raise ConflictError("Recommendation no longer passes validation")
            timestamp=utcnow()
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                current_revision=db.execute("SELECT revision FROM limits WHERE dc=?",(record["data_center_id"],)).fetchone()[0]
                if current_revision!=policy["revision"]:
                    raise ConflictError("Limits changed. Generate a new recommendation.")
                existing=db.execute("SELECT applied_at FROM runs WHERE id=?",(run_id,)).fetchone()[0]
                if existing:
                    return self.run(run_id)
                db.execute("UPDATE runs SET applied_at=? WHERE id=?",(timestamp,run_id))
                db.execute("INSERT INTO events(dc,event,created_at,body) VALUES(?,?,?,?)",
                    (record["data_center_id"],"simulation_setpoints_applied",timestamp,
                     json.dumps({"run_id":run_id,"state_id":record["state_id"],"controls":check["controls"]})))
        return self.run(run_id)

    def simulated_state(self,state_id):
        row=self.state(state_id)
        dc=str(row.iloc[0].data_center_id)
        with self.connect() as db:
            entries=db.execute("SELECT body FROM events WHERE dc=? AND event='simulation_setpoints_applied' ORDER BY id DESC",(dc,)).fetchall()
        for entry in entries:
            value=json.loads(entry["body"])
            if value["state_id"]==state_id: return value
        return None

