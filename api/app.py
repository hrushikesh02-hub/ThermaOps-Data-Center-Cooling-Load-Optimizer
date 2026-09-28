"""Local operator dashboard and JSON API. Run via python main.py."""
from pathlib import Path
import sys, os, json, csv, io, logging
from functools import wraps
from flask import Flask, jsonify, request, render_template, Response
from werkzeug.exceptions import HTTPException
from api.service import CoolingService, InputError, ConflictError
from ml.optimization.fitness import DECISIONS

def create_app(state_dir=None):
    from ml.config import PROJECT_ROOT
    app=Flask(__name__,template_folder=str(PROJECT_ROOT/"frontend/templates"),
              static_folder=str(PROJECT_ROOT/"frontend/static"),static_url_path="/static")
    app.config.update(MAX_CONTENT_LENGTH=16*1024,JSON_SORT_KEYS=False)
    service=CoolingService(state_dir or os.getenv("COOLING_STATE_DIR"))
    app.extensions["cooling"]=service

    @app.before_request
    def local_origin():
        if request.method in ["POST","PUT","PATCH","DELETE"]:
            origin=request.headers.get("Origin")
            if origin and origin!=request.host_url.rstrip("/"):
                return jsonify(error="Cross-origin writes are not allowed"),403
        if request.path.startswith("/api/") and request.method in ["POST","PUT"]:
            if not request.is_json: return jsonify(error="Content-Type must be application/json"),415

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"]="nosniff"
        response.headers["Referrer-Policy"]="same-origin"
        response.headers["X-Frame-Options"]="SAMEORIGIN"
        response.headers["Content-Security-Policy"]="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'self'"
        if request.path.startswith("/api/"): response.headers["Cache-Control"]="no-store"
        return response

    @app.errorhandler(Exception)
    def errors(error):
        if isinstance(error,InputError): code=422
        elif isinstance(error,ConflictError): code=409
        elif isinstance(error,KeyError): code=404
        elif isinstance(error,HTTPException): code=error.code
        else:
            app.logger.exception("Unhandled request error")
            return jsonify(error="Internal error. Check the local server log."),500
        return jsonify(error=str(error).strip("'")),code

    def body():
        value=request.get_json()
        if not isinstance(value,dict): raise InputError("Request body must be a JSON object")
        return value

    def integer_arg(name,default,low,high):
        value=request.args.get(name,str(default))
        try: number=int(value)
        except (TypeError,ValueError): raise InputError(f"{name} must be an integer")
        if not low<=number<=high: raise InputError(f"{name} must be between {low} and {high}")
        return number

    @app.get("/")
    def index(): return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify(status="ready",mode="historical_replay",models_loaded=4,dataset_status=service.data_quality["status"],
            optimization_busy=service.optimize_lock.locked())

    @app.get("/api/config")
    def config():
        return jsonify(data_centers=service.centers,mode="historical_replay",decisions=DECISIONS,
            algorithms=["GA","DE"],control_precision=0.1,physical_limits={"max_inlet_c":27,"max_hotspot_c":35},
            uncertainty="Validation one-sided 95th-percentile residual plus operator reserve. An empirical allowance, not a hardware guarantee.",
            calibration_margin_c=service.base_margins)

    @app.get("/api/states")
    def states():
        dc=request.args.get("dc",service.centers[0]); service.require_dc(dc)
        limit=integer_arg("limit",100,1,500); offset=integer_arg("offset",0,0,100000)
        frame=service.frame[service.frame.data_center_id==dc].sort_values("timestamp",ascending=False)
        policy=service.get_limits(dc)
        frame=frame.join(service.snapshot_support(policy))
        cols=["state_id","timestamp","data_center_id","it_power_kw","cpu_utilization_pct","outdoor_temperature_c"]
        cols += ["baseline_feasible","baseline_region_valid","region_reachable","minimum_region_distance"]
        return jsonify(items=json.loads(frame.iloc[offset:offset+limit][cols].to_json(orient="records")),total=len(frame),limit=limit,offset=offset,
            policy_revision=policy["revision"],support_summary={"baseline_feasible":int(frame.baseline_feasible.sum()),
                "region_reachable":int(frame.region_reachable.sum()),"total":len(frame)})

    @app.get("/api/states/<state_id>")
    def state(state_id):
        row=service.state(state_id)
        result=json.loads(row.to_json(orient="records"))[0]
        policy=service.get_limits(str(row.iloc[0].data_center_id))
        return jsonify(state=result,baseline=service.predict({"state_id":state_id}),simulated=service.simulated_state(state_id),
            region_support=service.region_diagnostic(row,policy))

    @app.get("/api/trends")
    def trends():
        dc=request.args.get("dc",service.centers[0]); service.require_dc(dc)
        limit=integer_arg("limit",36,2,300)
        frame=service.frame[service.frame.data_center_id==dc].sort_values("timestamp")
        if "until" in request.args:
            until=service.state(request.args["until"]).iloc[0]
            if until.data_center_id!=dc: raise InputError("Trend state belongs to a different data center")
            frame=frame[frame.timestamp<=until.timestamp]
        cols=["timestamp","cooling_power_kw","server_inlet_temperature_c","hotspot_temperature_c","it_power_kw"]
        return jsonify(items=json.loads(frame.tail(limit)[cols].to_json(orient="records")),
            source="Recorded dataset measurements; each point is a historical observation.")

    @app.get("/api/limits/<dc>")
    def get_limits(dc): return jsonify(service.get_limits(dc))

    @app.put("/api/limits/<dc>")
    def put_limits(dc): return jsonify(service.set_limits(dc,body()))

    @app.post("/api/predict")
    def predict(): return jsonify(service.predict(body()))

    @app.post("/api/optimize")
    def optimize():
        payload=body()
        if not service.optimize_lock.acquire(blocking=False):
            return jsonify(error="An optimization is already running. Try again shortly."),429
        try: return jsonify(service.optimize(payload)),201
        finally: service.optimize_lock.release()

    @app.get("/api/history")
    def history():
        return jsonify(service.history(request.args.get("dc"),integer_arg("limit",50,1,500),integer_arg("offset",0,0,100000)))

    @app.get("/api/history/<run_id>")
    def run(run_id): return jsonify(service.run(run_id))

    @app.post("/api/recommendations/<run_id>/apply")
    def apply(run_id): return jsonify(service.apply(run_id,body()))

    @app.get("/api/history.csv")
    def export():
        dc=request.args.get("dc")
        data=[]; offset=0
        while True:
            page=service.history(dc,500,offset); data.extend(page["items"]); offset+=len(page["items"])
            if offset>=page["total"]: break
        stream=io.StringIO(newline="")
        writer=csv.DictWriter(stream,fieldnames=["id","created_at","data_center_id","state_id","algorithm","status","savings_kw","savings_pct","applied_at"])
        writer.writeheader()
        for item in data: writer.writerow({k:item[k] for k in writer.fieldnames})
        return Response(stream.getvalue(),mimetype="text/csv",headers={"Content-Disposition":"attachment; filename=cooling-history.csv"})

    @app.get("/api/datasets")
    def datasets(): return jsonify(service.data_quality)

    @app.get("/api/models")
    def models(): return jsonify(models=service.model_report,features=34,profile=service.model_profile,
        selection="Model family selected by held-out validation MAE; validation determines uncertainty allowances. Test data is evaluation/replay only.")

    @app.get("/api/docs")
    def docs(): return render_template("docs.html")

    @app.get("/api/openapi.json")
    def openapi():
        return app.send_static_file("openapi.json")

    return app

if __name__=="__main__":
    create_app().run(host="127.0.0.1",port=8501,debug=False)

