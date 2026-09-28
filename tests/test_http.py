"""Loopback HTTP verification of every documented API operation."""
import sys, json, tempfile, threading, unittest
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from werkzeug.serving import make_server
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from api.app import create_app
from ml.config import RESULTS_DIR
from ml.optimization.ood_detector import OODDetector

class LiveHTTPTests(unittest.TestCase):
    def test_all_operations_over_real_http(self):
        evidence=[]
        with tempfile.TemporaryDirectory() as directory:
            app=create_app(directory);service=app.extensions["cooling"]
            server=make_server("127.0.0.1",0,app,threaded=True)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base=f"http://127.0.0.1:{server.server_port}"
            def request(method,path,body=None,status=200):
                data=None if body is None else json.dumps(body).encode()
                req=Request(base+path,data=data,method=method,headers={"Content-Type":"application/json"})
                try:
                    with urlopen(req,timeout=60) as response:
                        code=response.status;raw=response.read();content=response.headers.get_content_type()
                except HTTPError as error:
                    code=error.code;raw=error.read();content=error.headers.get_content_type()
                evidence.append({"method":method,"path":path,"expected":status,"actual":code,"passed":code==status})
                self.assertEqual(code,status,raw[:300])
                return json.loads(raw) if content=="application/json" else raw.decode()
            try:
                _,valid=OODDetector().inspect(service.frame)
                ids=service.frame.index[valid & (service.frame.data_center_id=="DC_1")]
                state_id=next(i for i in ids if service.predict({"state_id":i})["feasible"])
                request("GET","/")
                for path in ["/static/app.js","/static/styles.css","/api/health","/api/config","/api/states?dc=DC_1&limit=3","/api/states/"+state_id,"/api/trends?dc=DC_1&until="+state_id,"/api/datasets","/api/models","/api/docs","/api/openapi.json"]:
                    request("GET",path)
                policy=request("GET","/api/limits/DC_1")
                policy=request("PUT","/api/limits/DC_1",{"expected_revision":policy["revision"],"limits":policy["limits"]})
                request("POST","/api/predict",{"state_id":state_id})
                run=request("POST","/api/optimize",{"state_id":state_id,"algorithm":"GA"},201)
                self.assertEqual(run["status"],"feasible")
                request("POST","/api/optimize",{"state_id":state_id,"algorithm":"DE"},201)
                request("GET","/api/history?dc=DC_1")
                request("GET","/api/history/"+run["id"])
                request("POST","/api/recommendations/"+run["id"]+"/apply",{"state_id":state_id,"expected_revision":policy["revision"]})
                request("GET","/api/history.csv?dc=DC_1")
                request("POST","/api/predict",{"state_id":state_id,"controls":{}},422)
                request("GET","/api/states/missing",status=404)
                request("PUT","/api/limits/DC_1",{"expected_revision":policy["revision"]-1,"limits":policy["limits"]},409)
                operations={(r["method"],r["path"].split("?")[0]) for r in evidence}
                self.assertGreaterEqual(len(operations),17)
            finally:
                server.shutdown();thread.join(timeout=5);server.server_close()
        output=RESULTS_DIR/"verification";output.mkdir(parents=True,exist_ok=True)
        (output/"http_endpoint_results.json").write_text(json.dumps(evidence,indent=2),encoding="utf-8")
if __name__=="__main__":unittest.main(verbosity=2)

