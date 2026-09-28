"use strict";
const $ = id => document.getElementById(id);
const names = {cooling_setpoint_c:"Cooling target", supply_air_temperature_c:"Supply air temperature", fan_speed_pct:"Fan speed", pump_speed_pct:"Pump speed"};
const units = key => key.endsWith("_c") ? "°C" : "%";
const fmt = (v,d=1) => Number.isFinite(v) ? v.toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}) : "—";
const esc = value => String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const displayLabels = {
  ...names, cooling_power_kw:"Cooling power (kW)", cooling_load_kw:"Cooling load (kW)",
  server_inlet_temperature_c:"Air entering servers (°C)", hotspot_temperature_c:"Rack hot spot (°C)",
  outdoor_humidity_pct:"Outdoor humidity (%)", indoor_humidity_pct:"Indoor humidity (%)",
  outdoor_temperature_c:"Outdoor temperature (°C)", indoor_temperature_c:"Indoor temperature (°C)",
  server_count:"Number of servers", cpu_utilization_pct:"CPU use (%)", gpu_utilization_pct:"GPU use (%)",
  memory_utilization_pct:"Memory use (%)", network_load_gbps:"Network traffic (Gbps)",
  storage_io_load_pct:"Storage activity (%)", it_power_kw:"IT equipment power (kW)",
  rack_density_kw:"Power per rack (kW)", airflow_cfm:"Airflow (cfm)",
  max_inlet_c:"Server air temperature limit", max_hotspot_c:"Rack hot spot temperature limit",
  reserve_c:"Extra temperature allowance", max_power_kw:"Maximum cooling power",
  RandomForestRegressor:"Random Forest", XGBRegressor:"XGBoost", LGBMRegressor:"LightGBM",
  LinearRegression:"Linear Regression", DummyRegressor:"Reference average",
  train:"Used to learn", validation:"Used to check", test:"Used for final testing",
  seasonal_v1:"Seasonal model, version 1", legacy:"Original model",
  GA:"Method 1 (GA)", DE:"Method 2 (DE)"
};
const displayLabel = value => displayLabels[value] || String(value);
function operatorText(value){
  let text=String(value);
  const messages={
    "Inlet upper estimate exceeds the limit":"Server air temperature, including the allowance, is above your limit",
    "Hotspot upper estimate exceeds the limit":"Rack hot spot temperature, including the allowance, is above your limit",
    "Predicted cooling power exceeds the cap":"Estimated cooling power is above your limit",
    "Outside the learned physical operating region":"The model does not cover these operating conditions",
    "Hotspot limit must be at least the inlet limit":"The rack hot spot limit must be at least as high as the server air temperature limit",
    "Control bounds must use 0.1-unit increments":"Enter setting limits with no more than one decimal place",
    "Operating state not found":"This record could not be found. Select another date and time.",
    "Optimization run not found":"This saved recommendation could not be found.",
    "Operating state changed. Generate a new recommendation.":"A different record is selected. Generate a new recommendation.",
    "Limits or model version changed. Generate a new recommendation.":"The limits or prediction model have changed. Generate a new recommendation.",
    "This run has no feasible recommendation":"This result has no settings that meet your limits.",
    "Recommendation no longer passes validation":"These settings no longer pass the checks. Generate a new recommendation.",
    "An optimization is already running. Try again shortly.":"A recommendation is already being prepared. Please try again shortly.",
    "Internal error. Check the local server log.":"Something went wrong. Please try again or ask your support team to check the application log.",
    "Safety margins use 27 C inlet / 35 C hotspot. No measurement or target values changed.":"Temperature margins use limits of 27°C for server air and 35°C for rack hot spots. Recorded measurements are unchanged.",
    "Provided research dataset; source instrumentation and generation process were not supplied.":"These records were supplied for research. Details of how they were collected were not provided.",
    "No feasible recommendation found within this search budget. Review the operating state and limits.":"No settings meeting your limits were found in this search. Review the selected record and limits.",
    "No feasible recommendation found within this search budget. Review the reported thermal, power and control limits.":"No settings meeting your limits were found in this search. Review the temperature, power and setting limits shown."
  };
  if(messages[text])return messages[text];
  text=text.replace("This snapshot cannot enter the learned operating region within the saved control bounds:","The model does not cover these recorded conditions, even with different allowed settings:")
    .replace("minimum possible distance","lowest difference score")
    .replace("Changing temperature limits or running a longer search cannot resolve this operating-region blocker.","Changing temperature limits or searching longer will not solve this data coverage issue.")
    .replace("Choose a snapshot that passes checks.","Choose a record that passes checks.")
    .replace("Broader coverage requires representative training data and revalidation.","The model needs more training and testing with these conditions.")
    .replace("is outside operator bounds","is outside your allowed range")
    .replace("must be a finite number between","must be a number between")
    .replace("minimum must be below maximum","the minimum must be lower than the maximum");
  for(const [key,label] of Object.entries(displayLabels)){
    if(key.includes("_"))text=text.replaceAll(key,label);
  }
  return text;
}
let config, policy, current, run=null, dirty=false, busy=false, view="overview", epoch=0, historyOffset=0, historyTotal=0, toastTimer;
async function api(path, options={}){
  const controller=new AbortController();
  const timer=setTimeout(()=>controller.abort(),options.timeout || 120000);
  try{
    const response=await fetch(path,{...options,signal:controller.signal,headers:{"Content-Type":"application/json",...(options.headers||{})}});
    const data=await response.json();
    if(!response.ok) throw new Error(data.error||"Request failed ("+response.status+")");
    return data;
  } catch(error){if(error.name==="AbortError")throw new Error("This is taking longer than expected. Check the connection and recommendation history before trying again.");throw error;}
  finally{clearTimeout(timer);}
}
function toast(text){$("toast").textContent=text;$("toast").classList.remove("hidden");clearTimeout(toastTimer);toastTimer=setTimeout(()=>$("toast").classList.add("hidden"),5000);}
function failure(error){$("notice").textContent=operatorText(error.message);$("notice").classList.remove("hidden");}
function clearError(){$("notice").classList.add("hidden");}
function badge(text,type="neutral"){return '<span class="badge '+type+'">'+esc(text)+'</span>';}
function setBusy(value){
  busy=value;
  for(const id of ["dc","snapshot","algorithm","optimize","preview","reset-limits","supported-snapshot"]) $(id).disabled=value || !current;
  $("supported-snapshot").disabled=value||dirty||!current;
  $("optimize").disabled=value||dirty||!current;
  $("save-limits").disabled=value||!dirty;
  document.querySelectorAll("#limits-form input, #manual-form input").forEach(input=>input.disabled=value);
  document.querySelectorAll("#apply").forEach(button=>button.disabled=value||dirty||Boolean(run?.stale)||Boolean(run?.applied_at));
}
function markDirty(){
  dirty=true;$("limit-state").textContent="Unsaved changes";$("limit-state").className="badge warning";
  $("save-limits").disabled=busy;$("optimize").disabled=true;
  $("supported-snapshot").disabled=true;
  if(run){$("recommendation-state").textContent="Limits changed";$("recommendation-state").className="badge warning";const apply=$("apply");if(apply)apply.disabled=true;}
}
function populateLimits(){
  const limits=policy.limits;
  $("max-inlet").value=limits.max_inlet_c;$("max-hotspot").value=limits.max_hotspot_c;
  $("reserve").value=limits.reserve_c;$("power-cap").value=limits.max_power_kw??"";
  $("bounds").innerHTML=Object.keys(names).map(k=>'<div class="bound-row"><span>'+names[k]+' ('+units(k)+')</span><input aria-label="'+names[k]+' minimum" id="min-'+k+'" type="number" step="0.1" min="'+(k.endsWith("_c")?18:30)+'" max="'+(k.endsWith("_c")?27:100)+'" value="'+limits.bounds[k][0]+'" required><input aria-label="'+names[k]+' maximum" id="max-'+k+'" type="number" step="0.1" min="'+(k.endsWith("_c")?18:30)+'" max="'+(k.endsWith("_c")?27:100)+'" value="'+limits.bounds[k][1]+'" required></div>').join("");
  $("margin-note").textContent="Allowance for prediction error: +"+fmt(config.calibration_margin_c.inlet,2)+"°C server air / +"+fmt(config.calibration_margin_c.hotspot,2)+"°C rack hot spot, plus your extra allowance.";
  $("revision").textContent="Saved limits · version "+policy.revision;
  $("limit-state").textContent="Saved";$("limit-state").className="badge neutral";
  dirty=false;setBusy(busy);
}
function resetRecommendation(){
  run=null;$("recommendation-state").textContent="Ready to start";$("recommendation-state").className="badge neutral";
  $("recommendation").innerHTML='<div class="empty-state"><div class="empty-icon">⌁</div><h3>Get suggested cooling settings</h3><p>Get suggested settings for this record using your saved limits. Each suggestion is checked against temperature limits and the conditions covered by the model.</p></div>';
}
function kpi(label,value,unit,foot,tag="",type="success"){
  return '<div class="kpi"><div class="kpi-label">'+label+'<span>↗</span></div><div class="kpi-value">'+value+'<small>'+unit+'</small></div><div class="kpi-foot">'+(tag?'<span class="mini-tag '+type+'">'+tag+'</span>':"")+foot+'</div></div>';
}
function renderSnapshot(){
  const s=current.state,b=current.baseline,p=b.predictions,l=policy.limits;
  const support=current.region_support;
  $("snapshot-support-note").textContent=support&&!support.reachable
    ? "The model does not cover these recorded conditions, even with different allowed settings (lowest difference score "+fmt(support.minimum_distance,2)+"; limit "+fmt(support.threshold,2)+"). Changing temperature limits will not solve this data coverage issue."
    : b.feasible?"These recorded settings meet your saved limits and are covered by the model."
    : "The model can cover these conditions with different settings. Generate a recommendation to check your temperature limits, allowances and power limit.";
  $("replay-date").textContent=s.data_center_id+" · "+s.timestamp+" · Recorded conditions";
  const inletHead=l.max_inlet_c-b.upper_estimates_c.inlet;
  const hotHead=l.max_hotspot_c-b.upper_estimates_c.hotspot;
  const head=Math.min(inletHead,hotHead);
  $("kpis").innerHTML=kpi("Estimated cooling power",fmt(p.power),"kW","Measured: "+fmt(s.cooling_power_kw)+" kW","ESTIMATE")
    +kpi("Temperature margin",fmt(head),"°C","Includes the error allowance",head>=0?"WITHIN LIMIT":"LIMIT EXCEEDED",head>=0?"success":"danger")
    +kpi("IT equipment power",fmt(s.it_power_kw,0),"kW","CPU "+fmt(s.cpu_utilization_pct,0)+"% · GPU "+fmt(s.gpu_utilization_pct,0)+"%")
    +kpi("Model data coverage",b.ood.valid?"Covered":"Review","","Difference score "+fmt(b.ood.distance,2)+" / limit "+fmt(b.ood.threshold,2),b.ood.valid?"COVERED":"NOT COVERED",b.ood.valid?"success":"warning");
  $("thermal-status").innerHTML=[["inlet","Air entering servers",l.max_inlet_c],["hotspot","Rack hot spot",l.max_hotspot_c]].map(([k,label,lim])=>{
    const upper=b.upper_estimates_c[k],head=lim-upper,type=head<0?"danger":head<1?"warning":"";
    return '<div class="thermal-item"><div class="thermal-title"><span>'+label+'</span><b>'+fmt(p[k])+'°C '+badge(head>=0?"Within limit":"Over limit",head>=0?"success":"danger")+'</b></div><div class="meter '+type+'"><div class="meter-fill" data-percent="'+Math.min(100,upper/lim*100)+'"></div></div><div class="thermal-detail"><span>With allowance '+fmt(upper)+'°C</span><span>Limit '+fmt(lim)+'°C</span></div></div>';
  }).join("");
  document.querySelectorAll(".meter-fill").forEach(el=>el.style.width=el.dataset.percent+"%");
  $("region-status").innerHTML='<b>'+badge(b.feasible?"Recorded settings pass all checks":"Recorded settings need review",b.feasible?"success":"warning")+'</b><span>'+fmt(b.ood.distance,2)+' difference score</span>';
  $("region-status").title=b.violations.map(operatorText).join("; ");
  $("manual-inputs").innerHTML=Object.keys(names).map(k=>'<label>'+names[k]+' <span>'+units(k)+'</span><input aria-label="'+names[k]+' preview" id="manual-'+k+'" type="number" step="0.01" min="'+(k.endsWith("_c")?10:0)+'" max="'+(k.endsWith("_c")?40:100)+'" value="'+fmtInput(current.simulated?.controls[k]??s[k])+'" required></label>').join("");
  $("manual-result").innerHTML=current.simulated?'<div class="preview-result">'+badge("Simulation settings saved","success")+'<span>The fields show your saved simulation settings. The original recorded measurements are shown above.</span></div>':"";
}
function fmtInput(value){return Number(value).toFixed(2);}
function drawTrend(items){
  if(!items.length){$("power-chart").textContent="No records for this time period.";return;}
  const width=680,height=200,left=47,right=18,top=14,bottom=34;
  const values=items.map(r=>r.cooling_power_kw);
  const min=Math.max(0,Math.floor(Math.min(...values)*.85/10)*10);
  const max=Math.ceil(Math.max(...values)*1.1/10)*10 || 10;
  const x=i=>left+i/Math.max(1,items.length-1)*(width-left-right);
  const y=v=>top+(max-v)/(max-min)*(height-top-bottom);
  const points=values.map((v,i)=>x(i)+","+y(v)).join(" ");
  let grid="";
  for(let i=0;i<=4;i++){const v=min+(max-min)*i/4;grid+='<line class="grid-line" x1="'+left+'" y1="'+y(v)+'" x2="'+(width-right)+'" y2="'+y(v)+'"/><text x="'+(left-9)+'" y="'+(y(v)+3)+'" text-anchor="end">'+Math.round(v)+'</text>';}
  const indexes=[...new Set([0,Math.floor((items.length-1)/3),Math.floor((items.length-1)*2/3),items.length-1])];
  const labels=indexes.map(i=>'<text x="'+x(i)+'" y="'+(height-10)+'" text-anchor="'+(i===0?"start":i===items.length-1?"end":"middle")+'">'+esc(items[i].timestamp.slice(5,16))+'</text>').join("");
  $("power-chart").innerHTML='<svg viewBox="0 0 '+width+' '+height+'" role="img" aria-label="Measured cooling power in kilowatts by date and time"><title>Measured cooling power (kW) by date and time</title>'+grid+'<polygon class="area" points="'+x(0)+','+y(min)+' '+points+' '+x(items.length-1)+','+y(min)+'"/><polyline class="trace" points="'+points+'"/>'+values.map((v,i)=>'<circle cx="'+x(i)+'" cy="'+y(v)+'" r="2.3" fill="#149b9f"><title>'+esc(items[i].timestamp)+': '+fmt(v)+' kW</title></circle>').join("")+labels+'<text x="5" y="13">kW</text></svg>';
  $("trend-caption").textContent=items.length+" records · "+items[0].timestamp.slice(0,10)+" to "+items.at(-1).timestamp.slice(0,10);
}
async function loadCenter(){
  const token=++epoch,dc=$("dc").value;setBusy(true);clearError();
  try{
    const [p,list]=await Promise.all([api("/api/limits/"+encodeURIComponent(dc)),api("/api/states?dc="+encodeURIComponent(dc)+"&limit=500")]);
    if(token!==epoch)return;
    policy=p;
    renderSnapshotOptions(list);
    historyOffset=0;$("export").href="/api/history.csv?dc="+encodeURIComponent(dc);
    await loadSnapshot();
    if(view==="history")await loadHistory();if(view==="data")await loadData();
  }catch(error){current=null;failure(error);}
  finally{setBusy(false);}
}
function renderSnapshotOptions(list,selected){
  $("snapshot").innerHTML=list.items.map(item=>'<option value="'+item.state_id+'">'+esc(item.timestamp)+(item.baseline_feasible?" · Passes checks":!item.region_reachable?" · Model does not cover this record":" · Try different settings")+'</option>').join("");
  if(selected)$("snapshot").value=selected;
}
async function useSupportedSnapshot(){
  if(busy||dirty)return;
  clearError();setBusy(true);
  try{
    const list=await api("/api/states?dc="+encodeURIComponent($("dc").value)+"&limit=500");
    const candidate=list.items.find(item=>item.baseline_feasible);
    if(!candidate)throw new Error("No recorded settings meet all your saved limits at this data center. Review the limits or choose another data center. You can also generate a recommendation for a record labelled Try different settings.");
    renderSnapshotOptions(list,candidate.state_id);
    await loadSnapshot();
    toast("Selected the latest record that passes your checks. Generate a recommendation to check the suggested settings.");
  }catch(error){failure(error);}
  finally{setBusy(false);}
}
async function loadSnapshot(){
  const token=++epoch,stateId=$("snapshot").value;
  if(!stateId){current=null;throw new Error("No records are available for this data center.");}
  setBusy(true);clearError();resetRecommendation();
  try{
    const [data,trend,p]=await Promise.all([api("/api/states/"+stateId),api("/api/trends?dc="+encodeURIComponent($("dc").value)+"&until="+stateId),api("/api/limits/"+encodeURIComponent($("dc").value))]);
    if(token!==epoch)return;
    current=data;policy=p;populateLimits();renderSnapshot();drawTrend(trend.items);
  }catch(error){current=null;failure(error);}
  finally{setBusy(false);}
}
async function saveLimits(event){
  event.preventDefault();clearError();setBusy(true);
  const limits={max_inlet_c:Number($("max-inlet").value),max_hotspot_c:Number($("max-hotspot").value),reserve_c:Number($("reserve").value),max_power_kw:$("power-cap").value===""?null:Number($("power-cap").value),bounds:{}};
  for(const key of Object.keys(names))limits.bounds[key]=[Number($("min-"+key).value),Number($("max-"+key).value)];
  try{
    policy=await api("/api/limits/"+encodeURIComponent($("dc").value),{method:"PUT",body:JSON.stringify({expected_revision:policy.revision,limits})});
    populateLimits();resetRecommendation();
    current=await api("/api/states/"+$("snapshot").value);renderSnapshot();
    renderSnapshotOptions(await api("/api/states?dc="+encodeURIComponent($("dc").value)+"&limit=500"),current.state.state_id);
    toast("Limits saved for "+policy.data_center_id+". Generate a new recommendation.");
  }catch(error){failure(error);}
  finally{setBusy(false);}
}
function recommendationHtml(record,allowApply=true){
  if(!record.recommendation){
    const blocked=record.diagnostic.code==="unsupported_operating_state";
    const shifts=record.diagnostic.region?.largest_state_shifts||[];
    const details=blocked&&shifts.length?'<p>Conditions that differ most from the model’s training data: '+shifts.map(v=>esc(displayLabel(v.feature))+" "+fmt(v.value)+" (training average "+fmt(v.training_mean)+")").join("; ")+". These recorded conditions stay the same while settings are tested.</p>":"";
    return '<div class="infeasible"><h3>'+(blocked?"Model does not cover these conditions":"No settings found within your limits")+'</h3><p>'+esc(operatorText(record.diagnostic.message))+'</p>'+details+'<ul>'+record.diagnostic.violations.map(v=>'<li>'+esc(operatorText(v))+'</li>').join("")+'</ul><p>'+(blocked?"Data coverage check":"Search")+' completed in '+fmt(record.runtime_s,2)+' s · '+record.evaluations+' settings checked.'+(blocked?"":" Other settings may still meet your limits.")+'</p></div>';
  }
  const rec=record.recommendation,p=rec.predictions,base=record.baseline;
  return '<div class="recommendation-result"><div class="savings"><div><div class="savings-value '+(record.savings_pct<0?"warning-text":"")+'">'+(record.savings_pct>=0?"−":"+")+fmt(Math.abs(record.savings_pct))+'<small>% estimated power change</small></div><p>'+fmt(base.predictions.power)+' → '+fmt(p.power)+' kW · '+(record.savings_kw>=0?"reduction":"increase")+' of '+fmt(Math.abs(record.savings_kw))+' kW</p></div>'+badge(record.savings_kw>=0?"Meets your limits":"Higher power to meet limits",record.savings_kw>=0?"success":"warning")+'</div>'
  +'<table class="controls-table"><thead><tr><th>Setting</th><th>Recorded</th><th>Recommended</th></tr></thead><tbody>'+Object.keys(names).map(k=>'<tr><td>'+names[k]+'</td><td>'+fmt(base.controls[k])+' '+units(k)+'</td><td><b>'+fmt(rec.controls[k])+' '+units(k)+'</b></td></tr>').join("")+'</tbody></table>'
  +'<p class="field-note">Temperatures with allowance: server air '+fmt(rec.upper_estimates_c.inlet)+'°C / '+fmt(record.policy.limits.max_inlet_c)+'°C; rack hot spot '+fmt(rec.upper_estimates_c.hotspot)+'°C / '+fmt(record.policy.limits.max_hotspot_c)+'°C. Data difference score '+fmt(rec.ood.distance,2)+' / limit '+fmt(rec.ood.threshold,2)+'.</p>'
  +'<div class="result-footer"><p>'+esc(displayLabel(record.algorithm))+' · '+fmt(record.runtime_s,2)+' s · '+record.evaluations+' settings checked<br>Rounded to one decimal place and checked again</p>'
  +(allowApply?'<button id="apply" class="button primary" '+(record.stale||record.applied_at||dirty?"disabled":"")+'>'+(record.applied_at?"Applied to simulation":record.stale?"Outdated — generate again":"Apply to simulation")+'</button>':badge(record.applied_at?"Applied to simulation":record.stale?"Needs a new check":"Reviewed",record.applied_at?"success":"neutral"))+'</div></div>';
}
function renderRun(){
  $("recommendation-state").textContent=run.stale?"Outdated":run.status==="feasible"?"Checks passed":"Review required";
  $("recommendation-state").className="badge "+(run.status==="feasible"&&!run.stale?"success":"warning");
  $("recommendation").innerHTML=recommendationHtml(run);
  if($("apply"))$("apply").addEventListener("click",applyRun);
}
async function optimize(){
  if(dirty||busy)return;
  clearError();setBusy(true);
  $("recommendation-state").textContent="Searching";$("recommendation-state").className="badge neutral";
  $("recommendation").innerHTML='<div class="loading">Finding settings within your limits…<p class="field-note">Checking temperatures, allowed settings and model data coverage.</p></div>';
  try{
    run=await api("/api/optimize",{method:"POST",body:JSON.stringify({state_id:$("snapshot").value,algorithm:$("algorithm").value,seed:42})});
    renderRun();toast(run.recommendation?"Recommendation ready for review.":"No settings found within your limits. Review the reasons shown.");
  }catch(error){resetRecommendation();failure(error);}
  finally{setBusy(false);if(run)renderRun();}
}
async function applyRun(){
  if(!run||dirty||run.stale||busy)return;
  clearError();setBusy(true);
  try{
    run=await api("/api/recommendations/"+run.id+"/apply",{method:"POST",body:JSON.stringify({state_id:$("snapshot").value,expected_revision:policy.revision})});
    current=await api("/api/states/"+$("snapshot").value);
    renderSnapshot();toast("Settings applied to this record in the simulation.");
  }catch(error){failure(error);}
  finally{setBusy(false);renderRun();}
}
async function preview(event){
  event.preventDefault();clearError();setBusy(true);
  const controls=Object.fromEntries(Object.keys(names).map(k=>[k,Number($("manual-"+k).value)]));
  try{
    const data=await api("/api/predict",{method:"POST",body:JSON.stringify({state_id:$("snapshot").value,controls})});
    $("manual-result").innerHTML='<div class="preview-result">'+badge(data.feasible?"Passes saved limits":"Limits not met",data.feasible?"success":"warning")+'<span>Cooling power <b>'+fmt(data.predictions.power)+' kW</b></span><span>Server air <b>'+fmt(data.predictions.inlet)+'°C</b> · with allowance '+fmt(data.upper_estimates_c.inlet)+'°C</span><span>Rack hot spot <b>'+fmt(data.predictions.hotspot)+'°C</b> · with allowance '+fmt(data.upper_estimates_c.hotspot)+'°C</span></div>'+(data.violations.length?'<p class="long-note">'+data.violations.map(v=>esc(operatorText(v))).join(" · ")+'</p>':"")+(dirty?'<p class="long-note">This check uses your saved limits. Save your changes to check against the new limits.</p>':"");
  }catch(error){failure(error);}
  finally{setBusy(false);}
}
async function loadHistory(){
  const dc=$("dc").value,token=epoch;
  const data=await api("/api/history?dc="+encodeURIComponent(dc)+"&limit=10&offset="+historyOffset);
  if(dc!==$("dc").value||token!==epoch)return;
  historyTotal=data.total;
  $("history-content").innerHTML=data.items.length?'<div class="table-scroll"><table class="history-table"><thead><tr><th>CREATED</th><th>RECORD DATE</th><th>METHOD</th><th>STATUS</th><th>POWER CHANGE</th><th>SIMULATION</th><th></th></tr></thead><tbody>'+data.items.map(r=>'<tr><td>'+esc(new Date(r.created_at).toLocaleString())+'</td><td>'+esc(r.timestamp)+'</td><td>'+esc(displayLabel(r.algorithm))+'</td><td>'+badge(r.status==="feasible"?"Meets limits":"No suitable settings",r.status==="feasible"?"success":"warning")+'</td><td>'+(r.savings_pct===null?"—":fmt(r.savings_pct)+"% reduction")+'</td><td>'+(r.applied_at?"Applied":r.stale?"Needs a new check":"Not applied")+'</td><td><button class="link-button run-detail" data-run="'+r.id+'">View details ↗</button></td></tr>').join("")+'</tbody></table></div>':'<div class="empty-state"><h3>Your recommendation history</h3><p>Results for this data center will appear here, including searches that found no suitable settings and settings applied to simulation.</p></div>';
  $("history-page").textContent=data.total? (historyOffset+1)+"–"+Math.min(historyOffset+10,data.total)+" of "+data.total:"No results yet";
  $("history-prev").disabled=historyOffset===0;$("history-next").disabled=historyOffset+10>=data.total;
  document.querySelectorAll(".run-detail").forEach(button=>button.addEventListener("click",async()=>{
    try{const record=await api("/api/history/"+button.dataset.run);$("history-detail").innerHTML='<div class="panel detail-card"><div class="panel-heading"><div><h2>Recommendation details</h2><p>'+esc(record.timestamp)+' · saved limits version '+record.policy.revision+'</p></div>'+badge(record.stale?"Needs a new check":"Recorded result",record.stale?"warning":"neutral")+'</div>'+recommendationHtml(record,false)+'</div>';}catch(error){failure(error);}
  }));
}
async function loadData(){
  const [d,m]=await Promise.all([api("/api/datasets"),api("/api/models")]);
  $("data-content").innerHTML='<div class="panel data-section"><h2>Data checks</h2><p>Data is checked when the application starts. If a check fails, the application will not start.</p><div class="data-metrics"><div><span>RECORDS</span><b>'+d.total_rows.toLocaleString()+'</b></div><div><span>DATA CENTERS</span><b>'+d.data_centers.length+'</b></div><div><span>MODEL INPUTS</span><b>'+d.features+'</b></div><div><span>DATA CHECK</span><b>Passed</b></div></div><div class="table-scroll"><table class="data-table"><thead><tr><th>Data use</th><th>Records</th><th>From</th><th>To</th></tr></thead><tbody>'+["train","validation","test"].map(k=>'<tr><td>'+esc(displayLabel(k))+'</td><td>'+d.splits[k].rows.toLocaleString()+'</td><td>'+esc(d.splits[k].from)+'</td><td>'+esc(d.splits[k].to)+'</td></tr>').join("")+'</tbody></table></div><p class="field-note">Learning, checking and final testing use separate time periods. Reference recommendations are not used as prediction inputs.</p><div class="data-note">'+esc(operatorText(d.corrections))+'<br>Original data files are kept in datasets/raw/. '+esc(operatorText(d.provenance))+'</div></div>'
  +'<div class="panel data-section"><h2>Prediction models</h2><p>Model version: '+esc(displayLabel(m.profile?.name||'legacy'))+'. Records used to learn: '+d.splits.train.rows.toLocaleString()+'; separate records used to check: '+d.splits.validation.rows.toLocaleString()+'. </p><p>Accuracy is checked against 1,500 separate records. Average error shows how far estimates are from recorded values. RMSE gives more weight to large errors. R² shows overall fit; closer to 1 is better. Errors use °C for temperatures and kW for power and cooling load.</p><div class="table-scroll"><table class="data-table"><thead><tr><th>What is estimated</th><th>Model</th><th>Average error</th><th>Error score (RMSE)</th><th>Fit score (R²)</th></tr></thead><tbody>'+Object.values(m.models).map(r=>'<tr><td>'+esc(displayLabel(r.target))+'</td><td>'+esc(displayLabel(r.estimator))+'</td><td>'+fmt(r.test.mae,3)+'</td><td>'+fmt(r.test.rmse,3)+'</td><td>'+fmt(r.test.r2,4)+'</td></tr>').join("")+'</tbody></table></div><div class="data-note">Seasons in these records that the model has not learned from: '+esc(d.unknown_test_categories.season.join(", ")||"none")+'. '+d.test_baseline_region_support.supported+'/'+d.test_baseline_region_support.total+' recorded conditions ('+fmt(d.test_baseline_region_support.percent,2)+'%) are covered by the model’s training data. This check compares operating conditions and settings, not dates.<br>Temperature allowances are based on errors measured during model checks. Estimated savings compare settings while other recorded conditions stay the same. Actual equipment performance and savings may differ.</div></div>';
}
async function navigate(next){
  if(busy)return;
  view=next;clearError();
  document.querySelectorAll(".nav-item").forEach(el=>el.classList.toggle("active",el.dataset.view===next));
  document.querySelectorAll(".view").forEach(el=>el.classList.add("hidden"));
  $(next==="controls"?"overview-view":next+"-view").classList.remove("hidden");
  const titles={overview:["Operations overview","Overview"],controls:["Cooling settings","Cooling settings"],history:["Recommendation history","Recommendation history"],data:["Data & models","Data & models"]};
  $("heading").textContent=titles[next][0];$("page-name").textContent=titles[next][1];
  $("subtitle").textContent=next==="history"?"Review past suggestions, saved limits and settings applied to simulation.":next==="data"?"See where the records come from and how well the estimates match them.":"Check cooling needs, set limits and review suggested settings.";
  $("history-detail").innerHTML="";
  try{if(next==="history")await loadHistory();if(next==="data")await loadData();}catch(error){failure(error);}
  if(next==="controls")$("limits-panel").scrollIntoView({behavior:"smooth",block:"start"});else window.scrollTo({top:0,behavior:"smooth"});
}
document.querySelectorAll(".nav-item").forEach(el=>el.addEventListener("click",()=>navigate(el.dataset.view)));
$("dc").addEventListener("change",loadCenter);$("snapshot").addEventListener("change",loadSnapshot);
$("supported-snapshot").addEventListener("click",useSupportedSnapshot);
$("limits-form").addEventListener("input",markDirty);$("limits-form").addEventListener("submit",saveLimits);
$("reset-limits").addEventListener("click",async()=>{
  clearError();setBusy(true);
  try {
    policy=await api("/api/limits/"+encodeURIComponent($("dc").value));
    populateLimits();
    if(run){run.stale=run.policy.revision!==policy.revision;renderRun();}
    current=await api("/api/states/"+$("snapshot").value);renderSnapshot();
    toast("Restored the latest saved limits.");
  } catch(error){failure(error);}
  finally{setBusy(false);}
});
$("optimize").addEventListener("click",optimize);$("manual-form").addEventListener("submit",preview);
$("history-prev").addEventListener("click",()=>{historyOffset=Math.max(0,historyOffset-10);loadHistory().catch(failure);});
$("history-next").addEventListener("click",()=>{historyOffset+=10;loadHistory().catch(failure);});
async function health(){
  try{const data=await api("/api/health",{timeout:10000});$("connection").textContent=data.optimization_busy?"Finding a recommendation":"Connected";$("connection-dot").style.background="#25a59a";}
  catch{$("connection").textContent="Connection lost";$("connection-dot").style.background="#b64145";}
}
async function init(){
  setBusy(true);
  try{config=await api("/api/config");$("dc").innerHTML=config.data_centers.map(dc=>'<option>'+esc(dc)+'</option>').join("");await loadCenter();await health();}
  catch(error){failure(error);$("connection").textContent="Unable to start";}
  finally{setBusy(false);}
}
init();setInterval(health,30000);

