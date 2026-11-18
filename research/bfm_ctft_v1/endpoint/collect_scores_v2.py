"""Read completed immutable episodes while the sole simulation job runs."""
from pathlib import Path
import json,os,subprocess,sys,time
root=Path(os.environ["CTFT_PINNED_RUNTIME"]).parent
out=root/"runs/endpoint-evaluation-02";prep=root/"prep"
start=time.monotonic()
log=out/"score-ledger.json"; costs=[]
def save(**fields):
    tmp=log.with_suffix(".partial");tmp.write_text(json.dumps(dict(start_age_s=time.monotonic()-start,subjobs=costs,**fields),indent=2));tmp.replace(log)
while time.monotonic()-start<14400:
    n=len(list(out.glob("ctft-*.pkl.gz")))
    count=len(list(out.glob("*.metrics.json")))
    if n-count>=9 or n==252 or (out/"cost.json").exists():
        for f in ["score_endpoint_v1.py","response_audit_v1.py"]:
            t=time.monotonic()
            rc=subprocess.call([sys.executable,str(prep/f),str(out)])
            costs.append(dict(script=f,exit=rc,wall_s=time.monotonic()-t));save(scored=len(list(out.glob("*.metrics.json"))))
            if rc:raise RuntimeError(f)
        if n==252:
            subprocess.run([sys.executable,str(prep/"final_tables_v1.py"),str(out)],check=True)
            save(status="COMPLETE");break
        if (out/"cost.json").exists():
            save(status="COLLECTOR_STOPPED");break
    time.sleep(15)
