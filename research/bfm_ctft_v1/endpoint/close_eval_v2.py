"""Post-collection verification. Keep simulation-wrapper failures and native evidence."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(os.environ["CTFT_PINNED_RUNTIME"]);stage=root.parent
out=stage/"runs/endpoint-evaluation-02";prep=stage/"prep"
started=time.monotonic();jobs=[]
def save(status,**more):
    path=out/"postprocess-ledger.json";tmp=path.with_suffix(".partial")
    tmp.write_text(json.dumps(dict(status=status,wall_s_including_wait=time.monotonic()-started,jobs=jobs,**more),indent=2));tmp.replace(path)
def run(script,**env):
    start=time.monotonic();p=subprocess.run([sys.executable,str(prep/script),str(out)],env=dict(os.environ,**env))
    jobs.append(dict(script=script,seconds=time.monotonic()-start,exit=p.returncode));save("POSTPROCESSING")
    p.check_returncode()
save("WAITING_FOR_COLLECTOR_AND_SCORER")
while not ((out/"cost.json").exists() and (out/"paired-summary.json").exists()):
    if (out/"cost.json").exists() and not (out/"rollouts-complete.json").exists():
        save("COLLECTOR_INCOMPLETE");sys.exit(2)
    if time.monotonic()-started>14200:save("TIME_LIMIT");sys.exit(3)
    time.sleep(15)
p=json.loads((out/"progress.json").read_bytes())
assert len(p)==len({(r["parent"],r["model"],r["branch"]) for r in p})==252
run("native_retention_v1.py")
run("native_canary_v1.py")
run("endpoint_report_v1.py")
sys.path.insert(0,str(root/"research/bfm_ctft_v1"))
from resource_guard import check
check(stage)
run("render_endpoints_v1.py",CUDA_VISIBLE_DEVICES="0",MUJOCO_GL="egl",PYOPENGL_PLATFORM="egl",MUJOCO_EGL_DEVICE_ID="0")
files={}
for f in sorted(out.glob("*.pkl.gz")):
    files[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
(out/"episode-sha256.json").write_text(json.dumps(files,indent=2))
save("PHYSICS_SCORES_AND_VIDEO_COMPLETE",episodes=len(files))
