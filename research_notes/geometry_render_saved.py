"""Finite existing renderer invocation, recorded in the same stage ledger."""
import os,sys,json,subprocess,time,signal,argparse,hashlib
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'research_notes'))
from geometry_supervisor import ledger,snapshot,RUN
from causal_supervisor import processes
from m7_io import write
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True);p.add_argument('--gpu',type=int,default=6);a=p.parse_args();spec=Path(a.spec);out=Path(a.output);items=json.loads(spec.read_text());assert all(Path(x['path']).exists() for it in items for x in it['sides']);before=snapshot(a.gpu);assert before['used_mib']<100 and before['utilization']<5
out.mkdir();write(out/'video-pairs.json',items)
with ledger() as d:
 assert sum(x.get('wall_seconds',x['reserved_seconds']) for x in d['attempts'])+900<=21600
 active=[x for x in d['attempts'] if x['status']=='running'];affinity=[24,25,26,27];assert len(active)<3 and not any(x['gpu']==a.gpu or set(x['affinity'])&set(affinity) for x in active)
 idx=len(d['attempts']);row=dict(kind='video',arm='video',stage='saved-physics-render',status='running',source=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),path=str(out),gpu=a.gpu,affinity=affinity,reserved_seconds=900,reserved_decisions=0,reserved_updates=0,gpu_before=before);d['attempts'].append(row)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=before['uuid'],MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID=str(a.gpu),PYOPENGL_PLATFORM='egl',M11_VIDEO_RUN=str(out),BFM_RECOVERY_OUTPUT=str(out),OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
for k in ['TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR']:
 dest=out/k;dest.mkdir();env[k]=str(dest)
def limits():os.sched_setaffinity(0,set(affinity));os.nice(10)
start=time.monotonic();child=None
try:
 with (out/'stdout.log').open('wb') as f:
  child=subprocess.Popen([str(P/'.venv/bin/python'),'-u',str(repo/'research_notes/geometry_render.py')],env=env,cwd=out,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits);row['pid']=child.pid
  with ledger() as d:d['attempts'][idx]=row
  child.wait(timeout=880)
finally:
 if child and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
 row.update(status='finished',wall_seconds=time.monotonic()-start,exit_code=child.returncode if child else None,actual_updates=0,remaining_processes=processes(child.pid) if child else [],gpu_after=snapshot(a.gpu))
 with ledger() as d:d['attempts'][idx]=row
 write(out/'resources.json',row)
