import os,sys,json,subprocess,time,signal
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');repo=P/'bridge-v2-owned/trees/transfer-repair/repo';sys.path.insert(0,str(repo/'research_notes'))
from transfer_supervisor import ledger,snapshot,RUN
from causal_supervisor import processes
from m7_io import write
before=snapshot(6);assert before['used_mib']<100 and before['utilization']<5
out=RUN/'videos';out.mkdir();a=RUN/'A-prepare-01/job';s=RUN/'A-final-04/job';b=RUN/'B1-final-05/job/evaluation'
items=[dict(name='01_teacher_student_new_slow',sides=[dict(path=str(a/'initial__teacher_final_heldout_slow.json.gz'),title='TOP: Direct initialized Teacher | 7.99cm / 4.30N'),dict(path=str(s/'initial__student_final_heldout_slow.json.gz'),title='BOTTOM: ideal-force Student (corrected update) | 7.11cm / 3.74N')]),dict(name='02_ppo_generalization_failure',sides=[dict(path=str(a/'m10__heldout_fast.json.gz'),title='TOP: original M10 feedback expert | 7.47cm / 3.73N'),dict(path=str(a/'initial__teacher_final_heldout_fast.json.gz'),title='MIDDLE: Direct initialization | 7.45cm / 4.05N'),dict(path=str(b/'heldout_fast__learned.json.gz'),title='BOTTOM: narrow PPO B1 selected10 | 8.11cm / 4.28N')]),dict(name='03_stop_transfer_and_failure',sides=[dict(path=str(a/'ppo15__teacher_final_early_stop.json.gz'),title='TOP: frozen old PPO15 Teacher | 9.38cm / 3.78N'),dict(path=str(s/'ppo15__student_final_early_stop.json.gz'),title='MIDDLE: its pure-BC ideal-force Student | 9.41cm / 3.73N'),dict(path=str(s/'initial__student_final_early_stop.json.gz'),title='BOTTOM: other Student from initialization | 16.26cm / 4.43N')])]
assert all(Path(x['path']).exists() for it in items for x in it['sides']);write(RUN/'video-pairs.json',items)
with ledger() as d:
 assert sum(a.get('wall_seconds',a['reserved_seconds']) for a in d['attempts'])+600<=21600
 active=[a for a in d['attempts'] if a['status']=='running'];assert len(active)<3 and not any(a['gpu']==6 for a in active)
 idx=len(d['attempts']);row=dict(kind='video',arm='video',stage='saved-physics-render',status='running',source=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),path=str(out),gpu=6,affinity=[24,25,26,27],reserved_seconds=600,reserved_decisions=0,gpu_before=before);d['attempts'].append(row)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=before['uuid'],MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='6',PYOPENGL_PLATFORM='egl',M11_VIDEO_RUN=str(RUN),BFM_RECOVERY_OUTPUT=str(out),OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
for k in ['TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR']:
 dest=out/k;dest.mkdir();env[k]=str(dest)
def limits():os.sched_setaffinity(0,{24,25,26,27});os.nice(10)
start=time.monotonic();child=None
try:
 with (out/'stdout.log').open('wb') as f:
  child=subprocess.Popen([str(P/'.venv/bin/python'),'-u',str(repo/'research_notes/m11_render.py')],env=env,cwd=out,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits);row['pid']=child.pid
  with ledger() as d:d['attempts'][idx]=row.copy()
  child.wait(timeout=580)
finally:
 if child and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
 row.update(status='finished',wall_seconds=time.monotonic()-start,exit_code=child.returncode if child else None,remaining_processes=processes(child.pid) if child else [],gpu_after=snapshot(6))
 with ledger() as d:d['attempts'][idx]=row
 write(out/'resources.json',row)
