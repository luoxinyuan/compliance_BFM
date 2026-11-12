"""One owned serial qualification process, cumulative hard budget and storage guard."""
import os,sys,time,json,subprocess,shutil,signal
from pathlib import Path
from supervision import tree_bytes,stop_owned_group
S=Path(sys.argv[1]);case_id=sys.argv[2];name=sys.argv[3];R=S/'runtime';P=S.parents[1]
L=R/'research_notes/adapter_envelope_lock_v1.json';lock=json.loads(L.read_text());assert lock['status']=='qualification_locked_not_yet_executed'
case=next(c for c in lock['qualification_states'] if c['id']==case_id)
J=S/'jobs';J.mkdir(exist_ok=True);(S/'runs').mkdir(exist_ok=True);(S/'tmp').mkdir(exist_ok=True)
base=lock['time_projection']['already_charged']+lock['time_projection']['current_preparation_analysis_tests_reserve']
CAP=lock['chosen_cap_s'];OC=lock['chosen_cap_bytes'];roots=lock['counted_stage_roots']
def write(p,d):
 t=p.with_suffix('.pending');t.write_text(json.dumps(d,indent=2));os.replace(t,p)
def spent():
 d=[json.loads(f.read_text()) for f in J.glob('qualification-job-*.json')]
 assert all('wall_s' in r for r in d),'unfinished owned job must be reconciled'
 return base+sum(r['wall_s'] for r in d)
prior=spent();runtime=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','-C',str(R),'status','--porcelain'])
uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert not any(l.split(',')[0].strip()==uuid for l in apps.splitlines()),'GPU0 occupied'
assert shutil.disk_usage(S).free>100*2**30 and prior+600<CAP
cmd=[str(P/'.venv/bin/python'),'-u',str(R/'research/interaction_fb/benchmark.py'),'--output',str(S/'runs'/name),'--arm','A' if case_id.startswith('A') else 'B','--resume',case['checkpoint'],'--manifest',str(R/'research_notes/interaction_pilot_manifest_v2/train.json'),'--envelope-lock',str(L),'--envelope-qualification',case_id,'--qual-bundle']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(S/'tmp'))
side=J/('qualification-job-'+name+'.json');assert not side.exists();start=time.time();record=dict(name=name,start=start,runtime=runtime,command=cmd,supervisor=os.getpid())
with (J/(name+'.log')).open('x') as f:
 p=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);record['pid']=p.pid;write(side,record)
 try:
  while p.poll() is None:
   wall=time.time()-start
   if prior+wall>=CAP-120 or tree_bytes(roots)>OC-256*2**20 or shutil.disk_usage(S).free<100*2**30 or wall>600:
    record['guard_stop']='budget_storage_or_600s_job_projection';break
   time.sleep(5)
 finally:
  rc=stop_owned_group(p,grace=30);record.update(exit_code=rc,end=time.time(),wall_s=time.time()-start,output_bytes=tree_bytes(roots));write(side,record)
write(S/'ledger-envelope.json',dict(runtime=runtime,status='qualification_job_finished',charged_upper_s=prior+record['wall_s'],cap_s=CAP,output_bytes=record['output_bytes'],cap_bytes=OC,job=record,sealed_read=False))
assert rc==0 and 'guard_stop' not in record,record
