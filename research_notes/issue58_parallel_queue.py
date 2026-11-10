"""Finite Issue58 queue plus user-authorized acceleration qualification; original formal A/B and FB runtimes pinned."""
import hashlib,json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];S=ROOT.parent;P=S.parents[1]
PYTHON=str(P/'.venv/bin/python');FRL=S/'integration-runtime'
FB_RUNTIME=S/'fb-runtime'
PROFILE_RUNTIME=S/'efficiency-runtime'
status=S/'queue-status.json'
def write(state,**kw):
 temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(state=state,updated=time.time(),**kw),indent=2));temp.replace(status)
def wait_for(name):
 side=S/'jobs'/(name+'.json')
 while True:
  if (S/'QUEUE_STOP_REQUEST.json').exists():raise RuntimeError('Stage queue stop requested')
  if side.exists():
   row=json.loads(side.read_text())
   if row.get('status')!='running':
    assert row.get('returncode')==0,('Previous job failed',name,row)
    time.sleep(1);return row
  time.sleep(15)
def follow(name,kind,**kw):
 cmd=[PYTHON,str(FB_RUNTIME/'research_notes/issue58_followup_job.py'),name,'--kind',kind]
 for k,v in kw.items():cmd+=['--'+k.replace('_','-'),str(v)]
 write('running_followup',job=name,kind=kind,command=cmd)
 with (S/'jobs'/(name+'-supervisor.log')).open('x') as f:
  result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(('Follow-up supervisor failed',name,result.returncode))
 row=wait_for(name);return row

def sealed(arm):
 source=S/'runs'/('frl-formal-'+arm+'-01');choice=json.loads((source/'selection.json').read_text())
 follow('frl-sealed-'+arm+'-01','sealed',target_runtime=FRL,arm=arm,
        checkpoint=choice['checkpoint_path'],sha256=choice['sha256'],iteration=choice['iteration'])

def frozen_pair():
 wait_for('frl-formal-A-01')
 learn=json.loads((S/'runs/frl-formal-A-01/learning.json').read_text())
 assert learn[-1]['RL_decisions']==98304,'No matched B launch after incomplete/correctness-failed A'
 sealed('A')
 import hashlib
 initial=S/'runs/frl-formal-A-01/candidate-0.pt'
 initial_sha=hashlib.sha256(initial.read_bytes()).hexdigest()
 follow('frl-sentinel-initial-01','sentinel',checkpoint=initial,sha256=initial_sha)
 choice=json.loads((S/'runs/frl-formal-A-01/selection.json').read_text())
 if choice['iteration']!=0:follow('frl-sentinel-A-01','sentinel',checkpoint=choice['checkpoint_path'],sha256=choice['sha256'])
 write('running_formal_B')
 cmd=[PYTHON,str(FRL/'research_notes/frl_v2_job.py'),'frl-formal-B-01','--arm','B','--reward-gate',str(S/'prep/reward-gate-01.json')]
 with (S/'jobs/frl-formal-B-01-supervisor.log').open('x') as f:
  child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
  initial_file=S/'runs/frl-formal-B-01/initialization.json'
  while child.poll() is None and not initial_file.exists():time.sleep(3)
  if initial_file.exists():
   a=json.loads((S/'runs/frl-formal-A-01/initialization.json').read_text());b=json.loads(initial_file.read_text())
   if a['model_state_sha256']!=b['model_state_sha256']:
    (S/'runs/frl-formal-B-01/STOP_REQUEST.json').write_text(json.dumps(dict(reason='A/B initialization mismatch')))
    child.wait();raise RuntimeError('A/B initial full model state SHA mismatch')
  assert child.wait()==0,'B supervisor failed'
 wait_for('frl-formal-B-01')
 learn=json.loads((S/'runs/frl-formal-B-01/learning.json').read_text());assert learn[-1]['RL_decisions']==98304
 sealed('B')
 choice=json.loads((S/'runs/frl-formal-B-01/selection.json').read_text())
 if choice['iteration']!=0:follow('frl-sentinel-B-01','sentinel',checkpoint=choice['checkpoint_path'],sha256=choice['sha256'])
 return dict(status='formal_pair_and_sealed_complete',decisions=196608)

def physical_repair():
 follow('p0a-physical-reference-01','physical-reference')
 follow('p0a-physical-resumed-01','physical-resumed')
 ra=json.loads((S/'runs/p0a-physical-reference-01/physical-result.json').read_text());rb=json.loads((S/'runs/p0a-physical-resumed-01/physical-result.json').read_text())
 for k in ('physical_tree_digest','control_tree_digest','final_state_tree_digest','rng_tree_digest'):
  assert ra[k]==rb[k],('P0-A actual physical continuation mismatch',k)
 return dict(status='actual_physical_continuation_exact',boundary=ra['boundary'])

def safe_engineering():
 follow('p0b-safety-01','safe')
 package=S/'runs/p0b-safety-01/safe-user-stop-6157'
 if package.exists():
  follow('p0b-safe-resume-01','safe-resume',package=package)
  follow('p0b-safe-post-01','safe-post',package=S/'runs/p0b-safety-01/user-stop-safe-7168')
  return dict(status='engineering_and_exact_resume_and_post_check_completed',formal_FB_1M=0)
 return dict(status='recipe_stopped_before_full_engineering',formal_FB_1M=0)

def compiled_repair():
 follow('p0a-compiled-control-01','compile-control')
 follow('p0a-compiled-resume-01','compile-resume',package=S/'runs/p0a-compiled-control-01/user-stop-1037')
 a=json.loads((S/'runs/p0a-compiled-control-01/fixed16-results.json').read_text());b=json.loads((S/'runs/p0a-compiled-resume-01/fixed16-results.json').read_text())
 assert len(a)==len(b)==16
 assert all(all(x[k]==y[k] for k in ('model','adam','rng','gradients')) for x,y in zip(a,b))
 return dict(status='compiled_exact_pass')

def efficiency_check():
 cmd=[PYTHON,str(PROFILE_RUNTIME/'research_notes/frl_efficiency_job.py')]
 write('running_acceleration_qualification',command=cmd)
 with (S/'jobs/frl-efficiency-physics-01-supervisor.log').open('x') as log:
  child=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 assert child.returncode==0,'Acceleration qualification failed; original formal runtimes unchanged'
 row=wait_for('frl-efficiency-physics-01')
 return dict(status='physical_profile_completed',runtime=row['runtime'],formal_AB_runtime_unchanged=True)


def parallel_check():
 cmd=[PYTHON,str(ROOT/'research_notes/frl_parallel_job.py')]
 write('running_parallel_sampling_qualification',command=cmd)
 with (S/'jobs/frl-parallel-physics-01-supervisor.log').open('x') as log:
  result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0,'Parallel qualification failed; original formal runtimes unchanged'
 row=wait_for('frl-parallel-physics-01')
 return dict(status='sampling_qualification_completed',runtime=row['runtime'],formal_AB_runtime_unchanged=True)


def main():
 results={}
 write('waiting_for_formal_A',formal_runtime='19b3117c8f808510a0c387e7f3de267a67cf7e90')
 for package,function in [('F-RL-v2',frozen_pair),('Acceleration',efficiency_check),('Parallel',parallel_check),('P0-A-physics',physical_repair),('P0-B',safe_engineering),('P0-A-compiled',compiled_repair)]:
  if (S/'QUEUE_STOP_REQUEST.json').exists():
   write('stopped_by_request',completed=results,automatic_restart=False);return
  try:
   results[package]=function()
  except BaseException as exc:
   results[package]=dict(status='stopped_at_failure',error=repr(exc),traceback=traceback.format_exc())
   (S/(package.replace('/','_')+'-failure.json')).write_text(json.dumps(results[package],indent=2))
   # Independent packages do not restart this failed one. A surviving heavy
   # process is a resource blocker, never permission to overlap GPU work.
   rows=[json.loads(p.read_text()) for p in (S/'jobs').glob('*.json')]
   if any(x.get('status')=='running' for x in rows):
    write('paused_for_active_failed_package',completed=results);raise
 write('all_authorized_jobs_stopped_pending_scientific_closeout',packages=results,
       formal_FB_1M=0,Student=0,automatic_next_stage=False)
if __name__=='__main__':main()
