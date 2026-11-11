"""Finite authorized Issue59 order. Stops on failure; not a Bridge or retry daemon."""
import hashlib,json,os,subprocess,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R.parent;P=S.parents[1];PY=P/'.venv/bin/python';STATUS=S/'queue-status.json';start=time.time()
def read(p):return json.loads(p.read_text())
def write(v):
 v.update(updated=time.time(),coordinator_pid=os.getpid());tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(STATUS)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def job(name,mode,output,extra=()):
 row=dict(status='running',current=name,formal_started=mode=='high_train');write(row)
 cmd=[str(PY),str(R/'research_notes/frl_v3_job.py'),'--name',name,'--category','P2','--script','frl_v3_train_stage.py','--','--mode',mode,'--output',str(output),'--p1-output',str(P1),*map(str,extra)]
 with (S/'jobs'/(name+'-supervisor.log')).open('w') as f:subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 data=read(S/'jobs'/(name+'.json'));assert data['status']=='completed' and not data.get('budget_timeout'),data
 return output
try:
 # Independent immutable P1 runtime continues; only wait for the one owned finite job.
 write(dict(status='waiting_for_P1',formal_started=False))
 while True:
  state=read(S/'jobs/counterfactual-02.json')
  if state['status']!='running':break
  if state.get('expected_budget_exceeded'):write(dict(status='P1_over_expected_report_required',formal_started=False))
  time.sleep(10)
 assert state['status']=='completed','P1 correctness failed; no automatic retry or training'
 P1=S/'runs/counterfactual-02';gate=read(P1/'gate.json')
 # Always verify archived complete baseline continuations before interpreting existence.
 verify_start=time.time()
 try:
  with (S/'jobs/counterfactual-verification.log').open('w') as f:subprocess.run([str(PY),str(R/'research_notes/frl_v3_verify_counterfactual.py'),'--output',str(P1)],stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1'))
 finally:(S/'jobs/counterfactual-verification-job.json').write_text(json.dumps(dict(category='P1',name='counterfactual-verification',status='completed' if (P1/'counterfactual-verification.json').exists() else 'failed',elapsed_s=time.time()-verify_start),indent=2))
 assert read(P1/'counterfactual-verification.json')['all_archived_continuations_bitwise']
 if not gate['P2_allowed']:
  write(dict(status='stopped_P1_no_effective_low_slip_branch',formal_started=False,scientific_report_required=True));raise SystemExit(0)
 checks=job('lambda0-qualification-01','unit_checks',S/'runs/lambda0-qualification-01',('--risk-lambda','0'))
 ck=read(checks/'checks.json')
 assert ck['full_physical_prefix_bitwise'],'Complete state/physics restoration error; no training can repair this qualification'
 # Any update byte inequality forbids archive-as-control and requires a fresh matched control, as authorized.
 # Probability/reward/GAE assertions have already run unchanged inside qualification.
 launch_gate=dict(status='qualified_matched_physical_reward',formal_RL_launch_allowed=True,nonreward_correctness_passed=True,original_reward_audit='Issue58 and archive diagnostics retained; P1 actual old/v3 physical scoring',risk_lambda=.33,lambda0_checks=str(checks/'checks.json'),old_archived_control_qualified=ck['old_archived_control_qualified'])
 gatefile=S/'prep/formal-v3-launch-gate.json';assert not gatefile.exists();gatefile.write_text(json.dumps(launch_gate,indent=2))
 if not ck['old_archived_control_qualified']:
  control=job('formal-control-lambda0-01','high_train',S/'runs/formal-control-lambda0-01',('--risk-lambda','0','--reward-gate',gatefile))
 else:control=P/'experiments/frl-v2-fb-repair-20261110/runs/frl-formal-B-01'
 train=job('formal-v3-lambda033-01','high_train',S/'runs/formal-v3-lambda033-01',('--risk-lambda','.33','--reward-gate',gatefile))
 # Fixed archived control evaluated on SAME NEW development parents; never old sealed.
 if ck['old_archived_control_qualified']:
  for k in (0,16,32,64,96,128):
   checkpoint=control/f'candidate-{k}.pt';job(f'oldB-newdev-{k:03d}','high_eval',S/'runs'/f'oldB-newdev-{k:03d}',('--risk-lambda','0','--checkpoint',checkpoint,'--sha256',sha(checkpoint),'--iteration',k,'--record'))
 # Old32 are EXPOSED REGRESSION ONLY and cannot reselect a newdev checkpoint.
 for k in (64,128):
  checkpoint=train/f'candidate-{k}.pt';job(f'v3-oldreg-{k:03d}','high_eval',S/'runs'/f'v3-oldreg-{k:03d}',('--risk-lambda','.33','--checkpoint',checkpoint,'--sha256',sha(checkpoint),'--iteration',k,'--record','--regression'))
 write(dict(status='all_authorized_jobs_completed_pending_scientific_review',formal_started=True,scientific_report_required=True,archived_control=ck['old_archived_control_qualified'],no_automatic_next_phase=True))
except BaseException as e:
 if not isinstance(e,SystemExit):write(dict(status='stopped_failure_no_automatic_retry',error=repr(e),traceback=traceback.format_exc(),scientific_report_required=True));raise
