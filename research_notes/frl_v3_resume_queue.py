"""Finite pre-batch executed-reward check, then the original pinned v3 run."""
import hashlib,json,os,subprocess,time,traceback
from pathlib import Path
Q=Path(__file__).resolve().parents[1]; S=Q.parent; P=S.parents[1]
R=S/'learning-runtime'; PY=P/'.venv/bin/python'; P1=S/'runs/counterfactual-02'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def status(**row):
 row.update(updated=time.time(),coordinator_pid=os.getpid())
 t=S/'queue-status.tmp';t.write_text(json.dumps(row,indent=2));t.replace(S/'queue-status.json')
def job(name,mode,output,extra=(),root=R,script='frl_v3_train_stage.py'):
 status(status='running',current=name,formal_started=mode=='high_train')
 cmd=[str(PY),str(root/'research_notes/frl_v3_job.py'),'--name',name,'--category','P2','--script',script,'--','--mode',mode,'--output',str(output),'--p1-output',str(P1),*map(str,extra)]
 with (S/'jobs'/(name+'-supervisor.log')).open('w') as f:subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
 side=read(S/'jobs'/(name+'.json'));assert side['status']=='completed' and not side.get('budget_timeout'),side
 return output
try:
 status(status='waiting_for_initial_development_normal_stop',formal_actor_updates=0)
 first=S/'runs/formal-v3-lambda033-01'
 while read(S/'jobs/formal-v3-lambda033-01.json')['status']=='running':time.sleep(5)
 assert read(S/'jobs/formal-v3-lambda033-01.json')['status']=='completed'
 outcome=read(first/'result.json')['study']
 assert outcome['status']=='stopped_by_request' and outcome['Expert_RL_decisions']==0,outcome
 assert not (first/'learning.json').exists()
 checks=job('actual-lambda033-qualification-01','unit_checks',S/'runs/actual-lambda033-qualification-01',('--risk-lambda','.33'),root=Q,script='frl_v3_positive_check.py')
 c=read(checks/'checks.json')
 assert c['physics_bitwise'] and c['native_and_high_reward_exact_formula'] and c['positive_event_controls']>0
 status(status='actual_reward_qualified_resuming_same_runtime_and_complete_candidate0',formal_actor_updates=0)
 gate=S/'prep/formal-v3-launch-gate.json'
 train=job('formal-v3-lambda033-02','high_train',S/'runs/formal-v3-lambda033-02',('--risk-lambda','.33','--reward-gate',gate,'--resume',first/'candidate-0.pt'))
 outcome=read(train/'result.json')['study']
 assert outcome.get('status')!='stopped_by_request','No follow-up after a budget or explicit stop'
 old=P/'experiments/frl-v2-fb-repair-20261110/runs/frl-formal-B-01'
 for k in (0,16,32,64,96,128):
  checkpoint=old/f'candidate-{k}.pt';job(f'oldB-newdev-{k:03d}','high_eval',S/'runs'/f'oldB-newdev-{k:03d}',('--risk-lambda','0','--checkpoint',checkpoint,'--sha256',sha(checkpoint),'--iteration',k,'--record'))
 for k in (64,128):
  checkpoint=train/f'candidate-{k}.pt';job(f'v3-oldreg-{k:03d}','high_eval',S/'runs'/f'v3-oldreg-{k:03d}',('--risk-lambda','.33','--checkpoint',checkpoint,'--sha256',sha(checkpoint),'--iteration',k,'--record','--regression'))
 status(status='all_authorized_jobs_completed_pending_scientific_review',formal_started=True,scientific_report_required=True,no_automatic_next_phase=True)
except BaseException as exc:
 status(status='stopped_failure_no_automatic_retry',error=repr(exc),traceback=traceback.format_exc(),scientific_report_required=True)
 raise
