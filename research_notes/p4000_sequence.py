"""Finite two-seed stage sequence, not a daemon: fixed budgets, common extension, one final pass."""
import os,json,subprocess,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R.parent;P=S.parents[1];PY=P/'.venv/bin/python'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2))
def remaining():
 l=read(S/'ledger.json');return l['budget_job_seconds']-l.get('auxiliary_CPU_analysis_reserve_s',0)-sum(j.get('elapsed_s',0) for j in l['jobs'])
def job(name,repo,script,limit,args):
 assert remaining()>limit+30,('stage reserve',name,remaining(),limit)
 log=(S/'logs'/(name+'-supervisor.log')).open('w');cmd=[str(PY),'-u',str(repo/'research_notes/rolling_job.py'),name,str(repo/'research_notes'/script),str(limit),*args];q=subprocess.Popen(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write(S/(name+'-supervisor.json'),dict(pid=q.pid,start_ticks=Path(f'/proc/{q.pid}/stat').read_text().split()[21],command=cmd))
 assert q.wait()==0,name
 result=read(S/'runs'/name/'result.json');assert result['status']=='completed',(name,result)
def main():
 # Wait only for exact owned finite preflight; source and all jobs fixed, no restarts.
 seen=read(S/'preflight-supervisor.json');pid=seen['pid'];start=seen['start_ticks']
 while Path(f'/proc/{pid}/stat').exists() and Path(f'/proc/{pid}/stat').read_text().split()[21]==start:time.sleep(10)
 pf=read(S/'runs/preflight-01/result.json');assert pf['status']=='completed',pf
 std=pf['study']['selected_std'];write(S/'sequence-status.json',dict(status='PPO_running',std=std))
 final={};jobs={}
 for seed in [5201,5202]:
  repo=S/f'seed{seed}-repo';name=f'seed{seed}-32';out=S/'runs'/name;limit=min(5500.,(remaining()-1800)/2) # preserve the other seed and sealed final reserve
  args=['--output',str(out),'--seed',str(seed),'--iterations','32','--std',str(std),'--max-seconds',str(limit-60)];job(name,repo,'p4000_ppo.py',limit,args);jobs[seed]=out
 # Joint extension decisions use DEVELOPMENT only, never final cases.
 evidence={};est=0.
 for seed,out in jobs.items():
  sel=read(out/'selection.json');base=read(out/'dev-0000/score.json') if (out/'dev-0000/score.json').exists() else None
  from p4000_ppo import assess
  initial=read(out/'baseline-dev.json');initial_score=assess(initial,initial)['score'];last=read(out/'dev-0032/score.json');lines=[json.loads(x) for x in (out/'training.jsonl').read_text().splitlines()];rate=sum(x['sampling_seconds']+x['optimization_seconds'] for x in lines)/len(lines);est+=32*rate+500
  ok=sel['iteration'] in [24,32] and initial_score-sel['score']>=.1 and last['score']<=sel['score']+.25 and not last['severe'];evidence[str(seed)]=dict(initial_score=initial_score,selected=sel,latest_dev=last,qualifies=ok,estimated_more_seconds=32*rate+500)
 extend=all(x['qualifies'] for x in evidence.values()) and remaining()>=est+1200
 write(S/'extension-decision.json',dict(extend_both=extend,evidence=evidence,remaining_seconds=remaining(),estimate=est,final_test_consulted=False,rule=read(R/'research_notes/p4000_protocol.json')['extension']))
 if extend:
  for seed in [5201,5202]:
   repo=S/f'seed{seed}-repo';prev=jobs[seed];name=f'seed{seed}-64';out=S/'runs'/name;limit=min(5000.,evidence[str(seed)]['estimated_more_seconds']*1.3)
   job(name,repo,'p4000_ppo.py',limit,['--output',str(out),'--seed',str(seed),'--iterations','64','--std',str(std),'--resume',str(prev/'last.pt'),'--max-seconds',str(limit-60)]);jobs[seed]=out
 final['P4000']=read(R/'research_notes/p4000_protocol.json')['initial']['path']
 for seed,out in jobs.items():final[f'seed{seed}_best']=str(out/'best.pt');final[f'seed{seed}_last']=str(out/'last.pt')
 write(S/'final-models.json',final);job('final-01',R,'p4000_evaluate.py',min(1500.,remaining()-600),['--output',str(S/'runs/final-01')])
 l=read(S/'ledger.json');l['status']='PPO_and_sealed_physics_complete_pending_analysis_videos';l['actual_RL_decisions']=sum(read(out/'result.json')['RL_decisions'] for out in jobs.values());assert l['actual_RL_decisions']<=l['RL_decisions_limit'];write(S/'ledger.json',l);write(S/'sequence-status.json',dict(status=l['status'],std=std,models=final,RL_decisions=l['actual_RL_decisions'],automatic_next_training=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  write(S/'sequence-status.json',dict(status='failed_saved_no_auto_retry',error=traceback.format_exc()));raise
