"""Finite single-development-seed paired critic trial. No automatic replication or sealed tuning."""
import json,os,time,subprocess
from pathlib import Path
from p4000_sequence import job,remaining
from p4000_critic_sequence import resource_check,owned_alive,read,write
ROOT=Path(__file__).resolve().parents[1];S=ROOT.parent

def main():
 state=S/'single-seed-status.json'
 write(state,dict(status='waiting_for_owned_speed_probe',development_seed=5201,automatic_replication=False))
 identity=read(S/'speed-probe-01-supervisor.json');deadline=time.monotonic()+700
 while owned_alive(identity):
  assert time.monotonic()<deadline;time.sleep(10)
 result=S/'runs/speed-probe-01/probe-result.json'
 adoption=read(result) if result.exists() else dict(adopt_fast_diagnostics=False,reason='optional probe incomplete/failed; original diagnostic retained')
 os.environ['BFM_RECORD_NOMINAL_DIAGNOSTIC']='0' if adoption['adopt_fast_diagnostics'] else '1'
 os.environ['P4000_CANONICAL_ROW_ACTOR']='1'
 write(S/'single-seed-sampler-adoption.json',adoption)
 check=read(S/'runs/probability-check-01/result.json');assert check['status']=='completed' and check['actor_equal']
 assert all(r['wrong_action_rejected'] and r['actor_adam_equal'] and r['old_logprob_max_error']<3e-4 for r in check['rows'])
 assert read(S/'runs/seed5201-32/result.json')['status']=='completed'
 assert read(S/'runs/seed5202-32/result.json')['status']=='failed'
 budget=remaining();assert budget>2*5500+900,('Insufficient reserve for same-seed matched pair',budget)
 write(S/'single-seed-allocation.json',dict(seed=5201,modes=['independent','joint'],iterations_each=32,max_training_job_seconds=5500,remaining_before=budget,analysis_video_reserve_seconds=900,automatic_replication=False,heldout_evaluation='deferred until fixed recipe review; use unchanged7dev cases only'))
 outputs={}
 for mode in ['independent','joint']:
  resource_check();name='single5201-'+mode+'-32';out=S/'runs'/name
  write(state,dict(status='single_seed_paired_development_running',development_seed=5201,current_mode=mode,automatic_replication=False))
  job(name,ROOT,'p4000_critic_ppo.py',5500,['--output',str(out),'--seed','5201','--iterations','32','--std','.008','--critic-mode',mode,'--max-seconds','5440'])
  assert read(out/'result.json')['iterations']==32
  outputs[mode]=str(out)
 ledger=read(S/'ledger.json');new=sum(read(Path(v)/'result.json')['RL_decisions'] for v in outputs.values())
 total=48384+new;assert new==49152 and total<=ledger['RL_decisions_limit']
 ledger.update(status='single_seed_paired_development_complete_pending_analysis',original_actual_sampled_decisions=48384,new_actual_sampled_decisions=new,actual_total_sampled_RL_decisions=total,automatic_replication=False);write(S/'ledger.json',ledger)
 write(state,dict(status='single_seed_paired_development_complete_pending_analysis',outputs=outputs,development_seed=5201,new_RL_decisions=new,total_sampled_RL_decisions=total,automatic_replication=False,sealed_tests_executed=False))

if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  write(S/'single-seed-status.json',dict(status='failed_saved_no_auto_retry',automatic_replication=False,error=traceback.format_exc()));raise
