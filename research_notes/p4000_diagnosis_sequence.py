"""One bounded post-training diagnostic after exact owned pair exits, not new training."""
import time,traceback
from pathlib import Path
from p4000_critic_sequence import owned_alive,resource_check,read,write
from p4000_sequence import job,remaining
ROOT=Path(__file__).resolve().parents[1];S=ROOT.parent

def main():
 state=S/'mean-diagnosis-status.json';owned=read(S/'single-seed-process.json');deadline=time.monotonic()+12000
 write(state,dict(status='waiting_for_owned_paired_training_to_exit',no_new_training=True))
 while owned_alive(owned):
  assert time.monotonic()<deadline,'Finite diagnostic wait deadline exceeded';time.sleep(15)
 assert read(S/'single-seed-status.json')['status']=='single_seed_paired_development_complete_pending_analysis'
 assert remaining()>1100,('Insufficient diagnostic600 plus500report reserve',remaining())
 resource_check();write(state,dict(status='running_fixed_state_and_short_physics_diagnosis',max_job_seconds=600,new_RL_training_decisions=0))
 job('mean-diagnosis-01',ROOT,'p4000_mean_diagnosis.py',600,['--output',str(S/'runs/mean-diagnosis-01')])
 result=read(S/'runs/mean-diagnosis-01/diagnosis-result.json');assert result['new_RL_training_decisions']==0
 write(state,dict(status='diagnosis_complete_pending_analysis',result=result,automatic_new_training=False))

if __name__=='__main__':
 try:main()
 except BaseException:
  write(S/'mean-diagnosis-status.json',dict(status='failed_saved_no_auto_retry',error=traceback.format_exc(),automatic_new_training=False));raise
