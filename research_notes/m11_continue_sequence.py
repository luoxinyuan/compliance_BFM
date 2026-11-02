"""Finite M11 sequence, no watch loop or automatic retry/experiment expansion."""
import argparse,json,subprocess,os,time
from pathlib import Path
from m11_selection import atomic
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');ROOT=Path(__file__).resolve().parents[1];RUN=P/'bridge-v2-owned/runs/m11-continuation-20261102'
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu-index',type=int,required=True);p.add_argument('--after-zero',action='store_true');a=p.parse_args()
 import fcntl
 lock=(RUN/'sequence.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 stages=['fixed','annealed','final-prior','final-fixed','final-annealed','videos','analysis'] if a.after_zero else ['zero','fixed','annealed','final-prior','final-fixed','final-annealed','videos','analysis']
 if (RUN/'sequence-status.json').exists():
  import shutil
  shutil.copyfile(RUN/'sequence-status.json',RUN/f'sequence-status-before-{time.time_ns()}.json')
 record=dict(pid=os.getpid(),status='running',started=time.time(),stages=stages,finished=[]);atomic(RUN/'sequence-status.json',record)
 try:
  for stage in stages:
   record['current']=stage;atomic(RUN/'sequence-status.json',record)
   command=[str(P/'.venv/bin/python'),'-u',str(ROOT/'research_notes/m11_continue_supervisor.py'),'--stage',stage,'--gpu-index',str(a.gpu_index),'--execute-after-owner-gpu-notice']
   with (RUN/('sequence-'+stage+'-'+str(time.time_ns())+'.log')).open('xb') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
   ledger=json.loads((RUN/'compute-ledger.json').read_text());entry=ledger['attempts'][-1]
   if entry.get('result_status')!='completed' or entry.get('resource_boundary') or entry.get('remaining_processes'):raise RuntimeError('Sequence stopped at '+stage+': '+str(entry.get('result_status'))+' '+str(entry.get('resource_boundary')))
   record['finished'].append(stage);atomic(RUN/'sequence-status.json',record)
  record['status']='completed'
 except BaseException as e:
  record['status']='stopped';record['reason']=str(e);raise
 finally:
  record['ended']=time.time();atomic(RUN/'sequence-status.json',record)
if __name__=='__main__':main()
