"""One H0 supervisor; full child wall/IO charged once, GPU0 strictly serial."""
import argparse,json,os,signal,subprocess,time,shutil
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/h0-outcome-address-20261115';R=Path(__file__).resolve().parents[2]
CAPS={'prep':450,'fit':1500,'learn':600,'eval':1800,'closeout':450,'reserve':600}
def atomic(p,d):
 q=p.with_suffix('.pending.json');q.write_text(json.dumps(d,indent=2));os.replace(q,p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=list(CAPS),required=True);p.add_argument('--name',required=True);p.add_argument('--limit',type=float,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();assert a.command[0]=='--';a.command=a.command[1:]
 ledger=json.loads((S/'ledger.json').read_text());used=sum(x['wall_s'] for x in ledger['jobs'] if x['phase']==a.phase)
 assert used+a.limit<=CAPS[a.phase] and ledger['charged_upper_s']+a.limit<=5400
 assert shutil.disk_usage(S).free>=100*2**30
 assert not subprocess.check_output(['git','-C',str(R),'status','--porcelain'])
 runtime=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
 # No second heavy process or ledger writer may share this stage.
 for old in (S/'jobs').glob('*.json'):
  x=json.loads(old.read_text())
  if 'end' not in x and x.get('supervisor_pid'):
   try:os.kill(x['supervisor_pid'],0)
   except ProcessLookupError:pass
   else:raise RuntimeError('H0_OWN_JOB_STILL_RUNNING')
 rec=dict(name=a.name,phase=a.phase,start=time.time(),runtime=runtime,limit_s=a.limit,supervisor_pid=os.getpid(),command=a.command)
 side=S/'jobs'/(a.name+'.json');assert not side.exists();start=time.monotonic();log=open(side.with_suffix('.log'),'xb');child=subprocess.Popen(a.command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=child.pid;atomic(side,rec)
 try:
  while child.poll() is None:
   if time.monotonic()-start>a.limit or shutil.disk_usage(S).free<100*2**30:
    rec['stop_reason']='job_or_storage_guard';os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
    break
   time.sleep(1)
  child.wait()
 finally:
  log.close();rec.update(end=time.time(),wall_s=time.monotonic()-start,exit_code=child.returncode);atomic(side,rec)
  d=json.loads((S/'ledger.json').read_text());d['jobs'].append(rec);d['charged_upper_s']+=rec['wall_s'];d['status']='JOB_COMPLETE' if child.returncode==0 else 'JOB_FAILED';atomic(S/'ledger.json',d)
  op=P/'experiments/r2-v11-qentry-20261114/ledger.json';old=json.loads(op.read_text());old['job_entries'].append(dict(rec,category='Issue66-H0'));old['total_charged_upper_s']+=rec['wall_s'];old['status']='ISSUE66_H0_ACTIVE';atomic(op,old)
 print(json.dumps({k:v for k,v in rec.items() if k!='command'}),flush=True);raise SystemExit(child.returncode)
if __name__=='__main__':main()
