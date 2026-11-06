"""Stage-owned bounded subprocess ledger; no management of other processes."""
import argparse,json,os,subprocess,time,signal
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--ledger',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--limit',type=float,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
def save(obj):
 t=a.ledger.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(a.ledger)
l=json.loads(a.ledger.read_text());spent=sum(j.get('elapsed_s',0) for j in l['jobs']);assert spent+a.limit<=l['authorized_limits']['job_wall_seconds'];cmd=a.command[1:] if a.command[:1]==['--'] else a.command
assert not any(j.get('status')=='running' for j in l['jobs']), 'single GPU sequential stage; no competing managed jobs'
j={'name':a.name,'command':cmd,'started':time.time(),'status':'running','limit_s':a.limit,'native_mainline':True};l['jobs'].append(j)
f=open(a.ledger.parent/(a.name+'.log'),'x',buffering=1);t=time.monotonic();proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);j['pid']=proc.pid;j['start_ticks']=Path(f'/proc/{proc.pid}/stat').read_text().split()[21];save(l)
try:rc=proc.wait(timeout=a.limit)
except subprocess.TimeoutExpired:
 os.killpg(proc.pid,signal.SIGTERM)
 try:rc=proc.wait(timeout=20)
 except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
 j['timed_out']=True
finally:
 j.update(status='completed' if proc.returncode==0 else 'failed',returncode=proc.returncode,elapsed_s=time.monotonic()-t,ended=time.time());save(l);f.close()
raise SystemExit(proc.returncode)
