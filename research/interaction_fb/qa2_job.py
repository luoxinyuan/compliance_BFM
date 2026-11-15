"""Isolated QA2 job supervisor. Full subprocess wall/IO counted once."""
import argparse,json,os,signal,subprocess,time,shutil
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/r2-qa2-value-20261115'
R=Path(__file__).resolve().parents[2]
CAPS={'P0':900,'P1':900,'V1':1500,'V2':1700,'P2':1800,'A0':1000,'reserve':1200}
def atomic(p,d):
    q=p.with_suffix('.pending.json');q.write_text(json.dumps(d,indent=2));os.replace(q,p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=list(CAPS),required=True);p.add_argument('--name',required=True);p.add_argument('--limit',type=float,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    assert a.command and a.command[0]=='--';a.command=a.command[1:]
    assert shutil.disk_usage(S).free>=100*1024**3
    ledger=json.loads((S/'ledger.json').read_text());used=sum(x['wall_s'] for x in ledger['jobs'] if x.get('phase')==a.phase)
    assert used+a.limit<=CAPS[a.phase] and ledger['charged_upper_s']+a.limit<=9000
    side=S/'jobs'/f'{a.name}.json';assert not side.exists()
    runtime=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(R),'status','--porcelain'],text=True).strip()
    rec=dict(name=a.name,phase=a.phase,runtime=runtime,start=time.time(),limit_s=a.limit,command=a.command,supervisor_pid=os.getpid())
    t=time.monotonic();log=open(S/'jobs'/f'{a.name}.log','xb');child=subprocess.Popen(a.command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=child.pid;atomic(side,rec)
    try:
        while child.poll() is None:
            if time.monotonic()-t>a.limit or shutil.disk_usage(S).free<100*1024**3:
                rec['stop_reason']='job_time_or_shared_disk_guard';os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                break
            time.sleep(1)
        child.wait()
    finally:
        log.close();rec.update(end=time.time(),wall_s=time.monotonic()-t,exit_code=child.returncode);atomic(side,rec)
        ledger=json.loads((S/'ledger.json').read_text());ledger['jobs'].append(rec);ledger['charged_upper_s']+=rec['wall_s'];ledger['status']='JOB_COMPLETE' if child.returncode==0 else 'JOB_FAILED';atomic(S/'ledger.json',ledger)
        oldp=P/'experiments/r2-v11-qentry-20261114/ledger.json';old=json.loads(oldp.read_text());old['job_entries'].append(dict(rec,category='Issue65-QA2'));old['total_charged_upper_s']+=rec['wall_s'];atomic(oldp,old)
    print(json.dumps({k:v for k,v in rec.items() if k!='command'}),flush=True)
    raise SystemExit(child.returncode)
if __name__=='__main__':main()
