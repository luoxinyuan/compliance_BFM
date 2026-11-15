"""Single owned GPU0 job, QA1 and inherited R2 budget accounting."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=ROOT.parents[2]
S=P/'experiments/r2-g17-qa1-20261115';OLD=P/'experiments/r2-v11-qentry-20261114'
sys.path.insert(0,str(ROOT/'research/interaction_fb'))
from supervision import tree_bytes,stop_owned_group
def write(p,x):
    q=p.with_suffix('.pending.json');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--limit',type=int,default=1500);p.add_argument('--only-c',action='store_true');a=p.parse_args()
    started=time.time();ledger=json.loads((S/'ledger.json').read_text());total=json.loads((OLD/'ledger.json').read_text())
    assert ledger['charged_upper_s']+a.limit+300<=3600 and total['total_charged_upper_s']+a.limit+300<=57600
    assert tree_bytes([S])<1024**3-256*1024**2 and shutil.disk_usage(S).free>=100*1024**3
    runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
    gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(x.split(',')[0].strip()==gpu for x in apps.splitlines()),'GPU0 occupied'
    owned=S/'active-owned-job.lock';fd=os.open(owned,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    out=S/'runs'/a.name;assert not out.exists();side=S/'jobs'/(a.name+'.json');assert not side.exists()
    cmd=[str(P/'.venv/bin/python'),'-u',str(ROOT/'research/interaction_fb/qa1_trial.py'),'--output',str(out)]
    if a.only_c:cmd.append('--only-c')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(S/'tmp'))
    prior_physics=sum(j.get('physics_counts',{}).get('reserved',0) for j in ledger['jobs'])
    assert prior_physics<18000
    rec=dict(name=a.name,start=started,runtime=runtime,command=cmd,supervisor_pid=os.getpid(),gpu_uuid=gpu,limit_s=a.limit,prior_reserved_ticks=prior_physics)
    child=None
    try:
        with (S/'jobs'/(a.name+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=child.pid;write(side,rec)
            while child.poll() is None:
                counter=out/'physics-counter.json'
                try:counts=json.loads(counter.read_text()) if counter.exists() else dict(actual=0,reserved=0)
                except json.JSONDecodeError:counts=dict(actual=0,reserved=0)
                if time.time()-started>a.limit-30 or tree_bytes([S])>1024**3-64*1024**2 or shutil.disk_usage(S).free<100*1024**3 or prior_physics+counts['reserved']>=18000:
                    rec['guard_stop']='time_storage_or_physics';break
                time.sleep(5)
            rc=stop_owned_group(child,grace=20)
        rec.update(end=time.time(),wall_s=time.time()-started,exit_code=rc,output_bytes=tree_bytes([S]))
        cp=out/'physics-counter.json';rec['physics_counts']=json.loads(cp.read_text()) if cp.exists() else {}
        rp=out/'qa1-result.json';rec['status']=json.loads(rp.read_text())['status'] if rp.exists() else 'STOP_FIRST_FAILURE'
        write(side,rec);ledger['jobs'].append(rec);ledger['charged_upper_s']+=rec['wall_s'];ledger['status']=rec['status'];write(S/'ledger.json',ledger)
        total['job_entries'].append(dict(rec,category='G17-QA1_fixed_replay'));total['total_charged_upper_s']+=rec['wall_s'];total['status']=rec['status'];write(OLD/'ledger.json',total)
        print(json.dumps(rec),flush=True)
    finally:
        if child is not None and child.poll() is None:stop_owned_group(child,grace=20)
        if owned.exists() and owned.read_text()==str(os.getpid()):owned.unlink()
if __name__=='__main__':main()
