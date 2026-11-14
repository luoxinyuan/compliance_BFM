"""Single GPU0 supervisor; R2 cumulative inherited ledger; immutable job sidecars."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
S=Path(__file__).resolve().parents[3];R=S/'runtime';P=S.parents[1]
sys.path.insert(0,str(R/'research/interaction_fb'))
from supervision import tree_bytes,stop_owned_group

def write(p,x):
    q=p.with_suffix('.pending.json');q.write_text(json.dumps(x,indent=2));os.replace(q,p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--mode',choices=['mean','sample'],required=True);ap.add_argument('--limit',type=int,default=1600);a=ap.parse_args()
    ledger=json.loads((S/'ledger.json').read_text());prior=ledger['total_charged_upper_s']
    used=sum(j['wall_s'] for j in ledger['job_entries'] if j.get('category')=='qualification')
    assert used+a.limit<=3000 and prior+a.limit<57600
    assert shutil.disk_usage(S).free>=100*2**30
    assert tree_bytes(ledger['output_roots'])<12*2**30
    runtime=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(R),'status','--porcelain'])
    gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(s.split(',')[0].strip()==gpu for s in apps.splitlines()),'GPU0 occupied'
    lock=S/'active-owned-job.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    out=S/'runs'/a.name;side=S/'jobs'/(a.name+'.json');assert not side.exists() and not out.exists()
    cmd=[str(P/'.venv/bin/python'),'-u',str(R/'research/interaction_fb/qualification_v11.py'),'--output',str(out),'--mode',a.mode]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(S/'tmp'))
    start=time.time();rec=dict(name=a.name,category='qualification',runtime=runtime,start=start,supervisor_pid=os.getpid(),command=cmd,gpu_uuid=gpu,limit_s=a.limit,before_s=prior)
    try:
        with (S/'jobs'/(a.name+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=child.pid;write(side,rec)
            while child.poll() is None:
                if time.time()-start>a.limit-30 or shutil.disk_usage(S).free<100*2**30 or tree_bytes(ledger['output_roots'])>12*2**30-256*2**20:
                    rec['guard_stop']='time_or_storage';break
                time.sleep(5)
            rc=stop_owned_group(child,grace=20)
        rec.update(end=time.time(),wall_s=time.time()-start,exit_code=rc,output_bytes=tree_bytes(ledger['output_roots']))
        counter=out/'physics-counter.json';rec['physics_counts']=json.loads(counter.read_text()) if counter.exists() else {}
        result=out/'qualification.json';rec['qualification_status']=json.loads(result.read_text())['status'] if result.exists() else 'missing_result'
        write(side,rec);ledger['job_entries'].append(rec);ledger['total_charged_upper_s']=prior+rec['wall_s'];ledger['status']=rec['qualification_status'];write(S/'ledger.json',ledger)
        print(json.dumps(rec),flush=True)
    finally:
        if lock.exists() and lock.read_text()==str(os.getpid()):lock.unlink()
if __name__=='__main__':main()
