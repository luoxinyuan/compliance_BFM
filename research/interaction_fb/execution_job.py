"""Owned single-GPU bounded job, cumulative R2 engineering ledger and exact child PID."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

S=Path(__file__).resolve().parents[3]
R=S/'runtime';P=S.parents[1]
sys.path.insert(0,str(R/'research/interaction_fb'))
from supervision import tree_bytes,stop_owned_group


def write(path,value):
    tmp=path.with_suffix('.pending.json');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True)
    parser.add_argument('--case-index',type=int);parser.add_argument('--limit',type=int,default=600)
    parser.add_argument('--training',action='store_true');parser.add_argument('--evaluate',action='store_true');parser.add_argument('--arm',choices=['E','I'],default='E');parser.add_argument('--resume',type=Path);parser.add_argument('--qualification',action='store_true');parser.add_argument('--zero-u',action='store_true')
    args=parser.parse_args()
    pidfile=S/'jobs'/(args.name+'-supervisor.pid')
    launch_start=pidfile.stat().st_mtime if pidfile.exists() else time.time()
    ledger=json.loads((S/'ledger.json').read_text());before=ledger['total_charged_upper_s']
    assert before+args.limit+120<=3600
    assert shutil.disk_usage(S).free>=100*2**30 and tree_bytes([str(S)])<12*2**30
    runtime=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(R),'status','--porcelain'])
    gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(line.split(',')[0].strip()==gpu for line in apps.splitlines()),'GPU0 occupied'
    side=S/'jobs'/(args.name+'.json');assert not side.exists()
    lock=S/'active-owned-job.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    output=S/'runs'/args.name
    assert not args.training
    entry='execution_qualification.py' if args.qualification else 'execution_preflight.py'
    cmd=[str(P/'.venv/bin/python'),'-u',str(R/'research/interaction_fb'/entry),'--output',str(output)]
    if args.zero_u:
        assert args.qualification
        cmd+=['--zero-u']
    if args.case_index is not None:cmd+=['--case-index',str(args.case_index)]
    if not args.qualification:cmd+=['--arm',args.arm]
    if args.resume:cmd+=['--resume',str(args.resume)]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(S/'tmp'))
    start=launch_start;record=dict(name=args.name,supervisor_pid=os.getpid(),start=start,runtime=runtime,
                                 command=cmd,gpu_uuid=gpu,before_job_s=before,limit_s=args.limit)
    try:
        with (S/'jobs'/(args.name+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['pid']=child.pid;write(side,record)
            while child.poll() is None:
                wall=time.time()-start
                counter=output/'physics-counter.json'
                ticks=json.loads(counter.read_text()).get('reserved',0) if counter.exists() else 0
                old_ticks=sum(x.get('physics_ticks',0) for x in ledger['job_entries'])
                if wall>args.limit or before+wall>=3480 or tree_bytes([str(S)])>=12*2**30-128*2**20 or shutil.disk_usage(S).free<100*2**30:
                    record['guard_stop']='time_or_storage_boundary';break
                time.sleep(5)
            rc=stop_owned_group(child,grace=20)
        counter=output/'physics-counter.json'
        record['physics_ticks']=json.loads(counter.read_text()).get('reserved',0) if counter.exists() else 0
        record.update(exit_code=rc,end=time.time(),wall_s=time.time()-start,output_bytes=tree_bytes([str(S)]))
        write(side,record)
        ledger['job_entries'].append(record)
        ledger['total_charged_upper_s']=before+record['wall_s']
        ledger['remaining_job_s_upper']=3600-ledger['total_charged_upper_s']
        ledger['gpu_jobs']+=1;ledger['status']='JOB_FINISHED' if rc==0 else 'JOB_FAILED_PRESERVED'
        ledger['last_runtime']=runtime
        write(S/'ledger.json',ledger)
        assert rc==0 and 'guard_stop' not in record,record
        print(json.dumps(record),flush=True)
    finally:
        if lock.exists() and lock.read_text()==str(os.getpid()):lock.unlink()


if __name__=='__main__':main()
