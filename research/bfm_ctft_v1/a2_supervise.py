"""One approved A2 attempt: serial children, exact ownership, aggregate limits."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--accounting-start',type=float,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False,parents=True);jobs=[];status='RUNNING'
    def write(name,x):(a.output/name).write_text(json.dumps(x,indent=2)+'\n')
    write('launch.json',dict(pid=os.getpid(),accounting_start=a.accounting_start,deadline=a.accounting_start+1800,
        runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        scope='one approved A2 run, no new physics or formal U/T',output_cap_bytes=12*2**30))
    def used_bytes():
        return sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())
    try:
        commands=[('joint',[sys.executable,'-u',str(ROOT/'research/bfm_ctft_v1/native_joint_probe.py'),
                           '--output',str(a.output/'joint'),'--a2-checkpoint']),
                  ('restore',[sys.executable,'-u',str(ROOT/'research/bfm_ctft_v1/a2_resume.py'),
                              '--package',str(a.output/'joint/learner-checkpoint-4'),'--output',str(a.output/'restore')])]
        for label,cmd in commands:
            assert time.time()<a.accounting_start+1740,'A2 budget requires60s report reserve'
            begun=time.time()
            with (a.output/(label+'.log')).open('w') as log:
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
                row=dict(label=label,pid=proc.pid,command=cmd,started_unix=begun);jobs.append(row);write('jobs.json',jobs)
                try:
                    while True:
                        try:code=proc.wait(timeout=2);break
                        except subprocess.TimeoutExpired:
                            assert time.time()<a.accounting_start+1740,'A2 job time limit reached'
                            assert used_bytes()<12*2**30,'A2 aggregate output limit reached'
                except BaseException:
                    if proc.poll() is None:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    raise
            row.update(exit_code=code,outer_wall_s=time.time()-begun);write('jobs.json',jobs)
            assert code==0,('Approved A2 child failed; no further repair/flag changes',label,code)
        status='A2_PASSED_LEARNER_ONLY'
        write('result.json',dict(status=status,jobs=jobs,physical_task_restore=False,
            real_new_physics_ticks=0,formal_U_T_training=False,formal_N=None))
    except BaseException as exc:
        status='A2_STOPPED';write('failure.json',dict(status=status,error_type=type(exc).__name__,error=str(exc),jobs=jobs))
        raise
    finally:
        write('cost.json',dict(status=status,outer_accounting_s=time.time()-a.accounting_start,
            includes_preparation=True,includes_serial_children_not_added_again=True,output_bytes=used_bytes(),jobs=jobs,
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__=='__main__':main()
