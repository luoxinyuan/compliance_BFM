"""Bounded serial full native-update/fresh-process recovery qualification.

Engineering only: no physical collection, no formal U/T initialization reuse.
This has a distinct identity and cost from the closed A2 attempt.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wall-limit-s',type=int,default=1200)
    p.add_argument('--trace-package',type=Path);a=p.parse_args()
    started=time.time();a.output.mkdir(exist_ok=False,parents=True);jobs=[];status='RUNNING'
    def write(name,x):(a.output/name).write_text(json.dumps(x,indent=2)+'\n')
    def used():return sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())
    write('launch.json',dict(pid=os.getpid(),started_unix=started,deadline=started+a.wall_limit_s,
          runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
          scope='isolated strict engineering regression; not A2 rerun or formal U/T'))
    try:
        commands=[('joint',[sys.executable,'-u',str(ROOT/'research/bfm_ctft_v1/native_joint_probe.py'),
                    '--output',str(a.output/'joint'),'--a2-checkpoint']),
                  ('restore',[sys.executable,'-u',str(ROOT/'research/bfm_ctft_v1/a2_resume.py'),
                    '--package',str(a.output/'joint/learner-checkpoint-4'),'--output',str(a.output/'restore')])]
        if a.trace_package:
            commands=[('joint',[sys.executable,'-u',str(ROOT/'research/bfm_ctft_v1/native_joint_probe.py'),
                        '--output',str(a.output/'joint'),'--reuse-package',str(a.trace_package)]),
                      ('restore',[sys.executable,'-u',str(ROOT/'research/bfm_ctft_v1/a2_resume.py'),
                        '--package',str(a.trace_package),'--output',str(a.output/'restore'),
                        '--reference',str(a.output/'joint/uninterrupted-update5.json')])]
        for label,command in commands:
            begun=time.time()
            with (a.output/(label+'.log')).open('x') as log:
                env=os.environ.copy()
                if a.trace_package:env['CTFT_LEARNER_BOUNDARY_TRACE']=str(a.output/label/'trace')
                proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True,env=env)
                row=dict(label=label,pid=proc.pid,command=command,started_unix=begun);jobs.append(row);write('jobs.json',jobs)
                try:
                    while True:
                        try:code=proc.wait(timeout=2);break
                        except subprocess.TimeoutExpired:
                            assert time.time()<started+a.wall_limit_s-30,'Qualification local wall stop'
                            assert used()<12*2**30,'Qualification local output stop'
                            assert os.statvfs(a.output).f_bavail*os.statvfs(a.output).f_frsize>=100*2**30,'Shared free-space guard'
                except BaseException:
                    if proc.poll() is None:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    raise
            row.update(exit_code=code,outer_wall_s=time.time()-begun);write('jobs.json',jobs)
            assert code==0,('Qualification child failed',label,code)
        status='WARM_VS_COLD_TRACE_EXACT' if a.trace_package else 'LEARNER_EXACT_RESTORE_PASSED'
        write('result.json',dict(status=status,complete_task_restore=False,new_physics_ticks=0,formal_updates=0))
    except BaseException as e:
        status='ENGINEERING_STOPPED';write('failure.json',dict(type=type(e).__name__,error=str(e)));raise
    finally:
        write('cost.json',dict(status=status,outer_wall_s=time.time()-started,serial_children_not_added_twice=True,
             jobs=jobs,output_bytes=used(),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__=='__main__':main()
