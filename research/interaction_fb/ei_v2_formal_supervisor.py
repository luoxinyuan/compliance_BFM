"""Serial formal owner: immutable job sidecars and bounded cumulative accounting.
No optimizer semantics or scientific gates are implemented in this supervisor.
"""
import argparse,fcntl,hashlib,json,os,signal,shutil,subprocess,time,traceback
from pathlib import Path

def write(p,v):
    q=p.with_suffix('.pending');q.write_text(json.dumps(v,indent=2)+'\n');os.replace(q,p)

def size(root):
    total=0
    for directory,dirs,files in os.walk(root):
        for name in files:
            p=Path(directory)/name
            if not p.is_symlink():total+=p.stat().st_size
    return total

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,required=True);parser.add_argument('--mode',choices=['E','I'],required=True);a=parser.parse_args()
    S=a.stage.resolve();P=S.parent.parent;R=S/'runtime';job='formal-'+a.mode+'-01';side=S/'jobs'/f'{job}.json';out=S/'runs'/job
    assert not side.exists() and not out.exists()
    start=time.monotonic();meta=dict(name=job,arm=a.mode,start_epoch=time.time(),supervisor_pid=os.getpid(),status='PREFLIGHT');child=None;held=[];last_ticks=dict(actual=0,reserved=0);charged=False
    def stop():
        if child is not None and child.poll() is None:
            assert Path(f'/proc/{child.pid}/stat').read_text().split()[21]==meta['child_start_ticks']
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=240)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait();meta['forced_kill_after_save_grace']=True
    def ticks():
        nonlocal last_ticks
        p=out/'physics-counter.json'
        if p.exists():
            from research.interaction_fb.ei_v2_counter_lock import locked
            with locked(p):v=json.loads(p.read_text())
            assert 0<=v['actual']<=v['reserved'] and v['actual']>=last_ticks['actual']
            last_ticks=v
        return last_ticks
    try:
        for path in (P/'.dual-track-gpu0.lock',P/'gpu0-experiment.lock'):
            f=open(path,'a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held.append(f)
        L=json.loads((S/'ledger.json').read_text());B=json.loads((S/'locks/contact_goal_ei_v2_formal_budget.json').read_text());policy=json.loads((S/'locks/resource-policy-formal-v1.json').read_text())
        floor=policy['disk_free_floor_GiB']*2**30;reserve=policy['pending_atomic_peak_reservation_GiB']*2**30
        assert L['project_upper_before_formal_s']+B['formal_job_cap_s']<=240*3600
        assert L['charged_s']+20000<=B['formal_job_cap_s'],'NOT_ENOUGH_FORMAL_JOB_BUDGET'
        assert shutil.disk_usage(S).free>=floor+reserve
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=R,text=True).strip()
        meta['runtime']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
        expected=json.loads((S/'locks/runtime-lock.json').read_text());assert meta['runtime']==expected['commit']
        uuid=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==expected['gpu_uuid']
        assert int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())==0
        for name,digest in expected['locks'].items():assert hashlib.sha256((S/'locks'/name).read_bytes()).hexdigest()==digest
        physical_remaining=B['formal_physics_ticks_cap']-L['physics_reserved'];assert physical_remaining>0
        deadline=min(20000.,B['formal_job_cap_s']-L['charged_s'])
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',EI_FORMAL_PHYSICS_CAP=str(min(400000,physical_remaining)))
        cmd=[str(P/'.venv/bin/python'),'-u','-m','research.interaction_fb.ei_v2_formal','--output',str(out),'--mode',a.mode]
        log=open(S/'jobs'/f'{job}.log','w');child=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        meta.update(pid=child.pid,child_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],command=cmd,status='RUNNING',job_cap_s=deadline,physics_cap=int(env['EI_FORMAL_PHYSICS_CAP']));write(side,meta)
        last_check=0
        while child.poll() is None:
            elapsed=time.monotonic()-start;reason=None
            if elapsed>=deadline-300:reason='JOB_CAP_SAVE_RESERVE'
            if (S/'STOP').exists():reason='USER_OR_OWNER_STOP'
            if shutil.disk_usage(S).free<floor+reserve:reason='DISK_COMPLETE_SAVE_RESERVE'
            if elapsed-last_check>=30:
                last_check=elapsed;meta.update(elapsed_s=elapsed,physics=ticks(),new_output_bytes=size(S/'runs')+size(S/'jobs'))
                if meta['new_output_bytes']+reserve>B['formal_new_output_cap_GiB']*2**30:reason='OUTPUT_COMPLETE_SAVE_RESERVE'
                write(side,meta)
            if reason:meta['stop_reason']=reason;stop();break
            time.sleep(2)
        log.close();meta['exit_code']=child.returncode
        result=out/'result.json';payload=json.loads(result.read_text()) if result.exists() else {}
        if isinstance(payload.get('study'),dict):payload=payload['study']
        meta['status']=payload.get('status','EXIT_'+str(child.returncode)) if child.returncode==0 else 'FAILED_EXIT_'+str(child.returncode)
    except BaseException as exc:
        meta.update(status='PREFLIGHT_OR_SUPERVISION_FAILED',error=repr(exc),traceback=traceback.format_exc());stop()
    finally:
        meta.update(wall_s=time.monotonic()-start,end_epoch=time.time())
        try:meta['physics']=ticks()
        except BaseException as exc:meta.update(counter_error=repr(exc),status='COUNTER_UNREADABLE');meta['physics']=dict(last_ticks,reserved=meta.get('physics_cap',last_ticks['reserved']))
        write(side,meta)
        # Lock is still held through the single-writer cumulative ledger commit.
        if held:
            L=json.loads((S/'ledger.json').read_text());L['charged_s']+=meta['wall_s'];L['physics_reserved']+=meta['physics']['reserved'];L['jobs'].append(meta);L['status']=meta['status'];write(S/'ledger.json',L)
        print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
