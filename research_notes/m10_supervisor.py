"""Manual GPU-only inference/qualification supervisor; never watches for tasks."""
import argparse,datetime,json,os,resource,signal,subprocess,time
from pathlib import Path
from causal_supervisor import processes

P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
ROOT=Path(__file__).resolve().parents[1]
RUN=P/'bridge-v2-owned/runs/m10-task-appropriate-20261101'


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
from m7_io import write
def gpu(index):
    q=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    return {'uuid':q[0],'free_mib':int(q[1]),'utilization':int(q[2])}


def main():
    assert RUN.name == 'm10-task-appropriate-20261101'
    assert RUN != P/'bridge-v2-owned/runs/issue21-reward-stage-20261030A'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-index',type=int)
    parser.add_argument('--stage',choices=['qualification','validation','collect','train','final','videos'],required=True)
    parser.add_argument('--execute-after-owner-gpu-notice',action='store_true')
    args=parser.parse_args()
    protocol=json.loads((ROOT/'research_notes/m10_protocol.json').read_text())
    if not args.execute_after_owner_gpu_notice:
        print(json.dumps({'status':'prepared_only_awaiting_owner_gpu_notice','resources':protocol['resources']}));return
    assert args.gpu_index is not None
    assert not git('status','--porcelain'), 'Commit the private runtime before execution'
    assert git('branch','--show-current')=='gaoe05/m10-task-appropriate'
    before=gpu(args.gpu_index);assert before['free_mib']>=12288
    import fcntl
    lock=(RUN/'compute.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger_path=RUN/'compute-ledger.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'attempts':[],'no_os_sandbox':True}
    assert not any(a['status']=='running' for a in ledger['attempts'])
    assert not any(a.get('resource_boundary') for a in ledger['attempts']), 'Stage stopped at resource boundary'
    assert not any(a.get('result_status')=='physical-stopped' for a in ledger['attempts']), 'Unsafe selected controller ends this stage; qualification candidates are recorded separately'
    stage_attempts=[a for a in ledger['attempts'] if a.get('stage','inference')==args.stage]
    assert not any(a.get('result_status',a.get('inference_status'))=='completed' for a in stage_attempts), 'No repeat completed stage for selection'
    remaining=protocol['resources']['compute_cumulative_wall_seconds']-sum(a.get('wall_seconds',0) for a in ledger['attempts'])
    assert remaining>0
    attempt=RUN/f'{args.stage}-attempt-{len(stage_attempts)+1:02d}';attempt.mkdir()
    entry={'stage':args.stage,'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':git('rev-parse','HEAD'),'path':str(attempt),'gpu_before':before,'remaining_seconds':remaining,'peak_rss_bytes':0,'peak_gpu_mib':0}
    ledger['attempts'].append(entry);write(ledger_path,ledger)
    env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES=before['uuid'],MUJOCO_GL='disable',BFM_STAGE_SUPERVISED='1',BFM_STAGE_DEADLINE_MONOTONIC=str(time.monotonic()+remaining),PYTHONPATH=str(ROOT),TORCHDYNAMO_DISABLE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',WANDB_MODE='disabled')
    (attempt/'outputs').mkdir()
    env.update(BFM_RECOVERY_OUTPUT=str(attempt/'outputs'),M9_STAGE=args.stage,CCL_DEADLINE_MONOTONIC=env['BFM_STAGE_DEADLINE_MONOTONIC'])
    launcher='m10_render.py' if args.stage=='videos' else 'm10_physics.py'
    if args.stage.startswith('videos'):env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(args.gpu_index))
    command=[str(P/'.venv/bin/python'),'-u',str(ROOT/'research_notes'/launcher)]
    entry['command']=command;write(ledger_path,ledger)
    cache=attempt/'cache';cache.mkdir()
    for k in ['TMPDIR','TEMP','TMP','XDG_CACHE_HOME','MPLCONFIGDIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','HF_HOME','PYTHONPYCACHEPREFIX']:
        d=cache/k;d.mkdir();env[k]=str(d)
    affinity=sorted(os.sched_getaffinity(0))[-8:]
    def limits():
        os.sched_setaffinity(0,affinity);os.nice(10)
        resource.setrlimit(resource.RLIMIT_CPU,(int(remaining*8),int(remaining*8)))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    start=time.monotonic();child=None;boundary=None
    try:
        with (attempt/'stdout.log').open('wb') as stdout,(attempt/'stderr.log').open('wb') as stderr:
            child=subprocess.Popen(command,cwd=attempt,env=env,stdout=stdout,stderr=stderr,start_new_session=True,preexec_fn=limits)
            entry['pid']=child.pid;entry['initial_processes']=processes(child.pid);write(ledger_path,ledger)
            next_gpu=0;next_storage=0
            while child.poll() is None:
                ps=processes(child.pid);rss=sum(p['rss_bytes'] for p in ps)
                entry['peak_rss_bytes']=max(entry['peak_rss_bytes'],rss)
                if rss>32*1024**3:boundary='owned_RSS_over_32GiB'
                if time.monotonic()-start>remaining:boundary='cumulative_compute_wall_limit'
                if time.monotonic()>next_gpu:
                    next_gpu=time.monotonic()+2;g=gpu(args.gpu_index);assert g['uuid']==before['uuid']
                    apps=subprocess.check_output(['nvidia-smi','-i',str(args.gpu_index),'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
                    owned={p['pid'] for p in ps};usage={}
                    for line in apps.splitlines():
                        pid,mem=line.split(', ')
                        if int(pid) in owned:usage[int(pid)]=int(mem)
                    entry['peak_gpu_mib']=max(entry['peak_gpu_mib'],sum(usage.values()))
                    if sum(usage.values())>8192:boundary='owned_VRAM_over_8192MiB'
                    if g['free_mib']<2048:boundary='GPU_free_below_2GiB'
                    write(ledger_path,ledger)
                if time.monotonic()>next_storage:
                    next_storage=time.monotonic()+30
                    entry['output_bytes']=sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file())
                    if entry['output_bytes']>8*1024**3:boundary='continuation_outputs_over_8GiB'
                if boundary:break
                time.sleep(.5)
    finally:
        if child:
            if processes(child.pid):
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
            else:child.wait()
            remaining_ps=processes(child.pid)
            if remaining_ps:os.killpg(child.pid,signal.SIGKILL);time.sleep(.2)
            entry.update(exit_code=child.returncode,remaining_processes=processes(child.pid))
        entry.update(status='finished',wall_seconds=time.monotonic()-start,resource_boundary=boundary)
        result_path=attempt/'outputs/result.json'
        if result_path.exists():
            entry['result_status']=json.loads(result_path.read_text())['status']
            if args.stage=='inference':entry['inference_status']=entry['result_status']
        entry['gpu_after']=gpu(args.gpu_index);write(ledger_path,ledger)
        print(json.dumps(entry,indent=2))


if __name__=='__main__':main()
