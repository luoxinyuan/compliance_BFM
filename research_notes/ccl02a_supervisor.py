"""CCL-02A full 6s bounded right-arm teacher study; never consumes Bridge/GitHub queue state."""
import argparse, datetime, fcntl, hashlib, json, os, resource, signal, subprocess, sys, time
from pathlib import Path
PROJECT = Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
SOURCE = PROJECT / 'repo'
WORKTREE = PROJECT / 'bridge-v2-owned/trees/issue-21-ccl02a/repo'
MODEL_VIEW = PROJECT / 'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view'
LAUNCHER = 'research_notes/ccl02a_teacher.py'
BASE = '6023c6e257fab8fe58d13d9207fb5e85c32ee9c3'
RECOVERED = 'bb0833853c8e5cad1369d691b531972004635ff7'
GIB=1024**3
def sha(data):
    return hashlib.sha256(data).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))

def git(root, *args):
    return subprocess.check_output(['/usr/bin/git','-c','core.hooksPath=/dev/null','-c','core.fsmonitor=false','-C',str(root),*args],
        env={'PATH':'/usr/bin:/bin','GIT_OPTIONAL_LOCKS':'0','GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null'},text=True,timeout=30).strip()

def snapshot():
    return {'head':git(SOURCE,'rev-parse','HEAD'),'branch':git(SOURCE,'branch','--show-current'),
        'status':git(SOURCE,'status','--porcelain'),
        'tracked':{f:sha((SOURCE/f).read_bytes()) for f in git(SOURCE,'ls-files').splitlines()},
        'worktrees':git(SOURCE,'worktree','list','--porcelain')}

def processes(pgid):
    out=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2]) != pgid: continue
            out.append({'pid':int(p.name),'start_ticks':fields[19],
                'rss_bytes':int(fields[21])*os.sysconf('SC_PAGE_SIZE'),
                'vms_bytes':int(fields[20]),'state':fields[0]})
        except (FileNotFoundError,PermissionError,ProcessLookupError): pass
    return out

def gpu_snapshot():
    uuid='GPU-71345fda-5d4b-86df-ca85-6f66299aa7b0'
    memory=subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(', ')
    assert memory[0]==uuid, 'GPU5 UUID changed'
    apps=subprocess.check_output(['nvidia-smi','-i','5','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    ps=[]
    for row in apps.splitlines():
        parts=[x.strip() for x in row.split(',')]
        if parts[0]==uuid:ps.append({'pid':int(parts[1]),'used_mib':int(parts[2])})
    return {'uuid':uuid,'used_mib':int(memory[1]),'processes':ps}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    assert args.run_id.startswith('issue21-ccl02a-') and all(c.isalnum() or c=='-' for c in args.run_id)
    assert WORKTREE.resolve()==WORKTREE and SOURCE.resolve()==SOURCE
    assert git(WORKTREE,'branch','--show-current')=='gaoe05/ccl02a-teacher-six-seconds'
    assert git(WORKTREE,'merge-base',RECOVERED,'HEAD')==RECOVERED
    assert not git(WORKTREE,'status','--porcelain'), 'Runtime worktree must be clean'
    before=snapshot()
    assert before['head']==BASE and before['branch']=='gaoe05/research' and not before['status']
    run=PROJECT/'bridge-v2-owned/runs'/args.run_id
    run.mkdir(mode=0o700,exist_ok=True)
    lock=(run/'manual.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger_path=run/'ledger.json'
    if ledger_path.exists():ledger=json.loads(ledger_path.read_text())
    else:
        ledger={'run_id':args.run_id,'authorization':'Owner explicitly approved Issue21 CCL-02A fixed four 6s development tasks, baseline vs bounded right-arm teacher; no PPO/projector/student training; GPU5 sharing if VRAM sufficient, never interrupt SoftSONIC; CPU4/RSS8GiB/VRAM3GiB/cumulative1800s',
            'resources':{'cpu_count':4,'memory_mib':8192,'gpu_count':1,'gpu_physical_index':5,'gpu_memory_mib':3072,'wall_seconds':1800},'attempts':[],
            'os_isolation':False,'memory_enforcement':'Owned process-group RSS sum watchdog, 0.1 s, not OS hard quota'}
        write(run/'source-before.json',before)
    assert not any(x['state']=='running' for x in ledger['attempts'])
    for prior in ledger['attempts']:
        p=Path(prior['path'])/'outputs/result.json'
        if p.exists():
            r=json.loads(p.read_text())
            assert r.get('status')!='completed', 'This fixed teacher study already completed; do not repeat for selection'
    remaining=1800-sum(x.get('wall_seconds',0) for x in ledger['attempts'])
    assert remaining>0
    number=len(ledger['attempts'])+1
    attempt=run/f'attempt-{number:02d}';attempt.mkdir()
    outputs=attempt/'outputs';outputs.mkdir()
    (attempt/'source').mkdir()
    for f in ['research_notes/'+n for n in ['ccl02a_teacher.py','test_ccl02a_teacher.py','ccl02a_supervisor.py','frozen_bfm_placement.py']]:
        target=attempt/'source'/f;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((WORKTREE/f).read_bytes())
    write(attempt/'manifest.json',{'worktree_head':git(WORKTREE,'rev-parse','HEAD'),
        'worktree_status':git(WORKTREE,'status','--porcelain'),
        'source_hashes':{f:sha((WORKTREE/f).read_bytes()) for f in git(WORKTREE,'ls-files').splitlines()},
        'launcher_sha256':sha((WORKTREE/LAUNCHER).read_bytes()),
        'protocol':'Original Issue19 frozen BFM, four paired 6s dev tasks, original bounded right-arm correction with predeclared causal 0.2s release; no policy or projector training; terminal sealed',
        'model_view':str(MODEL_VIEW),'source_before':before})
    cache=outputs/'cache';cache.mkdir()
    env={'PATH':str(PROJECT/'.venv/bin')+':/usr/bin:/bin',
         'LANG':'C.UTF-8','CUDA_VISIBLE_DEVICES':'GPU-71345fda-5d4b-86df-ca85-6f66299aa7b0','MUJOCO_GL':'disable','PYTHONDONTWRITEBYTECODE':'1',
         'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
         'MALLOC_ARENA_MAX':'2','BFM_MODEL_FOLDER':str(MODEL_VIEW),'BFM_RECOVERY_OUTPUT':str(outputs),
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','WANDB_MODE':'disabled',
         'CCL_OUTPUT':str(outputs),'CCL_DEADLINE_MONOTONIC':str(time.monotonic()+remaining),
         'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
    for name in ['TMPDIR','TEMP','TMP','XDG_CACHE_HOME','XDG_CONFIG_HOME','MPLCONFIGDIR','NUMBA_CACHE_DIR',
                 'TORCH_HOME','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','HF_HOME','PYTHONPYCACHEPREFIX']:
        directory=cache/name;directory.mkdir();env[name]=str(directory)
    affinity=sorted(os.sched_getaffinity(0))[:4]
    os.sched_setaffinity(0,affinity)
    def limits():
        os.sched_setaffinity(0,affinity)
        resource.setrlimit(resource.RLIMIT_CPU,(int(remaining*4),int(remaining*4)))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        resource.setrlimit(resource.RLIMIT_FSIZE,(32*1024**2,32*1024**2))
    bootstrap='''import json,os,pathlib,runpy,sys
output=os.path.realpath(os.environ['BFM_RECOVERY_OUTPUT'])
def writable(x):
 if isinstance(x,int):return
 p=os.path.realpath(os.fsdecode(x))
 if p!=output and not p.startswith(output+os.sep):raise PermissionError('Experiment writes limited to owned output: '+p)
def guard(event,args):
 if event=='open':
  value,mode,flags=args
  if (isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):writable(value)
 elif event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.chown','os.utime','os.truncate'):writable(args[0])
 elif event in ('os.rename','os.replace'):writable(args[0]);writable(args[1])
 elif event in ('socket.connect','socket.bind') and isinstance(args[1],tuple):raise PermissionError('No network in bounded study')
 elif event in ('os.symlink','os.link','os.system'):raise PermissionError('No link/shell actions in bounded study')
sys.addaudithook(guard)
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    (attempt/'bootstrap.py').write_text(bootstrap)
    gpu_before=gpu_snapshot();write(attempt/'gpu-before.json',gpu_before)
    free=int(subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>=1024+2048, 'Not enough free GPU5 memory for at least 1GiB own budget plus 2GiB reserve; leave others alone'
    own_limit=min(3072,free-2048)
    entry={'number':number,'state':'running','script_sha256':sha((WORKTREE/LAUNCHER).read_bytes()),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'path':str(attempt),'affinity':affinity,'remaining_wall_seconds':remaining,'gpu_memory_limit_mib':own_limit,'gpu_free_mib_at_launch':free,'peak_rss_bytes':0,'peak_vms_bytes':0}
    ledger['attempts'].append(entry);write(ledger_path,ledger)
    started=time.monotonic(); child=None;reason=None;next_gpu_check=0.;entry['peak_gpu_mib']=0
    try:
        with (attempt/'stdout.log').open('wb') as stdout,(attempt/'stderr.log').open('wb') as stderr,(attempt/'telemetry.jsonl').open('w') as telemetry:
            child=subprocess.Popen([str(PROJECT/'.venv/bin/python'),'-I','-B',str(attempt/'bootstrap.py'),str(WORKTREE/LAUNCHER)],
                cwd=attempt/'outputs',env=env,stdout=stdout,stderr=stderr,start_new_session=True,preexec_fn=limits)
            entry.update(pid=child.pid,pgid=child.pid,initial_processes=processes(child.pid));write(ledger_path,ledger)
            print(json.dumps({'launched':entry}),flush=True)
            while child.poll() is None:
                ps=processes(child.pid);rss=sum(x['rss_bytes'] for x in ps)
                entry['peak_rss_bytes']=max(entry['peak_rss_bytes'],rss)
                entry['peak_vms_bytes']=max(entry['peak_vms_bytes'],sum(x['vms_bytes'] for x in ps))
                telemetry.write(json.dumps({'seconds':time.monotonic()-started,'rss_bytes':rss,'processes':ps})+'\n');telemetry.flush()
                if time.monotonic()>next_gpu_check:
                    gpu=gpu_snapshot();next_gpu_check=time.monotonic()+5
                    own={p['pid'] for p in ps}
                    own_usage=sum({p['pid']:p['used_mib'] for p in gpu['processes'] if p['pid'] in own}.values())
                    entry['peak_gpu_mib']=max(entry['peak_gpu_mib'],own_usage)
                    free=int(subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                    if own_usage>own_limit:reason='owned_GPU5_memory_exceeds_attempt_budget'
                    if free<2048:reason='GPU5_free_memory_below_2GiB_stop_own_group_only'
                if rss>8*GIB:reason='owned_group_RSS_exceeds_8_GiB'
                if time.monotonic()-started>remaining:reason='cumulative_wall_budget_exhausted'
                if reason:break
                time.sleep(.1)
    finally:
        if child:
            ps=processes(child.pid)
            if ps:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
            else:child.wait(timeout=5)
            deadline=time.monotonic()+5
            while processes(child.pid) and time.monotonic()<deadline:time.sleep(.1)
            if processes(child.pid):os.killpg(child.pid,signal.SIGKILL);time.sleep(.2)
            entry.update(exit_code=child.returncode,remaining_processes=processes(child.pid))
        entry.update(state='finished',wall_seconds=time.monotonic()-started,resource_boundary=reason)
        write(attempt/'gpu-after.json',gpu_snapshot())
        after=snapshot();write(run/'source-after.json',after)
        entry['shared_source_unchanged']=before==after
        write(ledger_path,ledger)
        write(attempt/'supervisor-result.json',entry)
        print(json.dumps({'finished':entry}),flush=True)
        for f in ['stdout.log','stderr.log']:
            print(f+'\n'+(attempt/f).read_text(errors='replace')[-3500:],flush=True)

if __name__=='__main__':
    main()
