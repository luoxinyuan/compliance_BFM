"""Bounded manual CPU anchor correction and action replay; never consumes Bridge/GitHub queue state."""
import argparse, datetime, fcntl, hashlib, json, os, resource, signal, subprocess, sys, time
from pathlib import Path
PROJECT = Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
SOURCE = PROJECT / 'repo'
WORKTREE = PROJECT / 'bridge-v2-owned/trees/issue-19-a8187804319761a4/repo'
MODEL_VIEW = PROJECT / 'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view'
LAUNCHER = 'research_notes/right_wrist_replay_minusx.py'
BASE = '6023c6e257fab8fe58d13d9207fb5e85c32ee9c3'
RECOVERED = 'b6ba7a37c0083d13ecb7dbdc223746e212097ac5'
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

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    assert args.run_id.startswith('issue19-wrist-replay-minusx-') and all(c.isalnum() or c=='-' for c in args.run_id)
    assert WORKTREE.resolve()==WORKTREE and SOURCE.resolve()==SOURCE
    assert git(WORKTREE,'branch','--show-current')=='gaoe05/lpp-issue-19'
    assert git(WORKTREE,'merge-base',RECOVERED,'HEAD')==RECOVERED
    before=snapshot()
    assert before['head']==BASE and before['branch']=='gaoe05/research' and not before['status']
    run=PROJECT/'bridge-v2-owned/runs'/args.run_id
    run.mkdir(mode=0o700,exist_ok=True)
    lock=(run/'manual.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger_path=run/'ledger.json'
    if ledger_path.exists():ledger=json.loads(ledger_path.read_text())
    else:
        ledger={'run_id':args.run_id,'authorization':'Owner requested full no-force replay gate then single -X; autonomous technical fixes',
            'resources':{'cpu_count':4,'memory_mib':8192,'gpu_count':0,'wall_seconds':3600},'attempts':[],
            'os_isolation':False,'memory_enforcement':'Owned process-group RSS sum watchdog, 0.1 s, not OS hard quota'}
        write(run/'source-before.json',before)
    assert not any(x['state']=='running' for x in ledger['attempts'])
    for prior in ledger['attempts']:
        p=Path(prior['path'])/'outputs/result.json'
        if p.exists():
            r=json.loads(p.read_text())
            assert r.get('scientific_outcome') not in ('no_force_replay_and_minus_x_completed','minus_x_negative'), 'Scientific outcome stops this pilot'
    remaining=3600-sum(x.get('wall_seconds',0) for x in ledger['attempts'])
    assert remaining>0
    number=len(ledger['attempts'])+1
    attempt=run/f'attempt-{number:02d}';attempt.mkdir()
    outputs=attempt/'outputs';outputs.mkdir()
    for f in ['right_wrist_replay_minusx.py','right_wrist_replay_minusx_supervisor.py','test_right_wrist_replay_minusx.py','right_wrist_anchor.py','right_wrist_pair.py','g1_action_contract_check.py']:
        (attempt/f).write_bytes((WORKTREE/'research_notes'/f).read_bytes())
    write(attempt/'manifest.json',{'worktree_head':git(WORKTREE,'rev-parse','HEAD'),
        'worktree_status':git(WORKTREE,'status','--porcelain'),
        'source_hashes':{f:sha((WORKTREE/f).read_bytes()) for f in git(WORKTREE,'ls-files').splitlines()},
        'launcher_sha256':sha((WORKTREE/LAUNCHER).read_bytes()),
        'protocol':'seed=0; 6s no-force and no-force replay; exact replay gates single 6s -X; no-force t1s anchor; 3cm -X; K80 D8 cap5N; on1s ramp2s release4.5s; original PD retained',
        'model_view':str(MODEL_VIEW),'source_before':before})
    cache=outputs/'cache';cache.mkdir()
    env={'PATH':str(PROJECT/'.venv/bin')+':/usr/bin:/bin','HOME':str(outputs),
         'LANG':'C.UTF-8','CUDA_VISIBLE_DEVICES':'','MUJOCO_GL':'disable','PYTHONDONTWRITEBYTECODE':'1',
         'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
         'MALLOC_ARENA_MAX':'2','BFM_MODEL_FOLDER':str(MODEL_VIEW),'BFM_RECOVERY_OUTPUT':str(outputs),
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','WANDB_MODE':'disabled'}
    for name in ['TMPDIR','TEMP','TMP','XDG_CACHE_HOME','XDG_CONFIG_HOME','MPLCONFIGDIR','NUMBA_CACHE_DIR',
                 'TORCH_HOME','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','HF_HOME','PYTHONPYCACHEPREFIX']:
        directory=cache/name;directory.mkdir();env[name]=str(directory)
    affinity=sorted(os.sched_getaffinity(0))[:4]
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
 if p!=output and not p.startswith(output+os.sep):raise PermissionError('Recovery writes limited to owned output: '+p)
def guard(event,args):
 if event=='open':
  value,mode,flags=args
  if (isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):writable(value)
 elif event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.chown','os.utime','os.truncate'):writable(args[0])
 elif event in ('os.rename','os.replace'):writable(args[0]);writable(args[1])
 elif event in ('socket.connect','socket.bind') and isinstance(args[1],tuple):raise PermissionError('No network in CPU recovery')
 elif event in ('os.symlink','os.link','os.system'):raise PermissionError('No link/shell actions in CPU recovery')
sys.addaudithook(guard)
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    (attempt/'bootstrap.py').write_text(bootstrap)
    entry={'number':number,'state':'running','script_sha256':sha((WORKTREE/LAUNCHER).read_bytes()),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'path':str(attempt),'affinity':affinity,'remaining_wall_seconds':remaining,'peak_rss_bytes':0,'peak_vms_bytes':0}
    ledger['attempts'].append(entry);write(ledger_path,ledger)
    started=time.monotonic(); child=None;reason=None
    try:
        with (attempt/'stdout.log').open('wb') as stdout,(attempt/'stderr.log').open('wb') as stderr,(attempt/'telemetry.jsonl').open('w') as telemetry:
            child=subprocess.Popen([str(PROJECT/'.venv/bin/python'),'-I','-B',str(attempt/'bootstrap.py'),str(WORKTREE/LAUNCHER)],
                cwd=attempt/'outputs',env=env,stdout=stdout,stderr=stderr,start_new_session=True,preexec_fn=limits)
            entry.update(pid=child.pid,pgid=child.pid);write(ledger_path,ledger)
            print(json.dumps({'launched':entry}),flush=True)
            while child.poll() is None:
                ps=processes(child.pid);rss=sum(x['rss_bytes'] for x in ps)
                entry['peak_rss_bytes']=max(entry['peak_rss_bytes'],rss)
                entry['peak_vms_bytes']=max(entry['peak_vms_bytes'],sum(x['vms_bytes'] for x in ps))
                telemetry.write(json.dumps({'seconds':time.monotonic()-started,'rss_bytes':rss,'processes':ps})+'\n');telemetry.flush()
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
        after=snapshot();write(run/'source-after.json',after)
        entry['shared_source_unchanged']=before==after
        write(ledger_path,ledger)
        write(attempt/'supervisor-result.json',entry)
        print(json.dumps({'finished':entry}),flush=True)
        for f in ['stdout.log','stderr.log']:
            print(f+'\n'+(attempt/f).read_text(errors='replace')[-14000:],flush=True)

if __name__=='__main__':
    main()
