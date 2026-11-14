"""R2 engineering only: initial equivalence, real sampling/update, complete restart.
Not a scientific training arm. No development/sealed selection.
"""
import argparse,json,sys,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
P=ROOT.parents[2]
LOCK=dict(status='engineering_only',lr=1e-4,batch=32,retention_lambda=1.,sf_coefficient=0.)

def study(e,m,c,out):
    import torch,numpy as np
    from native_env import install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from research.interaction_fb.env import Environment
    from research.interaction_fb.agent import Agent
    from research.interaction_fb.execution_agent import ExecutionAgent
    from research.interaction_fb.journal import read,save,load_state
    from research.interaction_fb.checkpoint import restore
    from diagnostic_fb_resume import tree_digest
    from dual_track_task import global_rng,restore_global_rng
    from f1_mjdata_contract import install_restore_guard
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    install();speed(e);(out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix');install_restore_guard()
    env=Environment(e,m,c);env.reset(0);obs=env.proxy.enrich(env.task.extended_observation())
    path=P/'experiments/interaction-pilot-v3-20261112/runs/pilot-A-v3/checkpoints/round-128'
    manifest=json.loads((path/'manifest.json').read_text());state=read(path/'core.ptz',manifest['files']['core.ptz'])['agent']
    old=Agent(m,obs,e.standing,torch.tensor(c['q_PD_scale'],device='cuda'),'A');old.load(state)
    torch.manual_seed(6301);np.random.seed(6301)
    ustate={k.removeprefix('u.'):v for k,v in state['nets'].items() if k.startswith('u.')}
    a=ExecutionAgent(m,obs,old.goal,old.scale,ARGS.arm,ustate,LOCK);env.proxy.agent=a
    base_digest=tree_digest(m._actor.state_dict())
    with torch.no_grad():
        aa,zz,_=a.action(obs);oa,oz,_=old.action(obs)
        assert torch.equal(aa,oa) and torch.equal(zz,oz),'Initial A128 execution function changed'
        hidden={k:v.clone() for k,v in obs.items()};hidden['privileged_state']+=100
        for lo,hi in [(9,15),(18,22)]:hidden['contact_privileged_v1'][:,lo:hi]+=100
        assert torch.equal(aa,a.action(hidden)[0]),'Privileged leakage'
        rng=global_rng();sample=a.action(obs,sample=True)[0];restore_global_rng(rng)
        expected=(aa+(.01*torch.randn_like(aa)).clamp(-.06,.06)).clamp(-1,1)
        assert torch.equal(sample,expected);restore_global_rng(rng)
    stats=dict(status='engineering_running',arm=ARGS.arm,tail_params=sum(p.numel() for p in a.tail.parameters()),initial_A128_action_z_exact=True,hidden_invariance=True,lock=LOCK,base_sha=manifest['base_sha'])
    write(out/'interface.json',stats)
    if ARGS.resume:
        restore(ARGS.resume,a,env)
        assert tree_digest(a.state())==tree_digest(load_state(ARGS.resume)['agent'])
    else:
        save(out/'checkpoints',0,a,env,RUNTIME,'engineering_initial')
        started=time.monotonic()
        with torch.no_grad():
            for _ in range(50):env.collect_call()
        stats['collect_s']=time.monotonic()-started
        assert env.next_cursor==250
        started=time.monotonic()
        with torch.enable_grad():
            metrics=[a.update(env.rows,env.native,False),a.update(env.rows,env.native,False),a.update(env.rows,env.native,True)]
        torch.cuda.synchronize();stats['complete_updates_s']=time.monotonic()-started;stats['updates']=metrics
        started=time.monotonic();save(out/'checkpoints',1,a,env,RUNTIME,'engineering_after_real_updates');stats['save_s']=time.monotonic()-started
    # Same saved state/RNG -> new real continuation and one complete update.
    before=len(env.physical_stream);started=time.monotonic()
    with torch.no_grad():env.collect_call()
    with torch.enable_grad():metric=a.update(env.rows,env.native,True)
    torch.cuda.synchronize()
    assert tree_digest(m._actor.state_dict())==base_digest,'Shared original actor changed'
    proof=dict(agent=tree_digest(a.state()),rng=tree_digest(global_rng()),physics=tree_digest(env.physical_stream[before:]),
               rows=tree_digest(env.rows[-5:]),metric=metric)
    if ARGS.resume:
        reference=json.loads((ARGS.resume.parent.parent/'continuation-proof.json').read_text())
        assert proof==reference,'New process continuation/update differs'
    write(out/'continuation-proof.json',proof)
    stats.update(status='engineering_complete',continued_update_s=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                 original_actor_unchanged=True,resume_exact=bool(ARGS.resume),real_online_transitions=5 if ARGS.resume else 255)
    write(out/'engineering.json',stats);return stats

def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
    global ARGS,RUNTIME
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=['E','I'],required=True);p.add_argument('--resume',type=Path);ARGS=p.parse_args()
    import subprocess
    RUNTIME=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    from research.interaction_fb import benchmark
    benchmark.study=study
    sys.argv=[__file__,'--output',str(ARGS.output),'--arm','A']
    benchmark.main()
if __name__=='__main__':main()
