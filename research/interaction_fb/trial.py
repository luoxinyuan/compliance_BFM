"""Isolated child entry. All monkeypatches scoped to this process and recorded."""
import os,sys,time,json,argparse,subprocess,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc

def study(e,m,c,out):
    import torch,numpy as np
    from native_env import install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from research.interaction_fb.env import Environment
    from research.interaction_fb.agent import Agent
    from research.interaction_fb.schema import deploy
    from research.interaction_fb.checkpoint import save,restore
    from diagnostic_fb_resume import tree_digest
    from dual_track_task import global_rng
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    install();speed(e);(out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix')
    env=Environment(e,m,c);env.reset(0)
    env.proxy.start=env.task.last_command.clone();env.proxy.target=env.proxy.start.clone()
    obs=env.proxy.enrich(env.task.extended_observation())
    torch.manual_seed(6201);np.random.seed(6201)
    agent=Agent(m,obs,e.standing,torch.tensor(c['q_PD_scale'],device='cuda'));env.proxy.agent=agent
    # Permission test: all hidden truth can change without changing deploy input/action.
    hidden={k:v.clone() for k,v in obs.items()};hidden['privileged_state']+=100
    for lo,hi in [(9,15),(18,22)]:hidden['contact_privileged_v1'][:,lo:hi]+=100
    with torch.no_grad():
        assert torch.equal(deploy(obs),deploy(hidden))
        assert torch.equal(agent.action(obs)[0],agent.action(hidden)[0])
    write(out/'interface.json',dict(hidden_input_invariance_exact=True,params={k:sum(p.numel() for p in v.parameters()) for k,v in agent.nets.items()},
        frozen_base=True,optimizer_names=list(agent.opts),robot_keys={k:list(v.shape) for k,v in e.obs.items()},
        global_hooks=['native_env.install:geometry_env.install_contact','dual_track_task.Task:clean_env.advance','rolling_backend.install:m8_stage.read'],
        isolated_one_Task_process=True))
    round0=0
    if ARGS.resume:round0=restore(ARGS.resume,agent,env)
    else:save(out/'checkpoints',0,agent,env,ARGS.runtime,'initial')
    records=[]
    try:
        for r in range(round0+1,4):
            begun=time.monotonic();sample=time.monotonic()
            with torch.no_grad():
                for i in range(50):
                    env.collect_call()
                    if ARGS.inject and r==2 and i==2:raise InterruptedError('planned mid-round injection')
            collect_s=time.monotonic()-sample;update=time.monotonic()
            with torch.enable_grad():metrics=[agent.update(env.rows,env.native) for _ in range(2)]
            torch.cuda.synchronize();update_s=time.monotonic()-update;io=time.monotonic()
            manifest=save(out/'checkpoints',r,agent,env,ARGS.runtime)
            io_s=time.monotonic()-io
            rec=dict(round=r,transitions=env.next_cursor,updates=agent.updates,collect_s=collect_s,update_s=update_s,io_s=io_s,
                total_s=time.monotonic()-begun,peak_cuda_bytes=torch.cuda.max_memory_allocated(),metrics=metrics,
                contact_rows=sum(bool(x['observation']['contact_privileged_v1'][0,21]) for x in env.rows),
                fingerprints=dict(agent=tree_digest(agent.state()),replay=tree_digest(env.rows),rng=tree_digest(global_rng()),
                    qpos=tree_digest(e.sim.data.qpos),qvel=tree_digest(e.sim.data.qvel),history=tree_digest(e.base.history_handler.history)))
            records.append(rec);write(out/'rounds.json',records);print('COMMITTED',r,rec['total_s'],flush=True)
    except Exception as exc:
        write(out/'failure-evidence.json',dict(type=type(exc).__name__,error=str(exc),cursor=env.next_cursor,updates=agent.updates,
            rounds=records,physics=env.task.physics[-20:],last_controls=env.task.controls[-5:]))
        # Preserve uncommitted work separately. Never call it a completed round.
        torch.save(dict(agent=agent.state(),rows=env.rows,commands=env.commands,actions=env.proxy.actions,
                        task_spec=env.task.spec,physics=env.task.physics,controls=env.task.controls),out/'uncommitted-stop.pt')
        if isinstance(exc,InterruptedError):return dict(status='injected_stop',rounds=records)
        raise
    return dict(status='trial_completed',rounds=records)

def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
    global ARGS
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path);p.add_argument('--inject',action='store_true');ARGS=p.parse_args()
    ARGS.runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
    ARGS.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(BFM_RECOVERY_OUTPUT=str(ARGS.output),P4000_CANONICAL_ROW_ACTOR='1',CLEAN_KIND='direct',BFM_RECORD_NOMINAL_DIAGNOSTIC='0')
    import mujoco
    original=mujoco.mj_step;counts={'actual':0,'reserved':0}
    def counted(*a,**kw):
        n=int(kw.get('nstep',1));counts['reserved']+=n;write(ARGS.output/'physics-counter.json',counts)
        result=original(*a,**kw);counts['actual']+=n;return result
    mujoco.mj_step=counted
    os.chdir(ARGS.output)
    import m10_stage,m10_physics
    m10_physics.OUT=ARGS.output;m10_stage.run_study=study
    try:m10_physics.main()
    finally:write(ARGS.output/'physics-counter.json',counts)
if __name__=='__main__':main()
