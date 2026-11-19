import os,json,sys,gzip,time,hashlib
from pathlib import Path
import numpy as np
s=Path(__file__).resolve().parent
base=s.parents[1]; old=s.parent/'native-rolling-h100-20261107';repo=old/'repo'; current=s.parent/'bfm_ctft_v1/runtime-v1a/research/bfm_ctft_v1'
sys.path[:0]=[str(repo/'research_notes'),str(repo)]
import m11_ipc
import mujoco,torch
from importlib.util import spec_from_file_location,module_from_spec
def mod(n,p):
    sp=spec_from_file_location(n,str(p));m=module_from_spec(sp);sys.modules[n]=m;sp.loader.exec_module(m);return m
rew=mod('ctft_cost50',current/'reward.py')
def load(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def wr(p,obj):p.write_text(json.dumps(obj,indent=2))
def study(e,model,contract,out):
    from low2 import names,path
    m,d=e.sim.model,e.sim.data;coef=json.loads((s.parent/'bfm_ctft_v1/runs/s0-online-probe-01/agent-identity.json').read_text())['original_training_config']['aux_rewards_scaling']
    for n in names:
        p=path(n);x=load(p);pre=load(p.with_name(n+'-source-prefix.json.gz'));i=0;aux=[];ema=rew.ForceEMA();state=np.array(pre['audit']['integration_state'])
        e.reset(seed=0,trajectory=(x['initial']['wrist'],[0.,0.,0.]));e.base.episode_length_buf.zero_()
        mj=mujoco;mj.mj_setState(m,d,state,mj.mjtState.mjSTATE_INTEGRATION);mj.mj_forward(m,d);mj.mj_setState(m,d,state,mj.mjtState.mjSTATE_INTEGRATION)
        device=e.standing.device;prev=torch.tensor(pre['audit']['previous_action'],device=device).reshape(1,-1)
        e.base._pre_physics_step(prev);e.base.last_actions.copy_(e.base.actions)
        def step():
            nonlocal i
            r=x['physics'][i];d.ctrl[-29:]=r['torque'];mj.mj_forward(m,d);d.xfrc_applied[31]=r['applied_contact_wrench'];mj.mj_step(m,d)
            assert np.array_equal(d.qpos,r['qpos']) and np.array_equal(d.qvel,r['qvel']),(n,i,'replay')
            d.xfrc_applied[31]=0;mj.mj_forward(m,d);mj.mj_forward(m,d);i+=1
        e.sim.simulate_at_each_physics_step=step
        for c in x['control']:
            a=torch.tensor(c['action'],device=device,dtype=torch.float32).reshape(1,-1)
            e.base._pre_physics_step(a)
            for _ in range(4):step()
            mj.mj_forward(m,d)
            native={k:float(getattr(e.base,'_reward_'+k)().reshape(-1)[0])*w if w else 0. for k,w in coef.items()}
            e.base.last_actions.copy_(e.base.actions)
            r=x['physics'][i-1];f=r['force'];jac=np.zeros((3,m.nv));mj.mj_jac(m,d,jac,None,d.xpos[31],31)
            v=np.asarray(c['post_velocity']);assert np.array_equal(d.xpos[31],c['post_point']) and np.array_equal(d.qvel[:3],c['post_root_velocity'])
            cost=rew.contact_cost(filtered_force_n=ema.update(f),applied_force_n=f,wrist_velocity_mps=v,root_velocity_mps=d.qvel[:3],guidance_mode=1)
            aux.append(dict(velocity_dot_roundoff=float(np.max(abs(v-jac@d.qvel))),contact=cost['contact_native_reward'],parts=cost['parts'],native=native))
        with gzip.open(out/(n+'.json.gz'),'wt') as stream:json.dump(aux,stream)
        print('AUX MATCH',n,i,flush=True)
    return dict(status='complete')
if __name__=='__main__':
    out=s/os.environ.get('ATTEMPT','replay01');out.mkdir(exist_ok=False)
    os.environ['BFM_RECOVERY_OUTPUT']=str(out);os.chdir(out)
    from m10_physics import main
    import m10_stage,m10_physics
    m10_stage.run_study=study;m10_physics.OUT=out
    t=time.monotonic();raw=mujoco.mj_step;cnt=0
    def counted(*a,**kw):
        global cnt
        cnt+=int(kw.get('nstep',1));assert cnt<30000
        return raw(*a,**kw)
    mujoco.mj_step=counted
    wr(out/'launch.json',dict(pid=os.getpid(),start=time.time(),sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    try:main()
    finally:wr(out/'cost.json',dict(ticks=cnt,seconds=time.monotonic()-t))
