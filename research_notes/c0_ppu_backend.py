"""Explicit hardware-backend reset qualification, not a change to BFM/PD.

Preserve the exact legacy initial integration state and 3s physical standing
prefix. NVIDIA-vs-PPU errors are reported, not called bitwise identical.
Repeated PPU resets and intervention prefixes must still be bitwise identical.
"""
import copy, hashlib, json
from pathlib import Path
REFERENCE=None
SERIAL=0

def install(out):
    import clean_env
    clean_env.reset_common=reset_common
    global OUT
    OUT=Path(out)

def reset_common(env,model,orig):
    import numpy as np,torch,mujoco
    from m8_stage import read,R
    from locomotion_rollout import snapshot
    global REFERENCE,SERIAL
    zero=read(R/'issue20-ccl01-20261028T082201Z/attempt-04/outputs/anchor-reference.json')['physics']
    initial=read(R/'issue21-ccl02a-20261029T153900Z/attempt-01/outputs/dev-2-initial-state.json')
    common=read(R/'issue21-direction-20261030T020000Z/attempt-02/outputs/common-state.json')
    dev=env.standing.device;m=env.sim.model
    model.act=orig;env.qualification_active=False;env.interruption_window=None;env.m1_anchor_hook=None
    env.cfg.body='right_wrist_yaw_link';env.cfg.local_point=[0.,0.,0.];env.cfg.enabled=False
    env.reset(seed=0,trajectory=(zero[0]['wrist'],[0,0,0]))
    for name in ['rew_buf','push_robot_counter','push_robot_plot_counter','command_counter','average_episode_length','torques','last_episode_length_buf']:
        v=getattr(env.base,name);v.copy_(torch.as_tensor(initial['base_tensors'][name],device=dev,dtype=v.dtype))
    env.base.common_step_counter=initial['common_step_counter']
    mujoco.mj_setState(m,env.sim.data,np.asarray(initial['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(m,env.sim.data)
    mujoco.mj_setState(m,env.sim.data,np.asarray(initial['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION)
    for _ in range(30):
        _,_,done,_=env.step(torch.zeros_like(env.standing),baseline=True)
        assert not done,'Standing backend prefix unsafe'
    parity={k:float(abs(np.asarray([r[k] for r in env.trace])-np.asarray([r[k] for r in zero[:600]])).max()) for k in ['qpos','qvel','wrist','root','force']}
    env.base.rew_buf.copy_(torch.as_tensor(common['base_tensors']['rew_buf'],device=dev,dtype=env.base.rew_buf.dtype))
    now=snapshot(env)
    block_diff={}
    def difference(a,b):
        if isinstance(a,dict):
            return max((difference(a[k],b[k]) for k in a),default=0.) if a.keys()==b.keys() else float('inf')
        try:
            x,y=np.asarray(a),np.asarray(b)
            if x.shape!=y.shape:return float('inf')
            if x.dtype.kind in 'biuf' and y.dtype.kind in 'biuf':return float(np.max(abs(x.astype(float)-y.astype(float)))) if x.size else 0.
            return 0. if a==b else float('inf')
        except (TypeError,ValueError):return 0. if a==b else float('inf')
    for k in now:
        value=difference(now[k],common[k]);block_diff[k]=value if np.isfinite(value) else 'structure differs'
    # Safety/numerical check applies only to the 3s unforced backend prefix, not
    # a relaxed causal handoff criterion or permission to change original state.
    assert parity['qpos']<1e-4 and parity['qvel']<1e-3 and parity['wrist']<1e-4 and parity['force']==0.,parity
    equal=REFERENCE is None or now==REFERENCE
    SERIAL+=1
    row=dict(reset=SERIAL,backend='PPU-ZW810E',legacy_prefix_max_abs=parity,
             legacy_snapshot_difference=block_diff,same_backend_exact=equal,
             snapshot_sha256=hashlib.sha256(json.dumps(now,sort_keys=True).encode()).hexdigest(),
             meaning='Cross-platform comparison is descriptive; same-backend equality is mandatory')
    with (OUT/'backend-resets.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    if REFERENCE is None:
        REFERENCE=copy.deepcopy(now)
        (OUT/'ppu-common-state.json').write_text(json.dumps(now))
    assert equal,'Repeated PPU reset is not bitwise identical; stop before interventions'
    return row
