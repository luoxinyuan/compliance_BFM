"""Issue47 future GPU-only experiment; complete live prefixes, no state reset at handoff.

Prepared while GPUs are busy. Compilation is not physical validation. Launch
only after resource recheck and from a clean pinned worktree.
"""
import copy
import gzip
import hashlib
import json
import time
from pathlib import Path
import transfer_student_validated as runner


def study(env, model, contract, out):
    import numpy as np
    import torch
    import mujoco
    from clean_policy import Expert, direct_command
    from geometry_policy import expand
    from transition_env import Task, install_frame
    from locomotion_rollout import spherical
    from m11_runtime_speed import install
    assert env.standing.is_cuda, 'BFM physics experiments require an authorized GPU'
    install_frame()
    import geometry_env, reach_contact
    geometry_env.install_contact = reach_contact.install
    runner.write(out/'runtime-speed.json', install(env))
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    cfg=json.loads(Path(runner.ARGS.resume).read_text());task=Task(env,model,contract,4301)
    nets={};identities={};baselines={};rows=[];queries=[]
    for name in ['I','C0']:
        identity=cfg['models'][name];assert runner.sha(identity['path'])==identity['sha256']
        ck=torch.load(identity['path'],map_location=env.standing.device,weights_only=True)
        net=Expert(1088,'direct').to(env.standing.device);expand(net,ck)
        nets[name]=net.eval().requires_grad_(False)
        identities[name]=dict(**identity,checkpoint_runtime=ck.get('runtime'),std=float(net.log_std.exp().mean()))
    runner.write(out/'identities.json',identities)
    def plain(v):
        if torch.is_tensor(v):return v.detach().cpu().tolist()
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)) or type(v).__name__=='deque':return [plain(x) for x in v]
        return v
    def digest(x):return hashlib.sha256(json.dumps(plain(x),sort_keys=True,allow_nan=False).encode()).hexdigest()
    def snapshot():
        data=env.sim.data;mask=mujoco.mjtState.mjSTATE_INTEGRATION
        state=np.empty(mujoco.mj_stateSize(env.sim.model,mask));mujoco.mj_getState(env.sim.model,data,state,mask)
        # Same Python Task/sim/PD instance continues. These hashes additionally
        # certify repeated prefix parity; they do not implement state restoration.
        sections=dict(integration=state,solver=[data.qacc_warmstart,data.qacc,data.qfrc_constraint,data.ctrl,data.xfrc_applied],
            bfm=env.obs,previous_action=env.previous_action,last_z=task.last_command,last_qpd=task.lastq,
            prevz=task.prevz,raw=task.raw,force_history=list(task.h.rows),contact_history=list(task.contact_history),
            input=task.features(),support=env.last_support,
            memory=[task.alpha,task.phase,task.phase_rate,task.tick,task.prepared],
            tether=dict(time=env.tether.time,force=env.tether.force,anchor=env.tether.anchor,active=env.tether.active,rows=env.tether.geometry_rows),
            physics_prefix=task.physics,control_prefix=task.controls,source_prefix=task.prefix_data)
        return {k:digest(v) for k,v in sections.items()}
    def query(label,t):
        x=task.features();prev=task.last_command.clone();obs={k:v.clone() for k,v in env.obs.items()}
        raw={k:n.distribution(x).mean for k,n in nets.items()}
        z={k:direct_command(v,env.standing) for k,v in raw.items()}
        q={k:torch.stack([task.qof(model.act(obs,spherical(prev,v,(j+1)/5),mean=True))[0] for j in range(5)]) for k,v in z.items()}
        queries.append(dict(prefix=label,time=t,raw_rms_difference=float((raw['I']-raw['C0']).square().mean().sqrt()),
            z_angle_deg=float(torch.rad2deg(torch.acos(((z['I']*z['C0']).sum()/256).clamp(-1,1)))),
            qpd_five_substep_joint_difference_deg=torch.rad2deg(q['I']-q['C0']).cpu().tolist(),
            scope='same current real state and previous z, fixed observed BFM input; not five-step physics'))
    def run(condition,prefix,suffix=None,switch=None):
        if time.monotonic()-runner.START>runner.ARGS.max_seconds-60:raise TimeoutError('stage job bound')
        label=f'{condition}__{prefix}' if suffix is None else f'{condition}__{prefix}_to_{suffix}_{switch:.1f}'
        task.reset(copy.deepcopy(cfg['diagnostic_specs'][condition]),record=True,label=label)
        snaps={};intervention=None;marks=cfg['switch_times'][condition]
        while not task.done:
            t=float(env.tether.time);tick=round((t-4)*10)
            marked=any(tick==round((v-4)*10) for v in marks)
            x=task.features()
            if marked:
                snaps[str(tick)]=snapshot()
                if suffix is None:query(label,t)
            if suffix is not None and tick==round((switch-4)*10):
                original=baselines[(condition,prefix)]['snapshots'][str(tick)]
                intervention=dict(time=t,sections_equal={k:snaps[str(tick)][k]==v for k,v in original.items()})
                assert all(intervention['sections_equal'].values()),intervention
            selected=suffix if suffix is not None and tick>=round((switch-4)*10) else prefix
            task.step(nets[selected].distribution(x).mean)
        row=task.save(out)
        if suffix is not None:assert intervention is not None, 'handoff not reached'
        row.update(prefix=prefix,suffix=suffix,switch=switch,intervention=intervention)
        if suffix is None:
            historical=Path(cfg['historical_physics'][condition][prefix])
            old=json.load(gzip.open(historical,'rt'))
            assert len(old['physics'])==len(task.physics),'historical rollout length differs'
            parity={k:float(np.max(np.abs(np.array([r[k] for r in task.physics])-np.array([r[k] for r in old['physics']])))) for k in ['qpos','qvel','force']}
            row['historical_parity']=dict(path=str(historical),max_abs_difference=parity)
            runner.write(out/(label+'-historical-parity.json'),row['historical_parity'])
            assert max(parity.values())<1e-8,'historical physics differs; diagnose before interventions'
            baselines[(condition,prefix)]=dict(snapshots=snaps,row=row)
        rows.append(row);runner.write(out/'summaries.json',rows)
        runner.write(out/'prefix-parity.json',{str(k):v['snapshots'] for k,v in baselines.items()})
        runner.write(out/'same-state.json',queries)
        print('C0_ENGAGEMENT',label,row['error_cm'],row['force_rms'],row['boundary'],flush=True)
        if row['boundary']:raise RuntimeError('physical boundary; retain failure and stop '+label)
    for condition in cfg['diagnostic_specs']:
        for prefix in ['C0','I']:run(condition,prefix)
        for t in cfg['switch_times'][condition]:
            run(condition,'C0','I',t);run(condition,'I','C0',t)
    return dict(status='completed',episodes=len(rows),rl_training_decisions=0,
        supervised_updates=0,elapsed_s=time.monotonic()-runner.START)


runner.study=study
if __name__=='__main__':runner.main()
