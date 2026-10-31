"""Frozen skill qualification only. No fitting/action residual. Clock arms are explicitly privileged qualification baselines."""
import copy,hashlib,json,os,time
from pathlib import Path
from locomotion_rollout import UnsafePhysics,snapshot,spherical
from contact_gate import smooth

def fixed_progress(t):
    return smooth((t-3.8)/4.) if t<8.8 else 1.-smooth((t-8.8)/.75)

def run_study(env,model,contract,out):
    import numpy as np
    import torch,mujoco
    from wholebody_physics import goal_observation
    from alpha_student import load_student,student_input
    from alpha_memory import ForceMemory
    from frozen_bfm_placement import tensor_digest
    root=Path(__file__).resolve().parents[1]
    runs=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs')
    read=lambda p:json.loads(Path(p).read_text())
    def save(name,x): (out/name).write_text(json.dumps(x,indent=2,allow_nan=False))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    study=read(root/'research_notes/beyond_alpha_protocol.json');save('protocol.json',study)
    dev=env.standing.device;m=env.sim.model
    assert dev.type=='cuda' and env.dt==.02 and env.interval==5 and env.history.shape==(3,3)
    assert abs(m.opt.timestep-.005)<1e-10
    assert contract['model_sha256']==study['model_sha256']
    target_path=runs/'issue21-goalcalibration-20261030A/attempt-01/outputs/same-state-forward.json'
    original_protocol=read(root/'research_notes/contact_gate_protocol.json')
    assert sha(target_path)==original_protocol['source_sha256']
    target=read(target_path)['states'][-1];assert target['index']==40
    goal=model.goal_inference(goal_observation(env,np.asarray(target['target_pose']),np.zeros(m.nv)))
    saved=torch.tensor(target['per_mode']['native']['z'],device=dev,dtype=torch.float32)
    err=float(abs(goal-saved).max());assert err<2e-6;goal=saved
    assert abs(float(goal.norm())-16)<1e-4
    nativepath=runs/study['A2']['path'];raisedpath=runs/study['raised']['path']
    assert sha(nativepath)==study['A2']['sha256'] and sha(raisedpath)==study['raised']['sha256']
    native,_=load_student(nativepath,dev);digests=[tensor_digest(native)]
    raised=torch.tensor(read(raisedpath)['z'],device=dev,dtype=torch.float32)
    assert raised.shape==env.standing.shape and abs(float(raised.norm())-16)<1e-4
    assert len(native.center)==1425
    save('interface-audit.json',dict(model_sha256=contract['model_sha256'],native_sha256=sha(nativepath),raised_sha256=sha(raisedpath),
        native_input_dim=1425,native_hz=50,bfm_hz=50,physics_hz=200,
        native_force_history='320 last completed50Hz samples /5N',goal_recompute_max_abs=err,
        goal_angle_deg=target['per_mode']['native']['angle_from_standing_deg'],
        raised_angle_deg=float(torch.rad2deg(torch.acos(((raised*env.standing).sum()/256).clamp(-1,1)))),
        no_direct_joint_correction=True,no_training=True,source='Pinned existing same-weight reward inference; no new latent inference'))
    zero=read(runs/'issue20-ccl01-20261028T082201Z/attempt-04/outputs/anchor-reference.json')['physics']
    initial0=read(runs/'issue21-ccl02a-20261029T153900Z/attempt-01/outputs/dev-2-initial-state.json')
    common=read(runs/'issue21-direction-20261030T020000Z/attempt-02/outputs/common-state.json')
    scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev)
    qtarget=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
    original_act=model.act
    record=False;controller='standing';control=[];decisions=[];last_q=None;last_z=None;memory=None;tick=0
    def act(obs,z,mean=True):
        nonlocal last_q,last_z,tick
        t=float(env.tether.time);force=env.tether.force.copy()
        alpha=0.;predicted_alpha=None
        if record:
            if controller=='A2':
                hist=memory.append(force.tolist());x=student_input(model,obs,[hist],env.keys)
                alpha=float(native(x).item());predicted_alpha=alpha;command=spherical(env.standing,goal,alpha)
            elif controller in ['forward_clock','raise_clock']:
                alpha=fixed_progress(t);command=spherical(env.standing,goal if controller=='forward_clock' else raised,alpha)
            else:command=env.standing
        else:command=env.standing
        assert torch.isfinite(command).all() and abs(float(command.norm())-16)<1e-4
        action=original_act(obs,command,mean=True)
        if record:
            assert torch.isfinite(action).all()
            nominal=original_act(obs,env.standing,mean=True);q=qtarget(action);q0=qtarget(nominal)
            control.append(dict(time=t,alpha=alpha,gate=None,predicted_alpha=predicted_alpha,
                measured_force_xyz=force.tolist(),executed_z=command.tolist(),
                z_angle_deg=float(torch.rad2deg(torch.acos(((command*env.standing).sum()/256).clamp(-1,1)))),
                z_step_deg=float(torch.rad2deg(torch.acos(((command*last_z).sum()/256).clamp(-1,1)))),
                action=action[0].tolist(),standing_action=nominal[0].tolist(),q_PD=q[0].tolist(),standing_q_PD=q0[0].tolist(),delta_q_PD=(q-q0)[0].tolist(),
                bfm_obs={k:v.tolist() for k,v in obs.items()},pre_qpos=env.sim.data.qpos.tolist(),pre_qvel=env.sim.data.qvel.tolist(),
                action_saturation_fraction=float(((action*5).abs()>=contract['clip']).float().mean()),q_PD_step_max_rad=float(abs(q-last_q).max())))
            last_q=q.clone();last_z=command.clone();tick+=1
        return action
    model.act=act
    reference_path=runs/'issue21-contact-gate-20261030A/physics-attempt-01/outputs/anchor_reference__standing.json'
    anchor_ref=read(reference_path);row=anchor_ref['physics'][round((3.8-3)*200)-1];assert abs(row['time']-3.8)<1e-8
    anchor=np.asarray(row['wrist'])
    save('fixed-anchors.json',dict(anchor=anchor.tolist(),source=str(reference_path),sha256=sha(reference_path)))
    jobs=[(c,ctl) for c in study['conditions'] for ctl in c['controllers']]
    results=dict(status='running',episodes=[],first_failure=None,student_frozen=True,optimizer_updates=0)
    standing_episodes={}
    try:
        for cond,controller in jobs:
            label=cond['name']+'__'+controller
            record=False;env.qualification_active=False;env.interruption_window=None
            env.cfg.enabled=False;env.cfg.activation_time=3.;env.cfg.ramp_duration=4.;env.cfg.release_time=8.
            env.reset(seed=0,trajectory=(zero[0]['wrist'],[0,0,0]))
            for name in ['rew_buf','push_robot_counter','push_robot_plot_counter','command_counter','average_episode_length','torques','last_episode_length_buf']:
                v=getattr(env.base,name);v.copy_(torch.as_tensor(initial0['base_tensors'][name],device=v.device,dtype=v.dtype))
            env.base.common_step_counter=initial0['common_step_counter']
            mujoco.mj_setState(m,env.sim.data,np.asarray(initial0['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(m,env.sim.data)
            mujoco.mj_setState(m,env.sim.data,np.asarray(initial0['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION)
            for _ in range(30):
                assert time.monotonic()<float(os.environ['BFM_STAGE_DEADLINE_MONOTONIC'])
                _,_,done,_=env.step(torch.zeros_like(env.standing),baseline=True);assert not done
            parity={k:float(abs(np.asarray([r[k] for r in env.trace])-np.asarray([r[k] for r in zero[:600]])).max()) for k in ['qpos','qvel','wrist','root','force']}
            save(label+'-prefix-parity.json',parity);assert max(parity.values())==0
            env.base.rew_buf.copy_(torch.as_tensor(common['base_tensors']['rew_buf'],device=dev,dtype=env.base.rew_buf.dtype))
            state=snapshot(env);assert state==common,'Full common standing state/history mismatch';save(label+'-initial-state.json',state)
            initial=copy.deepcopy(env.trace[-1]);start=len(env.trace)
            control=[];decisions=[];tick=0;last_q=qtarget(env.previous_action).clone();last_z=env.standing.clone();memory=ForceMemory(320);boundary=None
            for cfg in (env.cfg,env.tether.cfg):
                cfg.activation_time=cond['onset'];cfg.ramp_duration=4.;cfg.release_time=cond['onset']+cond['duration'];cfg.enabled=cond['enabled']
            env.tether.origin=anchor.copy();env.tether.anchor=anchor.copy();env.tether.delta=np.array(cond['delta_xyz_m'])
            record=True;env.qualification_active=True;steps=round((cond['end']-3)*50)
            for high in range(steps):
                assert time.monotonic()<float(os.environ['BFM_STAGE_DEADLINE_MONOTONIC'])
                try:
                    action=model.act(env.obs,env.standing,mean=True)
                    env.obs,_,terminated,truncated,_=env.wrapped.step(action,to_numpy=False)
                    env.steps+=1;env.previous_action=action.clone()
                    env.history=np.roll(env.history,-1,axis=0);env.history[-1]=env.tether.force
                    done=bool(torch.as_tensor(terminated).any() or torch.as_tensor(truncated).any())
                    if env.failed or bool(torch.as_tensor(terminated).any()) or (done and high!=steps-1):
                        boundary=dict(kind='native_failure_or_early_termination',time=float(env.tether.time));break
                except UnsafePhysics as exc:boundary=dict(kind='pre_registered_physical_boundary',physics=exc.row);break
                except FloatingPointError as exc:boundary=dict(kind='nonfinite',message=str(exc));break
            record=False;env.qualification_active=False
            ep=dict(label=label,condition=cond,controller=controller,initial=initial,physics=copy.deepcopy(env.trace[start:]),control=control,decisions=decisions,boundary=boundary,common_state_sha256=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest())
            save(label+'.json',ep)
            summary=dict(label=label,simulated_seconds=len(ep['physics'])/200,complete=len(ep['physics'])==steps*4 and boundary is None,boundary=boundary)
            results['episodes'].append(summary)
            if not summary['complete']:
                results.update(status='physical-stopped',first_failure=summary);save('physical-result.json',results);break
            if controller=='standing':standing_episodes[cond['name']]=ep
            else:
                ref=standing_episodes[cond['name']]
                assert [r['anchor'] for r in ep['physics']]==[r['anchor'] for r in ref['physics']]
                assert [r['active'] for r in ep['physics']]==[r['active'] for r in ref['physics']]
            save('physical-result.json',results);print('SKILL_BRANCH',label,summary,flush=True)
        else:results['status']='completed'
        assert [tensor_digest(native)]==digests
        assert all(not p.requires_grad and p.grad is None for net in [native] for p in net.parameters())
        assert sha(raisedpath)==study['raised']['sha256'] and sha(nativepath)==study['A2']['sha256']
        save('physical-result.json',results);return results
    finally:model.act=original_act;env.qualification_active=False
