"""Qualify one existing native goal endpoint, then paired physical contact.

The goal and time schedule are privileged teacher inputs. They are not proposed
as Student observations. No IK solve, joint residual, PD override or learning.
"""
import copy,hashlib,json,math,os,time
from pathlib import Path
from locomotion_rollout import UnsafePhysics,snapshot,spherical,coefficient


def smooth(u):
    u=min(1.,max(0.,u));return u*u*(3.-2.*u)


def progress(label,t):
    if label=='goal_free':return coefficient(t)
    if label=='goal_contact':
        return smooth(t/4.) if t<5 else smooth((5.75-t)/.75)
    return 0.


def basic_qualification(ep):
    import numpy as np
    rows=ep['physics'];cs=ep['control'];initial=ep['initial']
    hold=[r for r in rows if .75-1e-8<=r['force_time']-3<3.25-1e-8]
    c=[r for r in cs if .75-1e-8<=r['relative_time']<3.25-1e-8]
    T0=np.asarray(initial['torso_matrix']);Ts=np.asarray([r['torso_matrix'] for r in hold])
    evidence=dict(wrist_forward_mean_cm=float(np.mean([r['wrist'][0]-initial['wrist'][0] for r in hold])*100),
        shoulder_forward_mean_cm=float(np.mean([r['shoulder_xyz'][0]-initial['shoulder_xyz'][0] for r in hold])*100),
        torso_rotation_max_deg=float(np.degrees(np.arccos(np.clip((np.einsum('nij,ij->n',Ts,T0)-1)/2,-1,1))).max()),
        waist_qPD_delta_rms_deg=float(np.degrees(np.sqrt(np.mean(np.square([r['delta_q_PD'][12:15] for r in c]))))),
        final_root_speed_mps=float(np.linalg.norm(rows[-1]['qvel'][:3])),
        last_second_qPD_step_max_deg=float(np.degrees(max(r['q_PD_step_max_rad'] for r in cs if r['relative_time']>=5-1e-8))))
    checks=dict(full_episode=len(rows)==1200 and ep['boundary'] is None,
        wrist_forward=evidence['wrist_forward_mean_cm']>=5,
        shoulder_forward=evidence['shoulder_forward_mean_cm']>=3,
        actual_body_response=evidence['torso_rotation_max_deg']>=5,
        body_command=evidence['waist_qPD_delta_rms_deg']>=1,
        recovery_speed=evidence['final_root_speed_mps']<=.02,
        recovery_continuity=evidence['last_second_qPD_step_max_deg']<=2)
    return dict(evidence=evidence,checks=checks,passed=all(checks.values()),scope='One native no-force skill instance, not contact task success or general robustness')


def run_study(env,model,contract,out):
    import numpy as np
    import torch
    import mujoco
    from wholebody_physics import goal_observation
    root=Path(__file__).resolve().parents[1]
    runs=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs')
    read=lambda p:json.loads(Path(p).read_text())
    def save(name,x):(out/name).write_text(json.dumps(x,indent=2,allow_nan=False))
    protocol=read(root/'research_notes/skill_source_protocol.json');save('protocol.json',protocol)
    source=runs/'issue21-goalcalibration-20261030A/attempt-01/outputs/same-state-forward.json'
    oldroot=runs/'issue21-wholebody-20261030A/attempt-01/outputs'
    for p,key in [(source,'source_sha256'),(oldroot/'baseline.json','baseline_sha256')]:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==protocol[key]
    target=read(source)['states'][-1];assert target['index']==40
    m=env.sim.model;dev=env.standing.device
    assert dev.type=='cuda' and env.dt==.02 and abs(m.opt.timestep-.005)<1e-10
    goal=model.goal_inference(goal_observation(env,np.asarray(target['target_pose']),np.zeros(m.nv)))
    saved=torch.tensor(target['per_mode']['native']['z'],device=dev,dtype=torch.float32)
    err=float((goal-saved).abs().max());assert err<2e-6
    goal=saved # Preserve the verified command after independently checking identity.
    assert torch.isfinite(goal).all() and abs(float(goal.norm())-16)<1e-4
    save('native-command.json',dict(source=str(source),source_sha256=protocol['source_sha256'],target_pose=target['target_pose'],z=goal.tolist(),recomputed_z_max_abs=err,norm=float(goal.norm()),angle_from_standing_deg=target['per_mode']['native']['angle_from_standing_deg'],direct_joint_residual=False))
    zero=read(runs/'issue20-ccl01-20261028T082201Z/attempt-04/outputs/anchor-reference.json')['physics']
    initial0=read(runs/'issue21-ccl02a-20261029T153900Z/attempt-01/outputs/dev-2-initial-state.json')
    common=read(runs/'issue21-direction-20261030T020000Z/attempt-02/outputs/common-state.json')
    geometry=read(oldroot/'geometry.json');oldbase=read(oldroot/'baseline.json')
    scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev)
    qtarget=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
    original_act=model.act;record=False;label='standing';control=[];last_q=None
    def act(obs,z,mean=True):
        nonlocal last_q
        t=float(env.tether.time)-3
        alpha=progress(label,t) if record else 0.
        command=spherical(env.standing,goal,alpha)
        action=original_act(obs,command,mean=True)
        if record:
            nominal=original_act(obs,env.standing,mean=True);q=qtarget(action);q0=qtarget(nominal)
            control.append(dict(time=float(env.tether.time),relative_time=t,alpha=alpha,executed_z=command.tolist(),
                z_angle_deg=float(torch.rad2deg(torch.acos(((command*env.standing).sum()/256.).clamp(-1,1)))),
                action=action[0].tolist(),standing_action=nominal[0].tolist(),q_PD=q[0].tolist(),standing_q_PD=q0[0].tolist(),delta_q_PD=(q-q0)[0].tolist(),
                bfm_obs={k:v.tolist() for k,v in obs.items()},pre_qpos=env.sim.data.qpos.tolist(),pre_qvel=env.sim.data.qvel.tolist(),
                action_saturation_fraction=float(((action*5).abs()>=contract['clip']).float().mean()),q_PD_step_max_rad=0. if last_q is None else float(abs(q-last_q).max())))
            last_q=q.clone()
        return action
    model.act=act;results=dict(status='running',episodes=[],first_failure=None)
    try:
        for label in protocol['branches']:
            record=False;env.qualification_active=False;env.cfg.enabled=False
            env.cfg.activation_time=3.;env.cfg.ramp_duration=4.;env.cfg.release_time=8.
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
            diffs={k:float(abs(np.asarray([r[k] for r in env.trace])-np.asarray([r[k] for r in zero[:600]])).max()) for k in ['qpos','qvel','wrist','root','force']}
            save(label+'-prefix-parity.json',diffs);assert max(diffs.values())==0
            env.base.rew_buf.copy_(torch.as_tensor(common['base_tensors']['rew_buf'],device=dev,dtype=env.base.rew_buf.dtype))
            state=snapshot(env);assert state==common,'Full common standing state/history mismatch'
            save(label+'-initial-state.json',state)
            if label=='goal_free':
                checks=[];last=qtarget(env.previous_action)
                for t in np.arange(0,6,.02):
                    command=spherical(env.standing,goal,progress(label,float(t)))
                    a=original_act(env.obs,command,mean=True);q=qtarget(a)
                    assert torch.isfinite(a).all()
                    checks.append(dict(time=float(t),step_deg=float(torch.rad2deg(abs(q-last).max())),action_clip_ratio=float(a.abs().max())));last=q
                save('fixed-state-command-screen.json',dict(rows=checks,screen_max_step_deg=10,not_dynamic_safety_proof=True))
                if max(c['step_deg'] for c in checks)>10 or max(c['action_clip_ratio'] for c in checks)>=1:
                    results.update(status='command-screen-stopped',first_failure='Fixed candidate command precheck exceeded existing 10 degree step or raw-action clip screen');break
            start=len(env.trace);initial=copy.deepcopy(env.trace[-1]);control=[];last_q=qtarget(env.previous_action).clone();boundary=None
            if label!='goal_free':
                env.tether.origin=env.last_point.copy();env.tether.anchor=env.last_point.copy()
                env.tether.delta=np.array([geometry['delta_x_m'],0,0]);env.cfg.enabled=True
            record=True;env.qualification_active=True
            for high in range(300):
                assert time.monotonic()<float(os.environ['BFM_STAGE_DEADLINE_MONOTONIC'])
                try:
                    action=model.act(env.obs,env.standing,mean=True)
                    env.obs,_,terminated,truncated,_=env.wrapped.step(action,to_numpy=False)
                    env.steps+=1;env.previous_action=action.clone()
                    env.history=np.roll(env.history,-1,axis=0);env.history[-1]=env.tether.force
                    done=bool(torch.as_tensor(terminated).any() or torch.as_tensor(truncated).any())
                    if env.failed or bool(torch.as_tensor(terminated).any()) or (done and high!=299):
                        boundary=dict(kind='native_failure_or_early_termination',relative_time=float(env.tether.time)-3);break
                except UnsafePhysics as exc:boundary=dict(kind='pre_registered_physical_boundary',physics=exc.row);break
                except FloatingPointError as exc:boundary=dict(kind='nonfinite',message=str(exc));break
            record=False;env.qualification_active=False
            ep=dict(label=label,initial=initial,physics=copy.deepcopy(env.trace[start:]),control=control,boundary=boundary,common_state_sha256=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest())
            save(label+'.json',ep)
            summary=dict(label=label,simulated_seconds=len(ep['physics'])/200,complete=len(ep['physics'])==1200 and boundary is None,boundary=boundary)
            results['episodes'].append(summary)
            if not summary['complete']:
                results.update(status='physical-stopped',first_failure=summary);break
            if label=='standing_contact':
                parity={k:float(abs(np.asarray([r[k] for r in ep['physics']])-np.asarray([r[k] for r in oldbase['physics']])).max()) for k in ['qpos','qvel','root','wrist','force']}
                parity['action']=float(abs(np.asarray([c['action'] for c in control])-np.asarray([c['action'] for c in oldbase['control']])).max())
                save('old-contact-baseline-parity.json',parity);assert max(parity.values())==0
            if label!='goal_free':assert [r['anchor'] for r in ep['physics']]==[r['anchor'] for r in oldbase['physics']]
            if label=='goal_free':
                q=basic_qualification(ep);save('basic-skill-qualification.json',q)
                if not q['passed']:
                    results.update(status='capability-stopped',first_failure=q);break
            save('physical-result.json',results);print('SKILL_SOURCE_BRANCH',label,summary,flush=True)
        else:results['status']='completed'
        save('physical-result.json',results);return results
    finally:
        model.act=original_act;env.qualification_active=False
