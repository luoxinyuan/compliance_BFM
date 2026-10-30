"""Frozen learned alpha through unchanged BFM/PD; fixed paired fit and physical evaluation."""
import copy,hashlib,json,os,time
from pathlib import Path
from locomotion_rollout import UnsafePhysics,snapshot,spherical
from contact_gate import ContactGate,smooth


def clock_progress(t):
    if t < 3: return 0.
    if t < 7: return smooth((t-3)/4)
    if t < 8: return 1.
    return smooth((8.75-t)/.75)


def run_study(env,model,contract,out):
    import numpy as np
    import torch
    import mujoco
    from wholebody_physics import goal_observation
    root=Path(__file__).resolve().parents[1]
    runs=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs')
    read=lambda p:json.loads(Path(p).read_text())
    def save(name,x): (out/name).write_text(json.dumps(x,indent=2,allow_nan=False))
    alpha_protocol=read(root/'research_notes/alpha_student_protocol.json')
    assert hashlib.sha256((root/'research_notes/contact_gate.py').read_bytes()).hexdigest()==alpha_protocol['teacher_gate_sha256']
    save('alpha-protocol.json',alpha_protocol)
    protocol=read(root/'research_notes/contact_gate_protocol.json');save('protocol.json',protocol)
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
    goal=saved
    assert torch.isfinite(goal).all() and abs(float(goal.norm())-16)<1e-4
    save('native-command.json',dict(source=str(source),source_sha256=protocol['source_sha256'],
         z=goal.tolist(),standing_z=env.standing.tolist(),recomputed_z_max_abs=err,norm=float(goal.norm()),
         angle_from_standing_deg=target['per_mode']['native']['angle_from_standing_deg'],direct_joint_residual=False))
    if os.environ['ALPHA_STAGE']=='train':
        from alpha_fit import fit_pair
        return fit_pair(env,model,goal,contract,out,alpha_protocol)
    from alpha_memory import ForceMemory
    from alpha_student import load_student,student_input
    training=Path(os.environ['ALPHA_TRAIN_OUTPUT'])
    fit=read(training/'paired-fit-result.json');students={}
    for name in ['A','B']:
        assert fit['full_'+name]['static_command_screen_passed']
        checkpoint=training/alpha_protocol['final_checkpoints'][name]
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==fit['full_'+name]['checkpoint_sha256']
        students[name],meta=load_student(checkpoint,dev)
        assert meta['updates']==2000 and meta['protocol']==alpha_protocol
    save('student-identities.json',{k:fit['full_'+k]['checkpoint_sha256'] for k in students})
    source=runs/alpha_protocol['teacher_source']
    zero=read(runs/'issue20-ccl01-20261028T082201Z/attempt-04/outputs/anchor-reference.json')['physics']
    initial0=read(runs/'issue21-ccl02a-20261029T153900Z/attempt-01/outputs/dev-2-initial-state.json')
    common=read(runs/'issue21-direction-20261030T020000Z/attempt-02/outputs/common-state.json')
    geometry=read(oldroot/'geometry.json')
    old=read(runs/'issue21-skill-source-20261030A/physics-attempt-01/outputs/standing_contact.json')
    scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev)
    qtarget=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
    original_act=model.act;record=False;controller='standing';control=[];last_q=None;gate=None;memory=None
    def act(obs,z,mean=True):
        nonlocal last_q
        t=float(env.tether.time)
        force=env.tether.force.copy() # Last completed physics sample, never future Tether.measure().
        gate_record=gate.step(force.tolist()) if record and controller in ('contact','A','B') else None
        alpha=gate_record['alpha'] if gate_record else (clock_progress(t) if record and controller=='clock' else 0.)
        if record and controller in students:
            history=memory.append(force.tolist())
            x=student_input(model,obs,[history],alpha_protocol['memory']['keys'])
            alpha=float(students[controller](x).item())
            assert np.isfinite(alpha) and 0<=alpha<=1
        command=spherical(env.standing,goal,alpha)
        action=original_act(obs,command,mean=True)
        if record:
            assert torch.isfinite(action).all()
            nominal=original_act(obs,env.standing,mean=True);q=qtarget(action);q0=qtarget(nominal)
            control.append(dict(time=t,relative_time=t-3,alpha=alpha,gate=gate_record,measured_force_xyz=force.tolist(),
                executed_z=command.tolist(),z_angle_deg=float(torch.rad2deg(torch.acos(((command*env.standing).sum()/256.).clamp(-1,1)))),
                action=action[0].tolist(),standing_action=nominal[0].tolist(),q_PD=q[0].tolist(),standing_q_PD=q0[0].tolist(),delta_q_PD=(q-q0)[0].tolist(),
                bfm_obs={k:v.tolist() for k,v in obs.items()},pre_qpos=env.sim.data.qpos.tolist(),pre_qvel=env.sim.data.qvel.tolist(),
                action_saturation_fraction=float(((action*5).abs()>=contract['clip']).float().mean()),q_PD_step_max_rad=float(abs(q-last_q).max())))
            if controller in students:
                shadow=original_act(obs,spherical(env.standing,goal,gate_record['alpha']),mean=True)
                control[-1].update(shadow_alpha=gate_record['alpha'],shadow_action=shadow[0].tolist(),shadow_q_PD=qtarget(shadow)[0].tolist())
            last_q=q.clone()
        return action
    model.act=act
    results=dict(status='running',episodes=[],first_failure=None,student_frozen=True)
    anchor_reference=read(source/'anchor_reference__standing.json');anchors={}
    conditions=protocol['conditions']+[alpha_protocol['physical']['new_final_test']]
    for onset in sorted({c['onset'] for c in conditions}):
        row=anchor_reference['initial'] if onset==3 else anchor_reference['physics'][round((onset-3)*200)-1]
        assert abs(row['time']-onset)<1e-8
        anchors[str(onset)]=row['wrist']
    save('fixed-anchors.json',dict(anchors=anchors,delta_x_m=geometry['delta_x_m'],source=str(source/'anchor_reference__standing.json')))
    standing_episodes={c['name']:read(source/(c['name']+'__standing.json')) for c in protocol['conditions']}
    jobs=[(c,ctl) for c in protocol['conditions'] for ctl in ['A','B']]
    jobs += [(conditions[-1],ctl) for ctl in ['standing','clock','contact','A','B']]
    assert len(jobs)==alpha_protocol['resources']['max_full_branches']
    try:
        for cond,controller in jobs:
            label=cond['name']+'__'+controller
            record=False;env.qualification_active=False
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
            state=snapshot(env);assert state==common,'Full common standing state/history mismatch'
            save(label+'-initial-state.json',state)
            initial=copy.deepcopy(env.trace[-1]);start=len(env.trace)
            control=[];last_q=qtarget(env.previous_action).clone();gate=ContactGate(protocol['gate']);memory=ForceMemory(320);boundary=None
            if cond['name']!='anchor_reference':
                origin=np.asarray(anchors[str(cond['onset'])])
                for cfg in (env.cfg,env.tether.cfg):
                    cfg.activation_time=cond['onset'];cfg.ramp_duration=4.;cfg.release_time=cond['onset']+cond['duration'];cfg.enabled=cond['enabled']
                env.tether.origin=origin.copy();env.tether.anchor=origin.copy();env.tether.delta=np.array([geometry['delta_x_m'],0,0])
            record=True;env.qualification_active=True
            steps=round((cond['end']-3)*50)
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
            ep=dict(label=label,condition=cond,controller=controller,initial=initial,physics=copy.deepcopy(env.trace[start:]),control=control,boundary=boundary,
                    common_state_sha256=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest())
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
                pre=[i for i,r in enumerate(ep['physics']) if r['force_time']<cond['onset']-1e-8]
                parity={k:max((float(abs(np.asarray(ep['physics'][i][k])-np.asarray(ref['physics'][i][k])).max()) for i in pre),default=0.) for k in ['qpos','qvel','force','wrist']}
                save(label+'-precontact-difference.json',parity)
                if controller=='contact':assert max(parity.values())==0
            save('physical-result.json',results);print('CONTACT_GATE_BRANCH',label,summary,flush=True)
        else:results['status']='completed'
        save('physical-result.json',results);return results
    finally:
        model.act=original_act;env.qualification_active=False
