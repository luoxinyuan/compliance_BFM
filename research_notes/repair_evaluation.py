"""Fixed post-fit evaluations; final checkpoint only; no optimization here."""
import copy,hashlib,json,math,time,os
from pathlib import Path
import numpy as np
import torch
import mujoco
from contact_projector import ContactGate,deploy_input,constrained,save,check_deadline
from ccl02a_teacher import arm_ik,ReleaseFade,audit_episode,summarize
from targeted_repair import RUNS,SHARED,SINGLE,PROBE,TEACHER,read,sha
ORIGINAL_TEACHER=RUNS/'issue21-ccl02a-20261029T153900Z/attempt-01/outputs'
ZERO=RUNS/'issue20-ccl01-20261028T082201Z/attempt-04/outputs/anchor-reference.json'

def run_evaluations(env,model,net,contract,out,protocol):
    dev=env.standing.device;original_act=model.act;cfg=env.tether.cfg
    arm=contract['arm_indices'];nonarm=list(range(22));names=contract['dof_names'];m=env.sim.model
    ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names];vadr=m.jnt_dofadr[ids];limits=m.jnt_range[ids]
    ctl=env.base.config.robot.control;effort=torch.tensor(list(env.base.config.robot.dof_effort_limit_list),device=dev)
    scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=env.base.default_dof_pos+env.base.default_dof_pos_offset;clip=contract['clip']
    def qtarget(a):return offset+(a*5).clamp(-clip,clip)*float(ctl.action_scale)*effort/env.base.p_gains
    zero=read(ZERO)['physics'];initial0=read(ORIGINAL_TEACHER/'dev-2-initial-state.json');common={3.:read(PROBE/'common-state.json')}
    mode='baseline';record=False;control=[];last=None;raw=torch.zeros_like(env.standing);gate=ContactGate();fade=ReleaseFade();delta_action=torch.zeros(1,29,device=dev)
    def act(obs,z,mean=True):
        nonlocal last
        nominal=original_act(obs,env.standing,mean=mean);applied=delta_action
        if mode=='teacher' and fade.release_start is not None:
            applied=torch.zeros_like(nominal);applied[:,arm]=torch.as_tensor(fade.value(float(env.tether.time)),device=dev,dtype=nominal.dtype)/scale[:,arm]
        chosen=constrained(model,env.standing,raw,gate.value(float(env.tether.time)))[0] if mode=='student' else env.standing
        action=original_act(obs,chosen,mean=mean) if mode=='student' else nominal+applied if mode=='teacher' else nominal
        q=qtarget(action);q0=qtarget(nominal);dq=(q-q0)[0].cpu().numpy()
        if record:
            jac=env.tether.jac[:,vadr[arm]].copy();point=env.last_point.copy();error=env.tether.anchor-point
            bounds=torch.tensor(limits,device=dev);excess=(bounds[:,0]-q).clamp_min(0)+(q-bounds[:,1]).clamp_min(0)
            row=dict(time=env.steps*env.dt,tether_time=float(env.tether.time),gate=gate.value(float(env.tether.time)),executed_z=chosen.tolist(),bfm_obs={k:v.tolist() for k,v in obs.items()},
                action=action[0].tolist(),nominal=nominal[0].tolist(),q_PD=q[0].tolist(),nominal_q_PD=q0[0].tolist(),teacher_delta_q=dq.tolist(),student_delta_q=dq.tolist(),
                pre_qpos=env.sim.data.qpos.tolist(),pre_qvel=env.sim.data.qvel.tolist(),pre_wrist=point.tolist(),pre_error=error.tolist(),observed_force=env.tether.force.tolist(),
                rightarm_jacobian=jac.tolist(),rightarm_cartesian_target_delta=(jac@dq[arm]).tolist(),teacher_nonarm_action_delta_max=float((action[:,nonarm]-nominal[:,nonarm]).abs().max()),
                action_saturation=float(((action*5).abs()>=clip).float().mean()),q_PD_limit_excess_max_rad=float(excess.max()),
                q_PD_step_l2_rad=0. if last is None else float(np.linalg.norm(q[0].cpu().numpy()-last['q_PD'])),
                action_step_l2=0. if last is None else float(np.linalg.norm(action[0].cpu().numpy()-last['action'])),teacher_step_l2_rad=0. if last is None else float(np.linalg.norm(dq-last['teacher_delta_q'])))
            control.append(row);last=row
        return action
    model.act=act
    def snapshot():
        spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(env.sim.model,spec));mujoco.mj_getState(env.sim.model,env.sim.data,state,spec)
        tensors=lambda obj:{k:v.tolist() for k,v in vars(obj).items() if torch.is_tensor(v)}
        return dict(mj_integration=state.tolist(),base_tensors=tensors(env.base),sim_tensors=tensors(env.sim),wrapped_tensors=tensors(env.wrapped),
            bfm_obs={k:v.tolist() for k,v in env.obs.items()},previous_z=env.previous_z.tolist(),previous_action=env.previous_action.tolist(),
            force_history=env.history.tolist(),steps=env.steps,common_step_counter=env.base.common_step_counter,
            history={k:v.tolist() for k,v in env.base.history_handler.history.items()},
            env_bookkeeping=dict(done=bool(env.done),failed=bool(env.failed),last_qpos=env.last_qpos.tolist(),last_point=env.last_point.tolist(),last_support=env.last_support),
            tether={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in vars(env.tether).items() if k in ['origin','delta','time','force','anchor','active','body','jac']})
    def reset_integration(ref):
        for name in ['rew_buf','push_robot_counter','push_robot_plot_counter','command_counter','average_episode_length','torques','last_episode_length_buf']:
            v=getattr(env.base,name);v.copy_(torch.as_tensor(ref['base_tensors'][name],device=v.device,dtype=v.dtype))
        env.base.common_step_counter=ref['common_step_counter']
        mujoco.mj_setState(m,env.sim.data,np.asarray(ref['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(m,env.sim.data);mujoco.mj_setState(m,env.sim.data,np.asarray(ref['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION)
    result=dict(episodes=[],first_physical_failure=None,total_branch_seconds=0.,prefix_seconds=0.);reused={};summary={}
    for task in protocol['evaluations']:
        short='prefix' in task;onset=task.get('prefix',task.get('onset'));sign='plus' if task['direction']==1 else 'minus';key=task['id']
        evaluation_task=dict(id=key,direction=task['direction'],distance_m=.035,onset_s=onset,seed=0)
        labels=['baseline','teacher','student'] if key.startswith('new27') else ['student'];summary[key]={};baseline=None
        if key.startswith('dev3'):
            files={'baseline':PROBE/(sign+'-baseline.json'),'teacher':TEACHER/(sign+'-teacher.json'),'old-student':PROBE/(sign+'-student.json')}
        elif not short:
            files={'baseline':(SINGLE if key=='dev-2' else ORIGINAL_TEACHER if key=='dev-3' else SHARED)/(key+'-baseline.json'),
                   'teacher':(SINGLE if key=='dev-2' else SHARED)/(key+'-teacher.json'),'old-student':SHARED/(key+'-student.json')}
        else:files={}
        for label,path in files.items():
            reused[str(path)]=sha(path)
            if label=='baseline':baseline=read(path)
        for label in labels:
            check_deadline();record=False;mode='baseline';cfg.activation_time=onset
            if short:
                cfg.enabled=False;env.reset(seed=0,trajectory=(zero[0]['wrist'],[0.,0.,0.]));reset_integration(initial0)
                for _ in range(round(onset*10)):
                    check_deadline();_,_,done,_=env.step(torch.zeros_like(env.standing),baseline=True);assert not done
                count=round(onset*200);assert len(env.trace)==count
                diff={k:float(abs(np.array([r[k] for r in env.trace])-np.array([r[k] for r in zero[:count]])).max()) for k in ['qpos','qvel','wrist','root','force']}
                save(out,key+'-'+label+'-prefix-parity.json',diff);assert max(diff.values())==0
                if onset in common:
                    ref=torch.as_tensor(common[onset]['base_tensors']['rew_buf'],device=dev,dtype=env.base.rew_buf.dtype)
                    save(out,key+'-'+label+'-reward-cache.json',dict(max_abs_change=float((env.base.rew_buf-ref).abs().max())));env.base.rew_buf.copy_(ref)
                state=snapshot()
                if onset not in common:common[onset]=state;save(out,key+'-common-state.json',state)
                if state!=common[onset]:
                    save(out,'state-mismatch.json',dict(task=task,label=label,reference=common[onset],actual=state));raise AssertionError('Full common-state mismatch')
                initial=copy.deepcopy(env.trace[-1]);anchor=np.array(initial['wrist']);initial['anchor']=anchor.tolist()
                env.tether.origin=anchor.copy();env.tether.anchor=anchor.copy();env.tether.delta=np.array([task['direction']*.035,0.,0.]);cfg.enabled=True
                result['prefix_seconds']+=onset
            else:
                anchor=np.asarray(next(r['wrist'] for r in zero if r['time']>=onset));cfg.enabled=True
                env.reset(seed=0,trajectory=(anchor,[task['direction']*.035,0.,0.]))
                ref=read(ORIGINAL_TEACHER/('dev-2-initial-state.json' if task['direction']==1 else 'dev-3-initial-state.json'))
                for k in ['origin','anchor']:ref['tether'][k]=anchor.tolist()
                ref['tether']['delta']=[task['direction']*.035,0.,0.];reset_integration(ref);state=snapshot()
                legacy={k:state[k] for k in ref};legacy['previous_action']=None;legacy['tether']={k:state['tether'][k] for k in ref['tether']}
                if legacy!=ref:
                    save(out,'state-mismatch.json',dict(task=task,reference=ref,actual=legacy));raise AssertionError('Historical initial snapshot mismatch')
                initial=dict(qpos=env.sim.data.qpos.tolist(),qvel=env.sim.data.qvel.tolist(),wrist=env.initial_point.tolist(),root=env.sim.data.qpos[:3].tolist(),feet_xyz=env.sim.data.xpos[env.feet].tolist())
            name=key+'-'+label;save(out,name+'-initial-state.json',state)
            if baseline is not None:assert initial==baseline['initial']
            mode=label;record=True;control=[];last=None;gate=ContactGate();fade=ReleaseFade();raw=torch.zeros_like(env.standing);delta_action=torch.zeros(1,29,device=dev);decisions=[];start=len(env.trace)
            for high in range(round(task['seconds']*10)):
                check_deadline();point=env.last_point.copy();force=env.tether.force.copy();active=bool(env.tether.active);now=float(env.tether.time);gate.update(now,force)
                h=deploy_input(model,env.obs,env.history,env.keys)
                if label=='student':raw=net(h).detach()
                desired=np.zeros(7);error=env.tether.anchor-point;nominal=original_act(env.obs,env.standing,mean=True)
                if label=='teacher' and active:
                    point,_=env.tether.point_state();error=env.tether.anchor-point
                    desired=arm_ik(env.tether.jac[:,vadr[arm]],error)
                    q0=qtarget(nominal).cpu().numpy().reshape(-1)
                    desired=np.clip(q0[arm]+desired,limits[arm,0],limits[arm,1])-q0[arm]
                correction=fade.update(now,active,desired) if label=='teacher' else np.zeros(7)
                assert abs(correction).max()<.040001 and np.linalg.norm(correction)<.080001
                delta_action=torch.zeros_like(nominal);delta_action[:,arm]=torch.tensor(correction,device=dev,dtype=nominal.dtype)/scale[:,arm]
                decisions.append(dict(time=env.steps*env.dt,tether_time=now,active=active,force_xyz_n=force.tolist(),force_history=env.history.tolist(),bfm_obs={k:v.tolist() for k,v in env.obs.items()},student_h=h.tolist(),raw=raw.tolist(),correction_rad=correction.tolist()))
                _,_,done,info=env.step(torch.zeros_like(env.standing),baseline=True)
                if done:break
            record=False
            ep=dict(task=evaluation_task,label=label,initial=initial,physics=copy.deepcopy(env.trace[start:]),control=control,decisions=decisions,failed=bool(env.failed or info['failed']),same_initial_state_sha256=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest())
            save(out,name+'.json',ep)
            if label=='baseline':baseline=ep
            assert baseline is not None
            assert [r['anchor'] for r in ep['physics']]==[r['anchor'] for r in baseline['physics'][:len(ep['physics'])]]
            audit=audit_episode(ep,contract,state);save(out,name+'-pd-audit.json',audit)
            summary[key][label]=summarize(ep);save(out,'summary.json',summary)
            if ep['failed'] and result['first_physical_failure'] is None:result['first_physical_failure']=name
            result['episodes'].append(dict(name=name,complete=len(ep['physics'])==round(task['seconds']*200) and not ep['failed'],physics_samples=len(ep['physics']),audit=audit))
            result['total_branch_seconds']+=len(ep['physics'])*.005;save(out,'evaluation-result.json',result);print('EVAL_COMPLETE',name,len(ep['physics']),flush=True)
    save(out,'evaluation-reused-data.json',reused);assert all(sha(Path(p))==v for p,v in reused.items())
    model.act=original_act;result['all_complete']=all(e['complete'] for e in result['episodes']);save(out,'evaluation-result.json',result)
    return result
