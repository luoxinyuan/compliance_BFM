"""Bounded original teacher positive control at the saved t=3s common state.
Two one-second teacher branches; reuse pinned standing controls; no Student runs or training.
"""
import ast,copy,hashlib,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
OUT=Path(os.environ['BFM_RECOVERY_OUTPUT']) if 'BFM_RECOVERY_OUTPUT' in os.environ else None
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUNS=P/'bridge-v2-owned/runs'
OLD=RUNS/'issue20-ccl01-20261028T082201Z/attempt-04/outputs'
MODEL=RUNS/'issue19-recovery-20261028T045520Z/model-view'
STANDING=RUNS/'issue19-wrist-anchor-20261028T062503Z/attempt-02/outputs/standing_latent.json'
SOURCE=RUNS/'issue21-bidirectional-20261030T012400Z/attempt-01/outputs'
TEACHER_SOURCE=RUNS/'issue21-ccl02a-20261029T153900Z/attempt-01/outputs'
STUDENT_SHA='a5daf853276d55f5b8c099df4f48917f572a32c48e548d2c90f39e593d4a809d'
REFERENCE=RUNS/'issue21-direction-20261030T020000Z/attempt-02/outputs'
PREFIX_SECONDS=3.;BRANCH_SECONDS=1.



def ipc():
    if sys.platform!='linux' or OUT is None:return
    import multiprocessing.connection as c
    if getattr(c.arbitrary_address,'_ccl02a',False):return
    original=c.arbitrary_address
    def address(family):return b'\0bfm-ccl02a-'+os.urandom(12).hex().encode() if family=='AF_UNIX' else original(family)
    address._ccl02a=True;c.arbitrary_address=address
ipc()

def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    import multiprocessing as mp
    mp.set_start_method('forkserver',force=True);(OUT/'ipc').mkdir();mp.current_process()._config['tempdir']='ipc'
    with mp.Manager() as manager:q=manager.Queue();q.put('ready');assert q.get()=='ready'
    constants={}
    for node in ast.parse((ROOT/'research_notes/g1_action_contract_check.py').read_text()).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['PINS','WEIGHT_SHA']:
            constants[node.targets[0].id]=ast.literal_eval(node.value)
    assert all(digest(ROOT/f)==h for f,h in constants['PINS'].items())
    assert digest(MODEL/'checkpoint/model/model.safetensors')==constants['WEIGHT_SHA']
    assert digest('/mnt/data-oss/users/yi.gao/compliance_BFM/data/lafan_29dof.pkl')=='f3a0c2810363f5c50bf4146fa2db33c1ff5b90d00cb7c0bc2aa4622696375e11'
    assert digest(SOURCE/'contact-projector-final.pt')==STUDENT_SHA
    import numpy as np,torch,mujoco
    from dataclasses import asdict
    from omegaconf import OmegaConf
    from humanoidverse.force_compliance.config import ComplianceConfig
    from humanoidverse.force_compliance.environment import ComplianceEnv
    from humanoidverse.agents import load_utils
    from frozen_bfm_placement import cpu_load_then_stage,tensor_digest,INFERENCE_MODULES
    from ccl02a_teacher import audit_episode,arm_ik,ReleaseFade
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.device_count()==1
    class LoggedOriginalEnv(ComplianceEnv):
        def reset(self,*args,**kwargs):
            self.trace=[];obs=super().reset(*args,**kwargs)
            self.feet=[mujoco.mj_name2id(self.sim.model,mujoco.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link') for s in ['left','right']]
            self.initial_feet=self.sim.data.xpos[self.feet].copy()
            return obs
        def _physics_step(self):
            force_time=float(self.tether.time)
            # Analytic partner velocity for logging only; native Tether still applies force.
            u=float(np.clip((force_time-self.tether.cfg.activation_time)/self.tether.cfg.ramp_duration,0,1))
            anchor_velocity=self.tether.delta*6*u*(1-u)/self.tether.cfg.ramp_duration if 0<u<1 else np.zeros(3)
            super()._physics_step()
            m,d=self.sim.model,self.sim.data
            pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis');up=d.xmat[pelvis].reshape(3,3)[2,2]
            effort=self.base.torque_limits.cpu().reshape(-1).numpy()
            limited=np.flatnonzero(m.jnt_limited);pos=d.qpos[m.jnt_qposadr[limited]];bounds=m.jnt_range[limited]
            excess=np.maximum(bounds[:,0]-pos,0)+np.maximum(pos-bounds[:,1],0)
            self.trace.append(dict(time=float(self.tether.time),force_time=force_time,active=bool(self.tether.active),force=self.tether.force.tolist(),
                anchor=self.tether.anchor.tolist(),anchor_velocity=anchor_velocity.tolist(),wrist=self.last_point.tolist(),root=d.qpos[:3].tolist(),error=float(np.linalg.norm(self.tether.anchor-self.last_point)),
                height=float(d.qpos[2]),tilt_deg=math.degrees(math.acos(float(np.clip(up,-1,1)))),support=list(self.last_support),
                feet_xyz=d.xpos[self.feet].tolist(),joint_limit_excess_max_rad=float(excess.max()),
                torque_saturation=float(np.mean(abs(d.ctrl[6:])>=effort-1e-6)),failed=bool(self.failed),
                qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),torque=d.ctrl[6:].tolist(),**self.contact_sample))
    cfg=ComplianceConfig(enabled=False,max_force=5.,anchor_delay=0.,release_time=4.5,seed=0)
    placement=[]
    with cpu_load_then_stage(load_utils,placement):env=LoggedOriginalEnv(MODEL,cfg,device='cuda')
    write('inference-placement.json',placement)
    env.standing=torch.tensor(json.loads(STANDING.read_text()),device='cuda',dtype=torch.float32).reshape(1,256)
    model=env.model;original_act=model.act;native_step=env._original_step
    frozen={n:tensor_digest(getattr(model,n)) for n in INFERENCE_MODULES};versions={n:p._version for n,p in model.named_parameters()}
    def sample_native_contacts():
        # Capture forces BEFORE ComplianceEnv removes the applied Tether wrench
        # and calls mj_forward; querying later would recompute without that load.
        m,d=env.sim.model,env.sim.data;before_v=d.qvel.copy();native_step()
        forces=[0.,0.];speed=[];undesired=0
        for i,contact in enumerate(d.contact[:d.ncon]):
            bodies=[int(m.geom_bodyid[contact.geom1]),int(m.geom_bodyid[contact.geom2])]
            if 0 not in bodies:continue
            body=max(bodies);name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,body) or ''
            if 'ankle' not in name and 'foot' not in name:undesired+=1;continue
            side=0 if 'left' in name else 1;f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);forces[side]+=abs(float(f[0]))
            jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,contact.pos,body)
            tangent=contact.frame.reshape(3,3)[1:]@(jac@before_v);speed.append(float(np.linalg.norm(tangent)))
        env.contact_sample=dict(foot_normal_force_n=forces,contact_tangent_speed_max_mps=max(speed,default=0.),undesired_ground_contacts=undesired)
    env._original_step=sample_native_contacts
    names=list(env.sim.dof_names);arm=[i for i,n in enumerate(names) if n.startswith('right_') and any(x in n for x in ('shoulder','elbow','wrist'))]
    assert len(names)==29 and arm==list(range(22,29));nonarm=[i for i in range(29) if i not in arm]
    ids=[mujoco.mj_name2id(env.sim.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names];assert min(ids)>=0
    vadr=env.sim.model.jnt_dofadr[ids];qlimits=env.sim.model.jnt_range[ids]
    ctl=env.base.config.robot.control;assert ctl.control_type=='P' and ctl.action_rescale and ctl.normalize_action
    ratio=float(ctl.normalize_action_to)/float(ctl.normalize_action_from);assert ratio==5.
    effort=torch.tensor(list(env.base.config.robot.dof_effort_limit_list),device='cuda');clip=float(ctl.action_clip_value)
    scale=(ratio*float(ctl.action_scale)*effort/env.base.p_gains).reshape(1,-1);offset=env.base.default_dof_pos+env.base.default_dof_pos_offset
    def qtarget(action):return offset+(action*ratio).clamp(-clip,clip)*float(ctl.action_scale)*effort/env.base.p_gains
    contract=dict(dof_names=names,arm_indices=arm,q_PD_scale=scale.tolist(),q_PD_offset=offset.tolist(),normalize_ratio=ratio,clip=clip,
        control=OmegaConf.to_container(ctl,resolve=True),bfm_inputs=env.keys,source_pins=constants['PINS'],model_sha256=constants['WEIGHT_SHA'])
    write('action-contract.json',contract)
    zero=json.loads((OLD/'anchor-reference.json').read_text())['physics']
    ref_initial=json.loads((TEACHER_SOURCE/'dev-2-initial-state.json').read_text())
    protocol=json.loads((ROOT/'research_notes/direction_teacher_protocol.json').read_text());write('protocol.json',protocol)
    assert digest(ROOT/'research_notes/ccl02a_teacher.py')==protocol['teacher_source_sha256']
    reused={str(REFERENCE/k):v for k,v in protocol['reference_sha256'].items()}
    assert all(digest(path)==sha for path,sha in reused.items());write('reused-inputs.json',reused)
    common=json.loads((REFERENCE/'common-state.json').read_text());write('common-state.json',common)
    baselines={sign:json.loads((REFERENCE/(sign+'-baseline.json')).read_text()) for sign in ['plus','minus']}
    assert contract==json.loads((REFERENCE/'action-contract.json').read_text())
    mode='baseline';record=False;control=[];last_control=None;fade=ReleaseFade();delta_action=torch.zeros(1,29,device='cuda')
    def act(obs,z,mean=True):
        nonlocal last_control
        nominal=original_act(obs,env.standing,mean=mean)
        applied_delta=delta_action
        if mode=='teacher' and fade.release_start is not None:
            applied_delta=torch.zeros_like(nominal)
            applied_delta[:,arm]=torch.as_tensor(fade.value(float(env.tether.time)),device='cuda',dtype=nominal.dtype)/scale[:,arm]
        action=nominal+applied_delta if mode=='teacher' else nominal
        q=qtarget(action);q0=qtarget(nominal);dq=(q-q0)[0].cpu().numpy()
        if record:
            # Jacobian last computed by native Tether.point_state; no extra mj_forward.
            jac=env.tether.jac[:,vadr[arm]].copy()
            before=env.last_point.copy();err=env.tether.anchor-before
            qlimits_t=torch.tensor(qlimits,device='cuda');excess=(qlimits_t[:,0]-q).clamp_min(0)+(q-qlimits_t[:,1]).clamp_min(0)
            row=dict(time=env.steps*env.dt,tether_time=float(env.tether.time),active=bool(env.tether.active),executed_z=env.standing.tolist(),
                bfm_obs={k:v.tolist() for k,v in obs.items()},action=action[0].tolist(),nominal=nominal[0].tolist(),q_PD=q[0].tolist(),nominal_q_PD=q0[0].tolist(),
                pre_qpos=env.sim.data.qpos.tolist(),pre_qvel=env.sim.data.qvel.tolist(),pre_wrist=before.tolist(),pre_error=err.tolist(),
                observed_force=env.tether.force.tolist(),rightarm_jacobian=jac.tolist(),teacher_delta_q=dq.tolist(),rightarm_cartesian_target_delta=(jac@dq[arm]).tolist(),
                action_saturation=float(((action*ratio).abs()>=clip).float().mean()),q_PD_limit_excess_max_rad=float(excess.max()),
                q_PD_step_l2_rad=0. if last_control is None else float(np.linalg.norm(q[0].cpu().numpy()-last_control['q_PD'])))
            assert float(abs(dq).max())<.040001 and float(np.linalg.norm(dq))<.080001 and not np.any(dq[nonarm])
            control.append(row);last_control=row
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
    result=dict(status='running',model_loaded=True,original_wrapper_used=True,policy_updates=0,projection_steps=0,episodes=[],first_physical_failure=None)
    write('result.json',result);print('MODEL_READY_DIRECTION_TEACHER',flush=True)
    try:
        for direction in [1,-1]:
            for label in ['teacher']:
                # Reconstruct the one frozen common snapshot via the identical verified zero-force prefix.
                # This avoids partial restoration of hidden wrapper/history state.
                record=False;mode='baseline';cfg.enabled=False;cfg.activation_time=PREFIX_SECONDS
                env.reset(seed=0,trajectory=(zero[0]['wrist'],[0.,0.,0.]))
                for name in ['rew_buf','push_robot_counter','push_robot_plot_counter','command_counter','average_episode_length','torques','last_episode_length_buf']:
                    value=getattr(env.base,name);value.copy_(torch.as_tensor(ref_initial['base_tensors'][name],device=value.device,dtype=value.dtype))
                env.base.common_step_counter=ref_initial['common_step_counter']
                mujoco.mj_setState(env.sim.model,env.sim.data,np.asarray(ref_initial['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION)
                mujoco.mj_forward(env.sim.model,env.sim.data)
                mujoco.mj_setState(env.sim.model,env.sim.data,np.asarray(ref_initial['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION)
                for _ in range(30):
                    assert time.monotonic()<float(os.environ['CCL_DEADLINE_MONOTONIC'])
                    _,_,done,_=env.step(torch.zeros_like(env.standing),baseline=True);assert not done
                assert env.steps==150 and len(env.trace)==600
                name=('plus' if direction==1 else 'minus')+'-'+label
                diffs={k:float(abs(np.array([r[k] for r in env.trace])-np.array([r[k] for r in zero[:600]])).max()) for k in ['qpos','qvel','wrist','root','force']}
                write(name+'-prefix-parity.json',diffs);assert max(diffs.values())==0
                # Original reward scratch buffer is not an actor/physics input.
                # Reconstruct its saved bookkeeping value; every physical/observable field is still checked exactly.
                if common is not None:
                    ref_reward=torch.as_tensor(common['base_tensors']['rew_buf'],device=env.base.rew_buf.device,dtype=env.base.rew_buf.dtype)
                    write(name+'-reward-cache-restore.json',dict(max_abs_change=float((env.base.rew_buf-ref_reward).abs().max())))
                    env.base.rew_buf.copy_(ref_reward)
                state=snapshot()
                if state!=common:
                    write('state-mismatch.json',dict(name=name,reference=common,actual=state));raise AssertionError('Common full mechanical/proprio/action snapshot differs')
                assert not np.any(env.history) and not np.any(env.tether.force)
                initial=copy.deepcopy(env.trace[-1]);anchor=np.array(initial['wrist']);initial['anchor']=anchor.tolist()
                env.tether.origin=anchor.copy();env.tether.anchor=anchor.copy();env.tether.delta=np.array([direction*.035,0.,0.]);cfg.enabled=True
                write(name+'-branch-initial.json',dict(anchor=anchor.tolist(),delta=env.tether.delta.tolist(),spring_preload=(80*(anchor-np.asarray(initial['wrist']))).tolist(),
                    initial_damping_force=(-8*(env.tether.jac@env.sim.data.qvel)).tolist(),tether_time=env.tether.time,mechanical_history_numeric_equal=True))
                baseline=baselines['plus' if direction==1 else 'minus']
                assert initial==baseline['initial']
                assert hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()==baseline['common_state_sha256']
                control=[];last_control=None;mode=label;record=True;fade=ReleaseFade();delta_action=torch.zeros(1,29,device='cuda');decisions=[];start=len(env.trace)
                for high in range(10):
                    assert time.monotonic()<float(os.environ['CCL_DEADLINE_MONOTONIC'])
                    point=env.last_point.copy();force=env.tether.force.copy();active=bool(env.tether.active);now=float(env.tether.time)
                    desired=np.zeros(7);error=env.tether.anchor-point;nominal=original_act(env.obs,env.standing,mean=True)
                    if label=='teacher' and active:
                        point,_=env.tether.point_state();error=env.tether.anchor-point
                        desired=arm_ik(env.tether.jac[:,vadr[arm]],error)
                        q0=qtarget(nominal).cpu().numpy().reshape(-1)
                        desired=np.clip(q0[arm]+desired,qlimits[arm,0],qlimits[arm,1])-q0[arm]
                    correction=fade.update(now,active,desired) if label=='teacher' else np.zeros(7)
                    assert abs(correction).max()<.040001 and np.linalg.norm(correction)<.080001
                    delta_action=torch.zeros_like(nominal);delta_action[:,arm]=torch.tensor(correction,device='cuda',dtype=nominal.dtype)/scale[:,arm]
                    decisions.append(dict(time=env.steps*env.dt,tether_time=now,active=active,error_xyz_m=error.tolist(),force_xyz_n=force.tolist(),
                        correction_rad=correction.tolist(),release_detected_at=fade.release_start,
                        bfm_obs={k:v.tolist() for k,v in env.obs.items()},force_history=env.history.tolist()))
                    _,_,done,info=env.step(torch.zeros_like(env.standing),baseline=True)
                    if done:break
                record=False
                ep=dict(label=label,direction=direction,initial=initial,physics=copy.deepcopy(env.trace[start:]),control=control,decisions=decisions,
                    failed=bool(env.failed or info['failed']),common_state_sha256=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest())
                assert [r['anchor'] for r in ep['physics']]==[r['anchor'] for r in baseline['physics'][:len(ep['physics'])]]
                write(name+'.json',ep);audit=audit_episode(ep,contract,state);write(name+'-pd-audit.json',audit)
                if ep['failed'] and result['first_physical_failure'] is None:result['first_physical_failure']=name
                result['episodes'].append(dict(name=name,complete=len(ep['physics'])==200 and not ep['failed'],physics_samples=len(ep['physics']),audit=audit));write('result.json',result)
                print('BRANCH_COMPLETE',name,len(ep['physics']),flush=True)
        assert {n:tensor_digest(getattr(model,n)) for n in INFERENCE_MODULES}==frozen
        assert all(p._version==versions[n] and p.grad is None for n,p in model.named_parameters())
        assert all(digest(path)==sha for path,sha in reused.items())
        assert digest(SOURCE/'contact-projector-final.pt')==STUDENT_SHA
        assert digest(MODEL/'checkpoint/model/model.safetensors')==constants['WEIGHT_SHA']
        result.update(status='completed',all_complete=all(e['complete'] for e in result['episodes']),bfm_unchanged=True,student_unchanged=True,
            student_sha256=STUDENT_SHA,student_executions=0,total_branch_seconds=sum(e['physics_samples'] for e in result['episodes'])*.005,prefix_seconds=6.,total_simulated_seconds=6.+sum(e['physics_samples'] for e in result['episodes'])*.005)
        write('result.json',result);print('COMPLETED_DIRECTION_TEACHER',flush=True)
    finally:env.close()

if __name__=='__main__':
    try:main()
    except Exception as exc:write('failure.json',dict(error_type=type(exc).__name__,first_boundary=str(exc)));raise
