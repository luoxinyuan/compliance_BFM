"""CCL bidirectional supervised projector using the validated CCL02A runtime.
Original BFM/physics unchanged; two balanced teacher traces, unchanged bounded MLP, presealed heldout.
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
TASKS=[{'id': 'dev-2', 'direction': 1, 'distance_m': 0.035, 'onset_s': 1.1, 'seed': 0}, {'id': 'dev-3', 'direction': -1, 'distance_m': 0.035, 'onset_s': 1.1, 'seed': 0}, {'id': 'holdout-plus', 'direction': 1, 'distance_m': 0.035, 'onset_s': 1.25, 'seed': 0}, {'id': 'holdout-minus', 'direction': -1, 'distance_m': 0.035, 'onset_s': 1.25, 'seed': 0}]
TRAIN_SOURCE=RUNS/'issue21-projector-20261030T004000Z/attempt-01/outputs'
STUDENT_SHA='14b5a32798586d0810ff5e5461ce745890a90c717d1e6de2e8d6dc3f2a296fd8'
TEACHER_SOURCE=RUNS/'issue21-ccl02a-20261029T153900Z/attempt-01/outputs'


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


def arm_ik(jacobian,error,joint_bound=.04,norm_bound=.08,damping=.02):
    # Exact bounded positional Jacobian step from Issue20@53fae2d.
    import numpy as np
    delta=jacobian.T@np.linalg.solve(jacobian@jacobian.T+damping**2*np.eye(3),error)
    delta=np.clip(delta,-joint_bound,joint_bound)
    return delta*min(1.,norm_bound/max(float(np.linalg.norm(delta)),1e-12))


class ReleaseFade:
    """Causal 0.2s linear release from the last active correction; no future input."""
    def __init__(self):
        import numpy as np
        self.last=np.zeros(7);self.release_start=None;self.previous_active=False
    def value(self,now):
        factor=0. if self.release_start is None else max(0.,1.-(now-self.release_start)/.2)
        if factor<1e-12:factor=0.
        return self.last*factor
    def update(self,now,active,desired=None):
        import numpy as np
        if active:
            self.last=np.array(desired,copy=True);self.release_start=None;answer=self.last.copy()
        else:
            if self.previous_active:self.release_start=now
            answer=self.value(now)
        self.previous_active=active
        return answer


def phase(t,onset):
    if t<onset:return 'pre_contact'
    if t<onset+2.:return 'ramp'
    if t<4.5:return 'hold'
    return 'release'


def summarize(episode):
    import numpy as np
    task=episode['task'];rows=episode['physics'];control=episode['control'];output={}
    for label in ['whole','pre_contact','ramp','hold','release']:
        indices=[i for i,r in enumerate(rows) if label=='whole' or phase(r['force_time'],task['onset_s'])==label]
        cs=[r for r in control if label=='whole' or phase(r['tether_time'],task['onset_s'])==label]
        if not indices:output[label]=None;continue
        selected=[rows[i] for i in indices];before=episode['initial'] if indices[0]==0 else rows[indices[0]-1]
        start_w=np.array(before['wrist']);start_r=np.array(before['root']);err0=np.array(selected[0]['anchor'])-start_w
        norm=float(np.linalg.norm(err0));unit=err0/norm if norm>1e-9 else None
        wrist=np.array([r['wrist'] for r in selected]);root=np.array([r['root'] for r in selected]);force=np.array([r['force'] for r in selected]);err=np.array([r['error'] for r in selected])
        dw=wrist[-1]-start_w;dr=root[-1]-start_r;rel=dw-dr
        feet=np.array([r['feet_xyz'] for r in selected]);feet0=np.array(before['feet_xyz'])
        rms=lambda a:float(np.sqrt(np.mean(np.square(a))))
        arr=lambda key:np.array([c[key] for c in cs])
        output[label]=dict(samples=len(selected),duration_s=len(selected)*.005,error_rms_mm=1000*rms(err),error_peak_mm=1000*float(err.max()),
            force_rms_n=rms(np.linalg.norm(force,axis=1)),force_peak_n=float(np.linalg.norm(force,axis=1).max()),force_mean_xyz_n=force.mean(0).tolist(),
            wrist_delta_xyz_mm=(1000*dw).tolist(),root_delta_xyz_mm=(1000*dr).tolist(),wrist_relative_root_delta_xyz_mm=(1000*rel).tolist(),
            wrist_along_initial_error_mm=None if unit is None else float(1000*dw@unit),
            relative_along_initial_error_mm=None if unit is None else float(1000*rel@unit),initial_error_norm_mm=1000*norm,
            wrist_nominal_X_mm=float(1000*task['direction']*dw[0]),relative_nominal_X_mm=float(1000*task['direction']*rel[0]),
            root_height_min_m=min(r['height'] for r in selected),tilt_max_deg=max(r['tilt_deg'] for r in selected),
            root_displacement_phase_max_mm=float(1000*np.linalg.norm(root-start_r,axis=1).max()),
            double_contact_fraction=float(np.mean([all(r['support']) for r in selected])),
            both_feet_positive_normal_force_fraction=float(np.mean([all(v>1e-6 for v in r['foot_normal_force_n']) for r in selected])),
            total_foot_normal_force_min_n=min(sum(r['foot_normal_force_n']) for r in selected),
            contact_tangent_speed_rms_mps=rms([r['contact_tangent_speed_max_mps'] for r in selected]),
            contact_tangent_speed_peak_mps=max(r['contact_tangent_speed_max_mps'] for r in selected),
            foot_displacement_phase_max_mm=float(1000*np.linalg.norm(feet[:,:,:2]-feet0[None,:,:2],axis=2).max()),
            joint_limit_excess_max_rad=max(r['joint_limit_excess_max_rad'] for r in selected),
            torque_saturation_fraction=float(np.mean([r['torque_saturation'] for r in selected])),
            action_saturation_fraction=float(np.mean(arr('action_saturation'))),
            q_PD_limit_excess_max_rad=float(arr('q_PD_limit_excess_max_rad').max()),
            q_PD_step_max_l2_rad=float(arr('q_PD_step_l2_rad').max()),q_PD_step_rms_l2_rad=rms(arr('q_PD_step_l2_rad')),
            action_step_max_l2=float(arr('action_step_l2').max()),teacher_step_max_l2_rad=float(arr('teacher_step_l2_rad').max()),
            teacher_q_PD_rightarm_rms_rad=rms(arr('teacher_delta_q')[:,22:29]),teacher_q_PD_max_abs_rad=float(abs(arr('teacher_delta_q')).max()),
            teacher_q_PD_max_l2_rad=float(np.linalg.norm(arr('teacher_delta_q'),axis=1).max()),
            teacher_nonarm_action_delta_max=float(arr('teacher_nonarm_action_delta_max').max()),
            failed=any(r['failed'] for r in selected))
    return output


def audit_episode(episode,contract,state):
    """Recompute executed q_PD and 200Hz original PD torque from saved records."""
    import numpy as np
    scale=np.array(contract['q_PD_scale'])[0];offset=np.array(contract['q_PD_offset'])[0]
    b=state['base_tensors'];pg=np.array(b['p_gains'],dtype=np.float32);dg=np.array(b['d_gains'],dtype=np.float32)
    kp=np.array(b['_kp_scale'],dtype=np.float32).reshape(-1);kd=np.array(b['_kd_scale'],dtype=np.float32).reshape(-1);limits=np.array(b['torque_limits'],dtype=np.float32)
    errors=dict(q_PD_max_error=0.,torque_max_error=0.)
    for j,row in enumerate(episode['physics']):
        c=episode['control'][j//4];q=np.array(c['q_PD']);a=np.array(c['action'])
        errors['q_PD_max_error']=max(errors['q_PD_max_error'],float(abs(offset+np.clip(a*5,-5,5)*scale/5-q).max()))
        before=dict(qpos=c['pre_qpos'],qvel=c['pre_qvel']) if j%4==0 else episode['physics'][j-1]
        tau=kp*pg*(q.astype(np.float32)-np.array(before['qpos'][7:],dtype=np.float32))-kd*dg*np.array(before['qvel'][6:],dtype=np.float32)
        errors['torque_max_error']=max(errors['torque_max_error'],float(abs(np.clip(tau,-limits,limits)-np.array(row['torque'])).max()))
    assert errors['q_PD_max_error']<1e-6 and errors['torque_max_error']<1e-4
    return errors


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
    assert TASKS[:2]==[t for t in json.loads((OLD/'manifest.json').read_text())['dev'] if t['id'] in ['dev-2','dev-3']]
    sealed=json.loads((ROOT/'research_notes/bidirectional_protocol.json').read_text());assert sealed['tasks']==TASKS
    write('sealed-protocol.json',sealed);assert digest(TRAIN_SOURCE/'contact-projector-final.pt')==STUDENT_SHA
    import numpy as np,torch,mujoco
    from dataclasses import asdict
    from omegaconf import OmegaConf
    from humanoidverse.force_compliance.config import ComplianceConfig
    from humanoidverse.force_compliance.environment import ComplianceEnv
    from humanoidverse.agents import load_utils
    from frozen_bfm_placement import cpu_load_then_stage,tensor_digest,INFERENCE_MODULES
    from contact_projector import ContactGate,deploy_input,constrained,Projector
    from bidirectional_projector import fit_balanced
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
            force_time=float(self.tether.time);super()._physics_step()
            m,d=self.sim.model,self.sim.data
            pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis');up=d.xmat[pelvis].reshape(3,3)[2,2]
            effort=self.base.torque_limits.cpu().reshape(-1).numpy()
            limited=np.flatnonzero(m.jnt_limited);pos=d.qpos[m.jnt_qposadr[limited]];bounds=m.jnt_range[limited]
            excess=np.maximum(bounds[:,0]-pos,0)+np.maximum(pos-bounds[:,1],0)
            self.trace.append(dict(time=float(self.tether.time),force_time=force_time,active=bool(self.tether.active),force=self.tether.force.tolist(),
                anchor=self.tether.anchor.tolist(),wrist=self.last_point.tolist(),root=d.qpos[:3].tolist(),error=float(np.linalg.norm(self.tether.anchor-self.last_point)),
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
    zero=json.loads((OLD/'anchor-reference.json').read_text())['physics'];anchors={}
    for task in TASKS:
        row=next(r for r in zero if r['time']>=task['onset_s'])
        assert not any(row['force']) and row['time']-task['onset_s']<=.005+1e-10
        anchors[task['id']]=row['wrist']
    write('protocol.json',dict(base_commit='bb0833853c8e5cad1369d691b531972004635ff7',teacher_source_commit='53fae2df6a28c87b7d1be47b5b0f09178a3b6997',
        tasks=TASKS,config=asdict(cfg),absolute_anchors=anchors,state_source=str(OLD),standing_source=str(STANDING),standing_sha256=digest(STANDING),
        pairing='Original seed0 reset; paired integration snapshot restores leftover warm-start/ctrl/time; restore torque/episode bookkeeping; verify full mechanical/obs/history equality',
        episode_s=6.,teacher_hz=10,bfm_hz=50,pd_physics_hz=200,teacher_bounds=dict(joint_abs_rad=.04,joint_l2_rad=.08,damping=.02),
        contact_rule='At each 10Hz decision use last observed original Tether.active/anchor/force; pre-contact correction zero; active teacher queries current wrist/Jacobian exactly as Issue20; no extra Tether.measure calls',
        release_rule='On 10Hz observed active-to-inactive edge, linearly fade last correction over the same 0.2s, evaluated at each original 50Hz action; Jacobian still 10Hz; no future information or re-anchoring',
        phase_rule='Exact original floating Tether force_time: pre<onset, ramp<onset+2, hold<4.5, release>=4.5; inactive-anchor error is diagnostic geometry, not active tracking objective',
        benefit_reference='Balanced +/- training at1.1s; untouched +/-1.25s eval after final checkpoint freeze; see sealed-protocol',
        safety='Original min root .4m/up cosine .4/nonfinite/warnings; report physical failure, limits, saturation, support loads, tangential contact speed and action continuity; no outcome retries',
        privilege='Current anchor/world wrist/Jacobian belong to oracle; not student inputs',contact_forces='Captured after native mj_step before removing wrench/mj_forward; contact tangential velocity uses corresponding pre-integration qvel',
        policy_updates='projector only; see dataset-manifest',projection_steps='none; supervised MLP only',terminal_reopened=False,resources=dict(cpus=4,rss_mib=12288,owned_gpu_mib=4096,cumulative_wall_seconds=2400)))
    train_h=torch.load(TRAIN_SOURCE/'teacher-dataset.pt',map_location='cuda',weights_only=True)['h']
    net=Projector(train_h).to('cuda');net.load_state_dict(torch.load(TRAIN_SOURCE/'contact-projector-final.pt',map_location='cuda',weights_only=True)['state_dict'],strict=True)
    net.eval().requires_grad_(False);old_net_hash=tensor_digest(net);new_net_hash=None;fit_result=None
    raw_student=None;sensor_gate=ContactGate();executed_z=None
    mode='baseline';delta_action=torch.zeros(1,29,device='cuda');control=[];record=False;last_control=None
    def act(obs,z,mean=True):
        nonlocal last_control,executed_z
        nominal=original_act(obs,env.standing,mean=mean)
        applied_delta=delta_action
        if mode=='teacher' and fade.release_start is not None:
            applied_delta=torch.zeros_like(nominal)
            applied_delta[:,arm]=torch.as_tensor(fade.value(float(env.tether.time)),device='cuda',dtype=nominal.dtype)/scale[:,arm]
        action=nominal+applied_delta if mode=='teacher' else nominal
        if mode.startswith('student'):
            executed_z,_=constrained(model,env.standing,raw_student,sensor_gate.value(float(env.tether.time)))
            action=original_act(obs,executed_z,mean=mean)
        q=qtarget(action);q0=qtarget(nominal);corr=q-q0
        if record:
            qlimits_t=torch.tensor(qlimits,device='cuda');excess=(qlimits_t[:,0]-q).clamp_min(0)+(q-qlimits_t[:,1]).clamp_min(0)
            row=dict(bfm_obs={k:v.tolist() for k,v in obs.items()},executed_z=None if executed_z is None else executed_z.tolist(),gate=sensor_gate.value(float(env.tether.time)),time=env.steps*env.dt,tether_time=float(env.tether.time),pre_qpos=env.sim.data.qpos.tolist(),pre_qvel=env.sim.data.qvel.tolist(),action=action[0].tolist(),nominal=nominal[0].tolist(),q_PD=q[0].tolist(),
                nominal_q_PD=q0[0].tolist(),teacher_delta_q=corr[0].tolist(),teacher_nonarm_action_delta_max=float((action[:,nonarm]-nominal[:,nonarm]).abs().max()),
                action_saturation=float(((action*ratio).abs()>=clip).float().mean()),q_PD_limit_excess_max_rad=float(excess.max()),
                q_PD_step_l2_rad=0. if last_control is None else float(np.linalg.norm(q[0].cpu().numpy()-last_control['q_PD'])),
                action_step_l2=0. if last_control is None else float(np.linalg.norm(action[0].cpu().numpy()-last_control['action'])),
                teacher_step_l2_rad=0. if last_control is None else float(np.linalg.norm(corr[0].cpu().numpy()-last_control['teacher_delta_q'])))
            if not mode.startswith('student'):assert row['teacher_nonarm_action_delta_max']==0 and float(corr.abs().max())<.040001 and float(corr.norm())<.080001
            control.append(row);last_control=row
        return action
    model.act=act
    def snapshot():
        spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(env.sim.model,spec));mujoco.mj_getState(env.sim.model,env.sim.data,state,spec)
        return dict(mj_integration=state.tolist(),base_tensors={k:v.tolist() for k,v in vars(env.base).items() if torch.is_tensor(v)},
            bfm_obs={k:v.tolist() for k,v in env.obs.items()},previous_z=env.previous_z.tolist(),previous_action=None,force_history=env.history.tolist(),steps=env.steps,
            common_step_counter=env.base.common_step_counter,history={k:v.tolist() for k,v in env.base.history_handler.history.items()},
            tether={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in vars(env.tether).items() if k in ['origin','delta','time','force','anchor','active','body']})
    result=dict(status='running',model_loaded=True,original_wrapper_used=True,policy_updates=0,projection_steps=0,episodes=[],first_physical_failure=None)
    summaries={};pairs={};references={}
    for task in TASKS:
        sourceid='dev-2' if task['direction']==1 else 'dev-3'
        ref=json.loads((TEACHER_SOURCE/(sourceid+'-initial-state.json')).read_text())
        for k in ['origin','anchor']:ref['tether'][k]=anchors[task['id']]
        ref['tether']['delta']=[task['direction']*.035,0.,0.];references[task['id']]=ref
    reused={}
    for task in TASKS[:2]:
        summaries[task['id']]={}
        for label in ['baseline','teacher']:
            source=TRAIN_SOURCE if task['id']=='dev-2' else TEACHER_SOURCE
            path=source/(task['id']+'-'+label+'.json');ep=json.loads(path.read_text())
            pairs[(task['id'],label)]=ep;summaries[task['id']][label]=summarize(ep)
            reused[str(path)]=digest(path)
    write('reused-inputs.json',reused);write('summary.json',summaries);write('result.json',result)
    print('MODEL_READY_BIDIRECTIONAL',flush=True)
    schedule=[(TASKS[1],['student-old']),(TASKS[1],['teacher']),
              (TASKS[0],['student']),(TASKS[1],['student']),
              (TASKS[0],['student-force-zero']),(TASKS[1],['student-force-zero']),
              (TASKS[2],['baseline','teacher','student']),(TASKS[3],['baseline','teacher','student'])]
    try:
        for task,labels in schedule:
            for label in labels:
                if label=='student' and fit_result is None:
                    assert tensor_digest(net)==old_net_hash
                    net,train_h,fit_result=fit_balanced(model,env.standing,[pairs[('dev-2','teacher')],pairs[('dev-3','teacher')]],contract,env.keys,OUT,original_act)
                    new_net_hash=tensor_digest(net);result['projector_training']=fit_result;result['projector_updates']=2000;write('result.json',result)
                    write('checkpoint-frozen-before-evaluation.json',dict(sha256=digest(OUT/'contact-projector-final.pt'),tensor_hash=new_net_hash,updates=2000))
                if task['id'].startswith('holdout'):
                    assert fit_result is not None and tensor_digest(net)==new_net_hash
                record=False;mode='baseline';cfg.enabled=True;cfg.activation_time=task['onset_s']
                env.reset(seed=0,trajectory=(anchors[task['id']],[task['direction']*task['distance_m'],0.,0.]))
                if task['id'] in references:
                    ref=references[task['id']]
                    for name in ['rew_buf','push_robot_counter','push_robot_plot_counter','command_counter','average_episode_length','torques','last_episode_length_buf']:
                        value=getattr(env.base,name);value.copy_(torch.as_tensor(ref['base_tensors'][name],device=value.device,dtype=value.dtype))
                    env.base.common_step_counter=ref['common_step_counter']
                    # Original reset leaves warm-start/ctrl and bookkeeping from the prior episode.
                    # qpos/qvel/observations/history already match; restore integration cache before pairing.
                    mujoco.mj_setState(env.sim.model,env.sim.data,np.asarray(ref['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION)
                    mujoco.mj_forward(env.sim.model,env.sim.data)
                    # mj_forward updates warm-start numerically; preserve the saved solver input too.
                    mujoco.mj_setState(env.sim.model,env.sim.data,np.asarray(ref['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION)
                state=snapshot()
                if label=='baseline':write(task['id']+'-initial-state.json',state)
                if task['id'] not in references:references[task['id']]=state;write(task['id']+'-initial-state.json',state)
                if state!=references[task['id']]:
                    write('initial-state-mismatch.json',dict(task=task,label=label,reference=references[task['id']],actual=state))
                    raise AssertionError('Paired initial mechanical/controller/history state differs')
                initial=dict(qpos=env.sim.data.qpos.tolist(),qvel=env.sim.data.qvel.tolist(),wrist=env.initial_point.tolist(),root=env.sim.data.qpos[:3].tolist(),feet_xyz=env.sim.data.xpos[env.feet].tolist())
                mode=label;record=True;control=[];last_control=None;fade=ReleaseFade();decisions=[];sensor_gate=ContactGate();raw_student=torch.zeros_like(env.standing);executed_z=None;delta_action=torch.zeros(1,29,device='cuda')
                for high in range(60):
                    assert time.monotonic()<float(os.environ['CCL_DEADLINE_MONOTONIC'])
                    point=env.last_point.copy();force=env.tether.force.copy();active=bool(env.tether.active);now=float(env.tether.time)
                    sensor_gate.update(now,force)
                    student_diagnostic={}
                    if label.startswith('student'):
                        h=deploy_input(model,env.obs,env.history,env.keys);fed_h=h.clone()
                        if label=='student-force-zero':fed_h[:,-9:]=0.
                        raw_student=net(fed_h).detach()
                        distances=(((h-train_h)/net.scale).square().mean(-1)).sqrt()
                        nearest=int(distances.argmin());student_diagnostic=dict(student_h=h.tolist(),student_fed_h=fed_h.tolist(),nearest_teacher_distance=float(distances.min()),same_time_teacher_distance=float(distances[high]),nearest_teacher_decision=nearest,raw=raw_student.tolist())
                    desired=np.zeros(7);error=env.tether.anchor-point;nominal=original_act(env.obs,env.standing,mean=True)
                    if label=='teacher' and active:
                        point,_=env.tether.point_state();error=env.tether.anchor-point
                        desired=arm_ik(env.tether.jac[:,vadr[arm]],error)
                        q0=qtarget(nominal).cpu().numpy().reshape(-1)
                        desired=np.clip(q0[arm]+desired,qlimits[arm,0],qlimits[arm,1])-q0[arm]
                    correction=fade.update(now,active,desired) if label=='teacher' else np.zeros(7)
                    assert abs(correction).max()<.040001 and np.linalg.norm(correction)<.080001
                    delta_action=torch.zeros_like(nominal);delta_action[:,arm]=torch.tensor(correction,device='cuda',dtype=nominal.dtype)/scale[:,arm]
                    decisions.append(dict(**student_diagnostic,time=env.steps*env.dt,tether_time=now,active=active,error_xyz_m=error.tolist(),force_xyz_n=force.tolist(),
                        correction_rad=correction.tolist(),release_detected_at=fade.release_start,
                        bfm_obs={k:v.tolist() for k,v in env.obs.items()},force_history=env.history.tolist()))
                    _,reward,done,info=env.step(torch.zeros_like(env.standing),baseline=True)
                    if done:break
                record=False
                episode=dict(task=task,label=label,initial=initial,physics=copy.deepcopy(env.trace),control=control,decisions=decisions,
                    failed=bool(env.failed or info['failed']),same_initial_state_sha256=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest())
                name=task['id']+'-'+label+'.json';write(name,episode)
                summary=summarize(episode);summaries.setdefault(task['id'],{})[label]=summary;pairs[(task['id'],label)]=episode
                audit=audit_episode(episode,contract,state);write(task['id']+'-'+label+'-audit.json',audit)
                if label=='teacher' and task['id']=='dev-3':
                    old_teacher=json.loads((TEACHER_SOURCE/(task['id']+'-teacher.json')).read_text())
                    differences={key:float(abs(np.array([r[key] for r in episode['physics']])-np.array([r[key] for r in old_teacher['physics']])).max()) for key in ['qpos','qvel','force','wrist','root','torque']}
                    write('minus-teacher-recapture-parity.json',differences);assert max(differences.values())==0
                before=[r for r in episode['physics'] if r['force_time']<task['onset_s']]
                old_zero=zero[:len(before)]
                differences={key:float(abs(np.array([r[key] for r in before])-np.array([r[key] for r in old_zero])).max()) for key in ['qpos','qvel','force','wrist','root']}
                write(task['id']+'-'+label+'-precontact-audit.json',differences);assert max(differences.values())==0
                if label!='baseline':
                    expected=pairs[(task['id'],'baseline')]['physics'][:len(before)]
                    # Compare dynamics only: historical records have fewer observational fields.
                    assert all(r['qpos']==e['qpos'] and r['force']==e['force'] for r,e in zip(before,expected))
                if episode['failed'] and result['first_physical_failure'] is None:
                    result['first_physical_failure']=dict(file=name,time=next((r['force_time'] for r in env.trace if r['failed']),env.trace[-1]['force_time']))
                result['episodes'].append(dict(file=name,complete=len(env.trace)==1200 and not episode['failed'],physics_samples=len(env.trace),audit=audit))
                write('summary.json',summaries);write('result.json',result)
                print('EPISODE',task['id'],label,'hold',json.dumps(summary['hold']),'release',json.dumps(summary['release']),flush=True)
        assert {n:tensor_digest(getattr(model,n)) for n in INFERENCE_MODULES}==frozen
        assert all(p._version==versions[n] and p.grad is None for n,p in model.named_parameters())
        assert digest(MODEL/'checkpoint/model/model.safetensors')==constants['WEIGHT_SHA']
        assert digest(TRAIN_SOURCE/'contact-projector-final.pt')==STUDENT_SHA
        assert tensor_digest(net)==new_net_hash and all(p.grad is None and not p.requires_grad for p in net.parameters())
        assert all(digest(path)==sha for path,sha in reused.items())
        improvements={}
        for direction in [1,-1]:
            subset=[t['id'] for t in TASKS if t['direction']==direction]
            improvements[str(direction)]=1.-sum(summaries[t]['teacher']['hold']['error_rms_mm'] for t in subset)/sum(summaries[t]['baseline']['hold']['error_rms_mm'] for t in subset)
        result.update(status='completed',all_complete=all(r['complete'] for r in result['episodes']),hold_relative_improvement=improvements,
            frozen_bfm_unchanged=True,checkpoint_unchanged=True,projector_updates=2000,policy_updates=2000,projection_steps=0,ppo_updates=0,total_simulated_seconds=sum(r['physics_samples'] for r in result['episodes'])*.005)
        write('result.json',result);print('COMPLETED',json.dumps(result),flush=True)
    finally:env.close()

if __name__=='__main__':
    try:main()
    except Exception as exc:write('failure.json',dict(error_type=type(exc).__name__,first_boundary=str(exc)));raise
