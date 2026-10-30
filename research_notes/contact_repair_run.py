"""Targeted supervised repair: staged feasibility audit before any Student update.
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
    from contact_projector import Projector
    from targeted_repair import precheck,fit_repair,sha
    protocol=json.loads((ROOT/'research_notes/targeted_repair_protocol.json').read_text());write('protocol.json',protocol)
    train_h=torch.load(SOURCE/'teacher-dataset.pt',map_location='cuda',weights_only=True)['h']
    net=Projector(train_h).to('cuda');net.load_state_dict(torch.load(SOURCE/'contact-projector-final.pt',map_location='cuda',weights_only=True)['state_dict'],strict=True)
    net.eval().requires_grad_(False);old_student_hash=tensor_digest(net)
    stage=os.environ['REPAIR_STAGE'];result=dict(status='running',stage=stage,model_loaded=True,original_wrapper_used=True,projector_updates=0,ppo_updates=0)
    write('result.json',result)
    try:
        if stage=='precheck':
            result['precheck']=precheck(env,model,env.standing,contract,net,OUT)
            assert tensor_digest(net)==old_student_hash
            result['status']='prechecked'
        else:
            raise RuntimeError('Repair entry is prepared only after scientific gate review')
        assert {n:tensor_digest(getattr(model,n)) for n in INFERENCE_MODULES}==frozen
        assert all(p._version==versions[n] and p.grad is None for n,p in model.named_parameters())
        assert digest(SOURCE/'contact-projector-final.pt')==STUDENT_SHA
        assert digest(MODEL/'checkpoint/model/model.safetensors')==constants['WEIGHT_SHA']
        result.update(original_bfm_unchanged=True,original_student_unchanged=True)
        write('result.json',result);print('COMPLETED',json.dumps(result),flush=True)
    finally:env.close()

if __name__=='__main__':
    try:main()
    except Exception as exc:write('failure.json',dict(error_type=type(exc).__name__,first_boundary=str(exc)));raise
