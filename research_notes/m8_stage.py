"""M8 physical qualification and multi-contact LPP. Frozen native BFM/PD only."""
import copy,json,math,os,hashlib
from pathlib import Path
import numpy as np,torch,mujoco
from locomotion_rollout import UnsafePhysics,snapshot,spherical
from contact_projector import check_deadline
from contact_gate import smooth
from m1_stage import save,cached_point,physical_state_digest
from wholebody_physics import goal_observation
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs';RUN=R/'m8-contact-body-20261101A'
read=lambda p:json.loads(Path(p).read_text())

def reset_common(env,model,orig):
 m=env.sim.model;dev=env.standing.device
 zero=read(R/'issue20-ccl01-20261028T082201Z/attempt-04/outputs/anchor-reference.json')['physics'];initial=read(R/'issue21-ccl02a-20261029T153900Z/attempt-01/outputs/dev-2-initial-state.json');common=read(R/'issue21-direction-20261030T020000Z/attempt-02/outputs/common-state.json')
 model.act=orig;env.qualification_active=False;env.interruption_window=None;env.m1_anchor_hook=None;env.cfg.body='right_wrist_yaw_link';env.cfg.local_point=[0.,0.,0.];env.cfg.enabled=False
 env.reset(seed=0,trajectory=(zero[0]['wrist'],[0,0,0]))
 for name in ['rew_buf','push_robot_counter','push_robot_plot_counter','command_counter','average_episode_length','torques','last_episode_length_buf']:
  v=getattr(env.base,name);v.copy_(torch.as_tensor(initial['base_tensors'][name],device=dev,dtype=v.dtype))
 env.base.common_step_counter=initial['common_step_counter'];mujoco.mj_setState(m,env.sim.data,np.asarray(initial['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,env.sim.data);mujoco.mj_setState(m,env.sim.data,np.asarray(initial['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION)
 for _ in range(30):
  _,_,done,_=env.step(torch.zeros_like(env.standing),baseline=True);assert not done
 parity={k:float(abs(np.asarray([r[k] for r in env.trace])-np.asarray([r[k] for r in zero[:600]])).max()) for k in ['qpos','qvel','wrist','root','force']};assert max(parity.values())==0,parity
 env.base.rew_buf.copy_(torch.as_tensor(common['base_tensors']['rew_buf'],device=dev,dtype=env.base.rew_buf.dtype));assert snapshot(env)==common
 return parity

def advance(env,a):
 env.obs,_,terminated,truncated,_=env.wrapped.step(a,to_numpy=False);env.steps+=1;env.previous_action=a.clone();env.history=np.roll(env.history,-1,axis=0);env.history[-1]=env.tether.force
 return env.failed or bool(torch.as_tensor(terminated).any() or torch.as_tensor(truncated).any())

def qualify(env,model,contract,out):
 plan=read(Path(__file__).with_name('m8_protocol.json'))['qualification'];audit=read(RUN/plan['asset_audit']);orig=model.act;m=env.sim.model;dev=env.standing.device
 scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev);qof=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
 goals=[]
 for i,row in enumerate(audit['selected']):
  z=model.goal_inference(goal_observation(env,np.asarray(row['qpos']),np.zeros(m.nv)))
  goals.append(z);save(out,f'candidate-{i}.json',dict(**row,z=z.tolist(),norm=float(z.norm()),angle_deg=float(torch.rad2deg(torch.acos(((z*env.standing).sum()/256).clamp(-1,1)))),asset_sha256=audit['asset_sha256'],source='Existing natural motion frame, original goal_inference with zero velocity; no live pose reset')))
 result=dict(status='completed',episodes=[],student_optimizer_updates=0)
 def alpha(t):return smooth((t-plan['entry_start'])/plan['entry_duration']) if t<plan['hold_until'] else 1.-smooth((t-plan['hold_until'])/plan['return_duration'])
 for idx in [-1]+list(range(len(goals))):
  label='standing' if idx<0 else f'candidate-{idx}';parity=reset_common(env,model,orig);save(out,label+'-prefix.json',parity);goal=env.standing if idx<0 else goals[idx];initial=copy.deepcopy(env.trace[-1]);cs=[];last=qof(env.previous_action).clone();start=len(env.trace);boundary=None
  # Same-state command continuity check. This is not a dynamic proof.
  qs=[qof(orig(env.obs,spherical(env.standing,goal,alpha(float(t))),mean=True)) for t in np.arange(3,plan['end'],.02)]
  step=max(float(torch.rad2deg((b-a).abs().max())) for a,b in zip(qs,qs[1:]));save(out,label+'-command-screen.json',dict(max_qpd_step_deg=step))
  if step>plan['screen_qpd_step_deg']:boundary=dict(kind='command_screen',qpd_step_deg=step)
  else:
   env.qualification_active=True
   for _ in range(round((plan['end']-3)*50)):
    check_deadline();t=float(env.tether.time);z=spherical(env.standing,goal,alpha(t));a=orig(env.obs,z,mean=True);q=qof(a)
    cs.append(dict(time=t,alpha=alpha(t),executed_z=z.tolist(),q_PD=q[0].tolist(),action=a[0].tolist(),q_PD_step_max_rad=float((q-last).abs().max()),action_saturation_fraction=float(((a*5).abs()>=contract['clip']).float().mean()),bfm_obs={k:v.tolist() for k,v in env.obs.items()},pre_qpos=env.sim.data.qpos.tolist()))
    last=q.clone()
    try:
     if advance(env,a):boundary=dict(kind='native_termination',time=float(env.tether.time));break
    except UnsafePhysics as e:boundary=dict(kind='qualification_physics',physics=e.row);break
  env.qualification_active=False;ep=dict(label=label,initial=initial,physics=copy.deepcopy(env.trace[start:]),control=cs,boundary=boundary,condition=dict(enabled=False,onset=3.8,release=9.,end=17.),goal_index=idx);save(out,label+'.json',ep)
  result['episodes'].append(dict(label=label,complete=boundary is None,boundary=boundary,seconds=len(ep['physics'])/200));save(out,'qualification-summary.json',result);print('M8_QUALIFY',label,result['episodes'][-1]['complete'],flush=True)
 return result

def run_study(env,model,contract,out):
 stage=os.environ['M8_STAGE']
 if stage=='qualification':return qualify(env,model,contract,out)
 raise ValueError('Unimplemented stage '+stage)
