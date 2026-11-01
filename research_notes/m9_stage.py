"""Finite native locomotion family qualification; no controller rebuilt."""
import os,json,copy,math
from pathlib import Path
import numpy as np,torch,mujoco
from locomotion_rollout import UnsafePhysics,spherical
from m8_stage import reset_common,advance
from m1_stage import save
from contact_gate import smooth
from contact_projector import check_deadline
from m9_history import ContactHistory
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUN=P/'bridge-v2-owned/runs/m9-contact-locomotion-20261101A';ROOT=Path(__file__).resolve().parents[1]
def completed(stage):
 a=[a for a in json.loads((RUN/'compute-ledger.json').read_text())['attempts'] if a['stage']==stage and a.get('result_status')=='completed'];assert len(a)==1;return Path(a[0]['path'])/'outputs'
def run_study(env,model,contract,out):
 stage=os.environ['M9_STAGE']
 if stage=='speedcheck':
  from m9_speedcheck import run
  return run(env,model,contract,out)
 if stage=='turncheck':
  from m9_turncheck import run
  return run(env,model,contract,out)
 if stage=='diagnostic':
  from m9_composition import run
  return run(env,model,contract,out)
 if stage.startswith('train'):
  from m9_learning import train
  return train(env,model,contract,out,int(stage[-1]))
 if stage.startswith('collect') or stage=='final':
  from m9_contact import run
  import hashlib
  assert hashlib.sha256((RUN/'native-cycles.json').read_bytes()).hexdigest()==json.loads((ROOT/'research_notes/m9_protocol.json').read_text())['native_cycles_sha256']
  refs=json.loads((RUN/'native-cycles.json').read_text())['cycles']
  for r in refs.values():r['tensor']=torch.tensor(r['z'],device=env.standing.device)
  return run(env,model,contract,out,cycles=refs)
 if os.environ['M9_STAGE']=='infer':
  from m9_infer import run
  return run(env,model,contract,out)
 if os.environ['M9_STAGE']=='qualification':return qualify(env,model,contract,out)
 if os.environ['M9_STAGE']=='expert':
  from m9_cyclic import run
  return run(env,model,contract,out)
 if os.environ['M9_STAGE']=='contact':
  from m9_contact import run
  return run(env,model,contract,out)
 if os.environ['M9_STAGE']=='tracking':
  from m9_tracking import run
  return run(env,model,contract,out)
 raise ValueError('stage not implemented')
def episode(env,model,contract,out,label,zfn,end=16.,phases=None,prepare=None):
 orig=model.act;parity=reset_common(env,model,orig);save(out,label+'-prefix.json',parity)
 if prepare:prepare(env,model)
 env.cfg.enabled=False;env.tether.cfg.enabled=False;env.qualification_active=True;initial=copy.deepcopy(env.trace[-1]);start=len(env.trace);cs=[];boundary=None;scale=torch.tensor(contract['q_PD_scale'],device=env.standing.device);offset=torch.tensor(contract['q_PD_offset'],device=env.standing.device);qof=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5;last=qof(env.previous_action);history=ContactHistory()
 try:
  for _ in range(round((end-float(env.tether.time))*50)):
   check_deadline();t=float(env.tether.time);z=zfn(t,env);a=orig(env.obs,z,mean=True);q=qof(a);assert torch.isfinite(a).all() and abs(float(z.norm())-16)<1e-4
   sensor=history.append(env.tether.force,'right_wrist_yaw_link',[0,0,0],env.sim.data.xmat[1].reshape(3,3))
   cs.append(dict(time=t,executed_z=z.tolist(),action=a[0].tolist(),q_PD=q[0].tolist(),q_PD_step_max_rad=float((q-last).abs().max()),pre_qpos=env.sim.data.qpos.tolist(),pre_qvel=env.sim.data.qvel.tolist(),sensor=sensor,bfm_obs={k:v.tolist() for k,v in env.obs.items()},action_saturation_fraction=float(((a*5).abs()>=contract['clip']).float().mean())))
   last=q.clone()
   try:
    if advance(env,a):boundary=dict(kind='native_termination',time=float(env.tether.time));break
   except UnsafePhysics as e:boundary=dict(kind='physics_review',physics=e.row);break
 finally:env.qualification_active=False
 e=dict(label=label,initial=initial,physics=copy.deepcopy(env.trace[start:]),control=cs,boundary=boundary,phases=phases,condition=dict(enabled=False,onset=3,release=12,end=end));save(out,label+'.json',e);print('M9_PHYSICS',label,'complete',boundary is None,flush=True);return dict(label=label,complete=boundary is None,seconds=len(e['physics'])/200,boundary=boundary)
def qualify(env,model,contract,out):
 candidates=json.loads((completed('infer')/'reward-candidates.json').read_text());plan=json.loads((ROOT/'research_notes/m9_protocol.json').read_text())['qualification'];rows=[]
 for name,r in candidates.items():
  goal=torch.tensor(r['z'],device=env.standing.device)
  def zfn(t,e):
   u=smooth((t-3)/plan['entry']) if t<10.5 else 1-smooth((t-10.5)/plan['exit']);return spherical(e.standing,goal,u)
  rows.append(episode(env,model,contract,out,name,zfn,16.,dict(entry=[3,4.5],execute=[4.5,10.5],exit=[10.5,12],recover=[12,16])))
  save(out,'qualification-summary.json',dict(episodes=rows))
 return dict(status='completed',episodes=rows,student_optimizer_updates=0,scope='Candidates assessed independently, actual stop/partial retained; no shared skill capability claim from names')
