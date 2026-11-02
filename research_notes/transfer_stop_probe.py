"""Frozen same-prefix physical stop probe; no training or checkpoint selection."""
from pathlib import Path
import sys,os,json,hashlib,time
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');REPO=P/'bridge-v2-owned/trees/transfer-repair/repo';R=P/'bridge-v2-owned/runs/direct-transfer-20261103';sys.path[:0]=[str(REPO/'research_notes'),str(REPO)]
import transfer_student_validated as runner

def study(env,model,contract,out):
 import numpy as np,torch
 import clean_env
 from clean_policy import Expert
 from m11_runtime_speed import install
 from humanoidverse.force_compliance.interaction import spring_force
 from m10_contact import install_partner as original_partner
 runner.write(out/'runtime-speed.json',install(env));torch.set_grad_enabled(False);torch.set_num_threads(1)
 def partner(tether,spec,anchor):
  original_partner(tether,spec,anchor);original=tether.measure
  def measure():
   point,force=original()
   if tether.time<8.:return point,force
   u=np.clip((8.-spec['onset'])/spec['ramp'],0,1);d0=spec['distance']*(3*u*u-2*u*u*u);v0=spec['distance']*6*u*(1-u)/spec['ramp'];dt=np.clip(tether.time-8.,0,.6)
   tether.anchor=anchor+np.array([d0+v0*(dt-dt*dt/1.2),0,0]);tether.partner_velocity=np.array([v0*(1-dt/.6),0,0]);vel=tether.jac@tether.data.qvel;tether.force=spring_force(tether.anchor,point,tether.partner_velocity,vel,tether.cfg.stiffness,tether.cfg.damping,tether.cfg.max_force) if tether.active else np.zeros(3)
   return point,tether.force.copy()
  tether.measure=measure
 clean_env.install_partner=partner
 task=clean_env.Task(env,model,contract,404);plan=json.loads((REPO/'research_notes/transfer_protocol.json').read_text());spec=plan['single'];task.reset();dev=env.standing.device
 def sinput():
  task.prepare();rot=env.sim.data.xmat[1].reshape(3,3);h=task.h.features(rot)[-100:,:3].reshape(-1)
  return torch.cat([task.pro.clamp(-10,10),torch.tensor(h,dtype=torch.float32,device=dev)[None],task.last_command/16,torch.tensor([[1.,0,0,0]],device=dev)],-1)
 selected=json.loads((R/'B1-final-05/job/resume.json').read_text())
 cases=[('teacher_initial',Path(plan['teachers']['initial']['path']),plan['teachers']['initial']['sha256'],False),('teacher_ppo15',Path(plan['teachers']['ppo15']['path']),plan['teachers']['ppo15']['sha256'],False),('teacher_B1',Path(selected['path']),selected['sha256'],False),('student_initial',R/'A-final-04/job/initial-student-validated.pt','8ff5a5435702738b39df91340f8278720e7f850cab97d5c5479bb1e12d5738ec',True),('student_ppo15',R/'A-final-04/job/ppo15-student-validated.pt','0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12',True)]
 rows=[];identities=[]
 for label,p,sha,isstudent in cases+[('m10',cases[0][1],cases[0][2],False)]:
  assert runner.sha(p)==sha;ck=torch.load(p,map_location=dev,weights_only=True);net=Expert(ck['input_dim'],'direct').to(dev);net.load_state_dict(ck['state_dict']);net.eval().requires_grad_(False)
  task.reset(spec,record=True,label=label,query_reference=label=='m10')
  while not task.done:
   x=sinput() if isstudent else task.features();task.step(net.distribution(x).mean,mode='m10_expert' if label=='m10' else 'learned')
  row=task.save(out);row['partner_override']='same original prefix until8s, linear velocity deceleration to zero over.6s, fixed final anchor thereafter; unchanged release14.05/end20.05';rows.append(row);identities.append(dict(label=label,path=str(p),sha256=sha));runner.write(out/'summaries.json',rows);print('STOP_PROBE',label,row['error_cm'],row['force_rms'],row['boundary'],flush=True)
 runner.write(out/'identities.json',identities)
 return dict(status='completed',decisions=0,student_optimizer_updates=0,bfm_updated=False,scope='frozen post-training diagnostic, not a new blind evaluation',driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
runner.study=study
if __name__=='__main__':runner.main()
