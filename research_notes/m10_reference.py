"""Fixed source-level reference ablations, NOT joint control or live joint locking."""
import json,gzip,copy,os
from pathlib import Path
import numpy as np,torch
from wholebody_physics import goal_observation
from m10_stage import P,RUN,OLD
from m8_stage import reset_common
from m1_stage import save

def prepare(env,model,out):
 reset_common(env,model,model.act)
 standing_q=env.sim.data.qpos.copy();m=env.sim.model
 original=json.loads((OLD/'native-cycles.json').read_text())['cycles'];left=np.array(original['left']['q']);straight=np.array(original['straight']['q'])
 # Reference qpos indices follow the checked G1 29D joint order; waist pitch=14.
 assert env.sim.dof_names[14]=='waist_pitch_joint'
 waist_shift=float(straight[:,7+14].mean()-left[:,7+14].mean())
 variants={};evidence=[]
 for variant in ['roundtrip','arm','task']:
  refs=copy.deepcopy(original)
  for name,r in refs.items():
   raw=json.loads((OLD/'tracking-attempt-01/outputs'/(name+'-reference.json')).read_text());q=np.array(r['q']);v=np.array(raw['qvel_world_root_angular'])[r['start_index']:r['end_index']].copy();qnew=q.copy();vnew=v.copy()
   if variant!='roundtrip':qnew[:,22:]=standing_q[22:] # qpos7 + joint15: both free/reference arms
   if variant!='roundtrip':vnew[:,21:]=0. # qvel6 + joint15
   if variant=='task' and name=='straight':qnew[:,21]-=waist_shift
   zs=[];orig_errors=[]
   for i,(qi,vi) in enumerate(zip(qnew,vnew)):
    # Stored reference angular velocity is world-frame; private MuJoCo input is local.
    import mujoco
    d=mujoco.MjData(m);d.qpos[:]=qi;mujoco.mj_forward(m,d);vv=vi.copy();vv[3:6]=d.xmat[1].reshape(3,3).T@vi[3:6]
    z=model.goal_inference(goal_observation(env,qi,vv));zs.append(z[0].tolist())
    if i in [0,len(q)//2,len(q)-1]:
     d.qpos[:]=q[i];mujoco.mj_forward(m,d);ov=v[i].copy();ov[3:6]=d.xmat[1].reshape(3,3).T@v[i,3:6];oz=model.goal_inference(goal_observation(env,q[i],ov));saved=torch.tensor(r['z'][i:i+1],device=z.device)
     orig_errors.append(float(torch.rad2deg(torch.acos(((oz*saved).sum()/256).clamp(-1,1)))))
   r['z']=zs;r['q']=qnew.tolist();evidence.append(dict(variant=variant,reference=name,original_reconstruction_z_angle_deg=orig_errors,waist_shift_deg=float(np.rad2deg(waist_shift)) if variant=='task' and name=='straight' else 0))
  variants[variant]=refs
 body_path=P/'bridge-v2-owned/runs/m8-contact-body-20261101A/qualification-attempt-02/outputs/candidate-1.json';body=json.loads(body_path.read_text());q=np.array(body['qpos']);q[22:]=standing_q[22:]
 neutral=model.goal_inference(goal_observation(env,q,np.zeros(m.nv)))
 result=dict(variants=variants,body_goals=dict(original=body['z'],neutral=neutral.tolist()),evidence=evidence,standing_q=standing_q.tolist(),body_neutral_reference_qpos=q.tolist(),source='Original same-weight goal_inference with native dynamic observations; reference arm pose replacement only. Task variant centers straight reference waist mean at existing left walking mean. No live qpos reset or joint constraint.')
 save(out,'references.json',result);save(out,'reference-evidence.json',dict(evidence=evidence,body_z_angle_deg=float(torch.rad2deg(torch.acos(((neutral*torch.tensor(body['z'],device=neutral.device)).sum()/256).clamp(-1,1))))))
 # Variant file is an asset in the run, never included in Git.
 (RUN/'reference-path.json').write_text(json.dumps({'path':str(out/'references.json')}))

def load_variants(env,model):
 r=json.loads(Path(json.loads((RUN/'reference-path.json').read_text())['path']).read_text());dev=env.standing.device
 for refs in r['variants'].values():
  for v in refs.values():v['tensor']=torch.tensor(v['z'],device=dev)
 return r['variants'],{k:torch.tensor(v,device=dev).reshape(1,256) for k,v in r['body_goals'].items()}
