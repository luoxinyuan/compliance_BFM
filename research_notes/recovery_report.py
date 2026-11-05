"""Aggregate saved physics; kinematic reconstruction only, no new dynamics."""
import json,gzip,time,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'research_notes'))
from prior_metrics import summarize
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/prior-recovery-20261105');O=R.parent/'latent-prior-audit-20261105';start=time.monotonic()
def write(p,x):p.write_text(json.dumps(x,indent=2))
rows=[]
for d in ['physical-03','fit-C0','fit-C1','final-C','final-reference']:
 for p in sorted((R/d).glob('*.json.gz')):
  if any(k in p.name for k in ['labels','queries','prefix']):continue
  r=summarize(p)
  if r:rows.append(r)
write(R/'physical-metrics.json',rows)
summ=json.loads((R/'physical-03/summaries.json').read_text());parity={r['label']:r['intervention'] for r in summ if r.get('intervention')};assert all(all(r['section_equal'].values()) for r in parity.values())
write(R/'prefix-parity.json',parity)
queries=json.loads((R/'physical-03/same-state.json').read_text());small=[]
for prefix in ['I','A','B']:
 for t in [4,4.3,4.5,5,5.5,6,6.5,7]:
  rr=[v for v in queries if v['prefix']==prefix and v['model']=='B'];v=min(rr,key=lambda v:abs(v['time']-t));small.append({k:v[k] for k in ['prefix','time','z_deg','raw_rms','raw_norm','qpd_substep_rms_deg']})
write(R/'same-state-compact.json',small)
import mujoco
m=mujoco.MjModel.from_xml_path(str(ROOT/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'));data=mujoco.MjData(m);names=['right_elbow_link','right_shoulder_pitch_link','torso_link','pelvis'];ids={k:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,k) for k in names};assert min(ids.values())>=0;timeline={}
for name in ['I__walk','A__walk','B__walk','B_to_I_5.5','B_to_I_7.0','I_to_B_5.5','I__small','B__small']:
 e=json.load(gzip.open(R/'physical-03'/(name+'.json.gz'),'rt'));rr=[]
 for r in e['physics'][::20]:
  data.qpos[:]=r['qpos'];data.qvel[:]=r['qvel'];mujoco.mj_forward(m,data)
  rr.append(dict(time=r['time'],force_time=r['force_time'],wrist=r['wrist'],shoulder=r['shoulder_xyz'],elbow=data.xpos[ids['right_elbow_link']].tolist(),torso=data.xpos[ids['torso_link']].tolist(),root=r['root'],q_right_arm=r['qpos'][29:36],dq_right_arm=r['qvel'][28:35],feet=r['feet_xyz'],support=r['support'],force=r['force'],anchor=r['anchor']))
 timeline[name]=rr
write(R/'body-timeline.json',timeline)
# Full actor displacement, not only nominal update counts.
import torch
torch.set_num_threads(1);cfg=json.loads((ROOT/'research_notes/recovery_protocol.json').read_text());states={k:torch.load(v['path'],map_location='cpu',weights_only=True)['state_dict'] for k,v in cfg['models'].items()};dd={}
for k in ['A','B']:
 vec=torch.cat([(v-states['I'][n]).flatten() for n,v in states[k].items() if n.startswith('actor.')]);dd[k]=dict(parameter_delta_l2=float(vec.norm()),parameter_delta_rms=float(vec.square().mean().sqrt()),max=float(vec.abs().max()))
write(R/'parameter-update.json',dd);write(R/'analysis-cost.json',dict(elapsed_s=time.monotonic()-start,physics_steps=0,scope='saved trajectory metrics and mj_forward kinematic reconstruction, not physical rollouts'))
print(json.dumps(dict(metrics=len(rows),verified_prefixes=len(parity),parameter_delta=dd,seconds=time.monotonic()-start)))
