"""Same-weight native tracking candidates from preselected natural motion clips.
Full reference is expert-only. These qualification rollouts are not contact policies.
"""
import json
from pathlib import Path
import numpy as np,torch
from locomotion_rollout import spherical
from contact_gate import smooth
from m9_stage import RUN,episode
from m1_stage import save
from contact_projector import check_deadline

def encode(env,model,item):
 from humanoidverse.utils.helpers import get_backward_observation
 lib=env.base._motion_lib;old_ids=lib._curr_motion_ids.cpu().tolist();assert len(old_ids)==1
 idx=lib._motion_data_keys.tolist().index(item['motion'])
 try:
  lib.load_motions(random_sample=False,start_idx=idx,num_motions_to_load=1)
  obs,ref=get_backward_observation(env.base,0,use_root_height_obs=bool(env.base.config.obs.get('root_height_obs',True)))
  first=round(item['start']/item['fps']/.02);indices=torch.arange(first+1,first+302,device=env.standing.device)
  batch={k:v[indices] for k,v in obs.items()};z=model.project_z(model.backward_map(batch))
  q=torch.cat([ref['ref_body_pos'][indices,0],ref['ref_body_rots'][indices,0][:,[3,0,1,2]],ref['dof_pos'][indices]],-1).cpu().numpy()
  v=torch.cat([ref['ref_body_vels'][indices,0],ref['ref_body_angular_vels'][indices,0],ref['ref_dof_vel'][indices]],-1).cpu().numpy()
  # Saved qvel here is explicitly world root angular velocity, not MuJoCo local.
 finally:lib.load_motions(random_sample=False,start_idx=old_ids[0],num_motions_to_load=1)
 assert torch.isfinite(z).all() and (z.norm(dim=-1)-16).abs().max()<1e-4
 return z,q,v
def run(env,model,contract,out):
 inv=json.loads((RUN/'motion-inventory.json').read_text());rows=[]
 for family,item in inv['selected'].items():
  if item is None:continue
  check_deadline();z,q,v=encode(env,model,item);save(out,f'{family}-reference.json',dict(source=item,z=z.tolist(),qpos=q.tolist(),qvel_world_root_angular=v.tolist(),scope='Expert/reference only; B(next reference obs),50Hz; same model. Not deployable timed contact policy.'))
  def cmd(t,e):
   if t<4.5:return spherical(e.standing,z[0:1],smooth((t-3)/1.5))
   if t<10.5:
    idx=min(round((t-4.5)/.02),len(z)-1);return z[idx:idx+1]
   return spherical(z[-1:],e.standing,smooth((t-10.5)/1.5))
  rows.append(episode(env,model,contract,out,'tracking-'+family,cmd,16.,dict(entry=[3,4.5],execute=[4.5,10.5],exit=[10.5,12],recover=[12,16])))
  save(out,'tracking-summary.json',dict(episodes=rows))
 return dict(status='completed',episodes=rows,student_optimizer_updates=0,scope='Original BFM feedback under native motion sequence; standing entry tested, no claim natural reference reset equals standing entry')
