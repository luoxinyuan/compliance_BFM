"""Replay expert command-memory math from real recorded Student states. No BFM inference."""
import json,gzip,math
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
from m9c_stage import RUN,OLD,completed
from m9_cyclic import CyclicFeedback
from m9c_rl import command
class Sphere:
 def project_z(self,z):return 16*torch.nn.functional.normalize(z,dim=-1)
def main():
 torch.set_num_threads(1);ck=torch.load(completed('rl')/'expert-final.pt',map_location='cpu',weights_only=True);basis=ck['basis'];refs=json.loads((OLD/'native-cycles.json').read_text())['cycles']
 for r in refs.values():r['tensor']=torch.tensor(r['z'])
 from m9b_physics import STANDING
 standing=torch.tensor(json.loads(STANDING.read_text()),dtype=torch.float32).reshape(1,256);result=[]
 for stage in ['collect','final']:
  for p in completed(stage).glob('*.json.gz'):
   e=json.loads(gzip.decompress(p.read_bytes()));cs=e['control']
   if not cs or cs[0].get('learned_query_raw') is None or e['condition'].get('family')=='old':continue
   fb=CyclicFeedback({k:r['period'] for k,r in refs.items()},True);rate=1.;maxz=0.;maxstate=0.;held=0.;prior=None
   for tick,c in enumerate(cs):
    R=Rotation.from_quat(np.array(c['pre_qpos'])[[4,5,6,3]]).as_matrix();before=fb.phase;s=fb.step(c['measured_force_xyz'],R,guide=np.array(c['expert_query_guide_world']));fb.phase=before+(fb.phase-before)*rate;s['phase']=fb.phase;maxstate=max(maxstate,max(abs(s[k]-c['oracle_controller_state'][k]) for k in ['alpha','turn','phase']))
    raw=torch.tensor(c['learned_query_raw']);z,rate=command(Sphere(),standing,refs,s,raw,basis);maxz=max(maxz,float(abs(z-torch.tensor(c['oracle_z'])).max()))
    if tick%5!=0:held=max(held,float(abs(raw-prior).max()))
    prior=raw
   assert maxstate<1e-5 and maxz<1e-4 and held==0,(p,maxstate,maxz,held);result.append(dict(path=str(p),controls=len(cs),state_replay_max=maxstate,legal_z_replay_max=maxz,within_block_raw_change_max=held))
 (RUN/'query-audit.json').write_text(json.dumps(dict(no_BFM_forward=True,scope='causal alpha/phase/turn and held own-command replay from actual recorded force, orientation and current guide; not expert optimality',episodes=result),indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
