"""Read-only aggregate of preserved training and first-divergence traces. No model or physics."""
import json,gzip,hashlib,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
R=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True,parents=True)
def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def first(times,mask,n=1):
 for i in range(len(mask)-n+1):
  if all(mask[i:i+n]):return float(times[i])
 return None
result={'training':{},'traces':{},'limitations':['Rollout tensors/advantages were not persisted per batch. No retrospective task gradient cosines can be reconstructed honestly.','Global phase and family marginals do not give an exact family x phase cross-tab. Completed episode coverage excludes interrupted partial episodes.']}
models={}
for arm,d in [('A','A-ppo-03'),('B','B-ppo-04')]:
 p=R/d/'job';rows=[json.loads(l) for l in (p/'training.jsonl').read_text().splitlines()];families=Counter();phases=Counter();episodes=Counter();eph=defaultdict(list)
 for row in rows:
  families.update(row['coverage']['decisions_by_family']);phases.update(row['coverage']['decisions_by_phase']);episodes.update(row['coverage']['completed_episodes_by_family'])
  for e in row['episodes']:eph[e['condition']['demand_family']].append({k:e[k] for k in ['episode_return','rl_decisions','error_cm','force_rms','boundary']})
 result['training'][arm]={'decisions':sum(families.values()),'by_family':dict(families),'by_phase':dict(phases),'completed_by_family':dict(episodes),'completed_episode_data':dict(eph),'updates':[{'iteration':r['iteration'],'reward':r['reward'],'optimization':r['optimization'],'coverage':r['coverage']} for r in rows]}
 for it in [16,32,48,64]:
  pth=p/f'candidate-{it:04d}.pt';models[f'{arm}{it}']={'path':str(pth),'sha256':hashlib.sha256(pth.read_bytes()).hexdigest()}
result['models']=models
base=read(R/'B-ppo-04/job/dev-0000/stand_walk__mean.json.gz');bp=np.array([r['qpos'] for r in base['physics']]);bc=np.array([r['q_PD'] for r in base['control']]);bz=np.array([r['executed_z'][0] for r in base['control']]);br=np.array([r['raw'] for r in base['decisions']]);ct=np.array([r['time'] for r in base['control']]);pt=np.array([r['time'] for r in base['physics']]);dt=ct[::5] # decisions saved with END time; use actual control start, not that end timestamp
for name in ['I','A16','A32','B16','B32','B48']:
 ep=base if name=='I' else read(R/('A-ppo-03' if name[0]=='A' else 'B-ppo-04')/'job'/f'dev-{int(name[1:]):04d}'/'stand_walk__mean.json.gz')
 p=ep['physics'];c=ep['control'];raw=np.array([r['raw'] for r in ep['decisions']]);z=np.array([r['executed_z'][0] for r in c]);q=np.array([r['q_PD'] for r in c]);qpos=np.array([r['qpos'] for r in p]);vel=np.array([r['qvel'] for r in p]);force=np.array([r['force'] for r in p]);support=np.array([r['foot_normal_force_n'] for r in p]);rootx=qpos[:,0]-qpos[0,0]
 rawdiff=np.sqrt(np.mean((raw-br)**2,axis=1));zdiff=np.degrees(np.arccos(np.clip(np.sum(z*bz,axis=1)/(np.linalg.norm(z,axis=1)*np.linalg.norm(bz,axis=1)),-1,1)));qd=np.sqrt(np.mean(np.degrees(q-bc)**2,axis=1));qdiff=np.max(abs(qpos[:,7:]-bp[:,7:]),axis=1)
 result['traces'][name]={'first_raw_nonzero':first(dt,rawdiff>1e-7),'raw_sustained_300ms':first(dt,rawdiff>1e-5,3),'z_first_0_1deg':first(ct,zdiff>.1),'z_sustained_100ms_0_1deg':first(ct,zdiff>.1,5),'qpd_sustained_100ms_0_1deg':first(ct,qd>.1,5),'actual_joint_sustained_100ms_1deg':first(pt,np.degrees(qdiff)>1,20),'cap_first':first(pt,np.linalg.norm(force,axis=1)>=4.999),'root_x_5cm_first':first(pt,rootx>.05),'unload_one_foot_100ms':first(pt,np.min(support,axis=1)<5,20),'boundary':ep['boundary'],'windows':[]}
 for t in np.arange(4,8.01,.1):
  i=int(np.argmin(abs(pt-t)));j=int(np.argmin(abs(ct-t)));k=min(j//5,len(rawdiff)-1)
  result['traces'][name]['windows'].append(dict(time=float(pt[i]),raw_rms=float(rawdiff[k]),z_deg=float(zdiff[j]),qpd_rms_deg=float(qd[j]),root=qpos[i,:3].tolist(),root_velocity=vel[i,:3].tolist(),force=force[i].tolist(),error_xyz=(np.array(p[i]['anchor'])-p[i]['wrist']).tolist(),support=support[i].tolist(),tilt=p[i]['tilt_deg'],qdiff_max_deg=float(np.degrees(qdiff[i])),qpd_substeps_deg=np.degrees(q[(j//5)*5:(j//5)*5+5]-bc[(j//5)*5:(j//5)*5+5]).tolist()))
(out/'saved-audit.json').write_text(json.dumps(result,indent=2))
(out/'models.json').write_text(json.dumps(models,indent=2))
print(json.dumps({a:{k:v for k,v in x.items() if k not in ['updates','completed_episode_data']} for a,x in result['training'].items()},indent=2))
print(json.dumps({a:{k:v for k,v in x.items() if k!='windows'} for a,x in result['traces'].items()},indent=2))
