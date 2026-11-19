import os,sys,gzip,json,itertools
from pathlib import Path
import numpy as np
import mujoco
here=Path(__file__).resolve().parent
sys.path.append(str(here.parent/'ctft_quiet_01'))
from score_endpoint import material,stance_segments
def monitor(env):
 step=env._original_step;m=env.sim.model;d=env.sim.data;enabled=[False]
 def wrap():
  step()
  if not enabled[0]:return
  cs=[]
  for i,c in enumerate(d.contact[:d.ncon]):
   ids=[int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])]
   if 0 not in ids:continue
   b=max(ids);name=m.body(b).name
   if 'ankle' not in name and 'foot' not in name:continue
   f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
   cs.append(dict(body=b,foot=int('right' in name),geom_pair=[int(c.geom1),int(c.geom2)],normal_n=abs(float(f[0]))))
  env.contact_sample['ctft_material_contacts']=cs
 env._original_step=wrap
 return enabled
def metrics(task,env):
 ph=task.physics;cs=task.controls
 m=env.sim.model;mg=m.body_mass.sum()*np.linalg.norm(m.opt.gravity)
 def ar(k):return np.array([r[k] for r in ph])
 load=ar('foot_normal_force_n');q0=task.initial['qpos'];t=ar('force_time').round(5);n0=np.flatnonzero(t>=5)[0]
 foot=material(m,ph,q0,stance_segments(load,mg));f=ar('force');err=np.linalg.norm(ar('wrist')-ar('anchor'),axis=1);w=ar('wrist')[:,0];sh=ar('shoulder_xyz')[:,0];root=ar('root')[:,:2];yaw=np.unwrap(np.arctan2(ar('pelvis_matrix')[:,1,0],ar('pelvis_matrix')[:,0,0]))*180/np.pi
 velocity=np.array([r['post_velocity'] for r in cs]);ct=np.round([r['time'] for r in cs],5)
 def rms(x):return float(np.mean(x*x)**.5)
 def stats(lo,hi):
  mask=(t>=lo)&(t<hi);first,last=np.flatnonzero(mask)[[0,-1]];pre=max(first-1,0);sel=(ct>=lo)&(ct<hi)
  return dict(force=rms(np.linalg.norm(f[mask],axis=1)),err=100*rms(err[mask]) if 5<=lo<11.8 else None,v95=float(np.percentile(np.linalg.norm(velocity[sel,:2],axis=1),95)),lateral_v95=float(np.percentile(abs(velocity[sel,1]),95)),root_cm=100*(root[last,0]-root[pre,0]),wrist_cm=100*(w[last]-w[pre]),root_path_cm=100*np.linalg.norm(np.diff(root[pre:last+1],axis=0),axis=1).sum(),yaw_change=yaw[last]-yaw[pre],wrist_lead_cm=100*np.max((ar('wrist')-ar('anchor'))[mask,0]) if 5<=lo<11.8 else None)
 single=(load[:,0]>=.05*mg).astype(int)-(load[:,1]>=.05*mg).astype(int);ix=n0;bouts=[]
 for side,g in itertools.groupby(single[n0:]):
  ln=sum(1 for _ in g)
  if side and ln>=20 and t[ix]<11.8:bouts.append((side,ix))
  ix+=ln
 swap=next((t[k+19]-5 for i,(side,k) in enumerate(bouts) if any(o!=side for o,_ in bouts[:i]) and root[k+19,0]-root[n0,0]>=.02),None)
 recoil={}
 if bouts:
  # Positions are post-physics; archive time is 5ms after applied-force timestamp.
  sl=slice(n0-1,bouts[0][1]+1)
  for label,y in [('world',w),('relative',w-sh)]:
   a=y[sl];recoil[label]=100*float((np.maximum.accumulate(a)-a).max())
 return dict(early=stats(5,5.5),move=stats(5,9.8),hold=stats(9.8,11.8),release=stats(11.8,15.8),contact=stats(5,11.8),recoil=recoil,first_single=t[bouts[0][1]]-5 if bouts else None,exchange=swap,stances=foot,slip_pass=bool(foot) and all(r['measured'] and r['p95']<=.05 and r['path']<=.02 for r in foot),hard=task.boundary)
