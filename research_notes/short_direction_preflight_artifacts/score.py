"""Score paired executed trajectories only; do not change an actor or simulate new actions."""
import json,gzip,csv,pickle,sys
from pathlib import Path
import numpy as np
import mujoco
from near_paths import names,path
s=Path(__file__).resolve().parent
e=s.parent
sys.path.insert(0,str(e/'bfm_ctft_v1/runtime-v1a/research/bfm_ctft_v1'))
from reward import ForceEMA,contact_cost
model=mujoco.MjModel.from_xml_path(str(e/'native-rolling-h100-20261107/repo/humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'))
data=mujoco.MjData(model);wid=model.body('right_wrist_yaw_link').id
def read(p):
 with gzip.open(p) as f:return json.load(f)
def native_parts(x):
 ema=ForceEMA();result=[]
 for c,r in zip(x['controls'],x['physics'][3::4],strict=True):
  data.qpos[:]=r['qpos'];data.qvel[:]=r['qvel'];mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
  jac=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jac,None,data.xpos[wid],wid)
  f=np.array(r['force']);v=jac@data.qvel
  co=contact_cost(filtered_force_n=ema.update(f),applied_force_n=f,wrist_velocity_mps=v,root_velocity_mps=data.qvel[:3],guidance_mode=1)
  orig=c['reward_components'];assert abs(orig['contact_aux']-co['contact_native_reward'])<1e-7,('exact component mismatch',c['time'],orig,co)
  result.append(dict(contact=orig['contact_aux'],native={'total':orig['native_aux']},parts=co['parts']))
 return result
def score(n,grp,x,a,t0,hold,rel,end):
 ph=x['physics'];arr=lambda k:np.asarray([r[k] for r in ph]);t=arr('force_time').reshape(-1,4).round(6)
 assert len(a)==len(t)
 dx=arr('wrist')-arr('anchor');err=np.linalg.norm(dx,axis=1);force=np.linalg.norm(arr('force'),axis=1)
 pos=((err/.10)**2*arr('active')).reshape(-1,4).mean(1)
 old=np.array([r['contact']+sum(r['native'].values()) for r in a])
 response=np.array([r['parts']['response'] for r in a])
 new=old+response-pos
 out=[]
 for label,lo,hi in [('onset',t0,rel),('move',t0,hold),('hold',hold,rel),('released',rel,end)]:
  use=(t[:,0]>=round(lo,6))&(t[:,-1]<round(hi,6))
  if label=='released':use &= ~arr('active').reshape(-1,4).any(1)
  if not use.any():continue
  i=np.repeat(use,4);weights=.98**np.arange(use.sum());w=arr('wrist')[i,0];sh=arr('shoulder_xyz')[i,0];root=arr('root')[i,0]
  out.append(dict(group=grp,controller=n,window=label,n=len(weights),old=weights@old[use],new=weights@new[use],error_cm=100*np.mean(err[i]**2)**.5 if label!='released' else None,force_n=np.mean(force[i]**2)**.5,overrun_cm=100*max(0,dx[i,0].max()) if label!='released' else None,root_dx_cm=100*(root[-1]-root[0]),wrist_shoulder_dx_cm=100*((w-sh)[-1]-(w-sh)[0])))
 return out
def write(name,rows):
 with (s/name).open('x') as f:
  wr=csv.DictWriter(f,rows[0]);wr.writeheader();wr.writerows(rows)
if sys.argv[1]=='short':
 rows=[];ref=None
 for n in names:
  x=read(path(n));c=x['condition'];a=read(s/'aux01'/(n+'.json.gz'))
  ident=(x['initial']['qpos'],x['initial']['qvel'],{k:v for k,v in c.items() if k!='handoff_time'})
  if ref is None:ref=ident
  else:assert ref==ident,n
  rows+=score(n,'train_short08',x,a,c['onset'],c['onset']+c['ramp'],c['release'],c['end'])
 write('short.csv',rows)
else:
 rows=[]
 for p in sorted((e/'ctft_start_pair_01/paired').glob('*-stop-*.gz')):
  with gzip.open(p) as f:x=pickle.load(f)
  sp=x['spec']['partner'];t=x['start'];hold=t+sp['distance_m']/sp['plateau_speed_mps']+sp['accel_s'];rel=hold+sp['hold_s']
  rows+=score(x['model'],x['spec']['parent'],x,native_parts(x),t,hold,rel,rel+4)
 write('stop.csv',rows)
