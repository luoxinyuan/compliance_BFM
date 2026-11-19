import numpy as np,mujoco as mj,json,gzip,sys,hashlib,time
from pathlib import Path
s=Path(__file__).resolve().parent
old=s.parent/'native-rolling-h100-20261107'
model=old/'repo/humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'
m=mj.MjModel.from_xml_path(str(model));m.opt.timestep=.005
names=['rolling_continue','rolling_stop','I__speed0.15','C32__speed0.15','motion__speed0.15']
def path(n):
 return old/'runs'/('rolling-02' if n.startswith('rolling') else 'qualification-03')/(n+'.json.gz')
def load(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def run(n):
 p=path(n);x=load(p);pref=load(p.with_name(n+'-source-prefix.json.gz'));d=mj.MjData(m)
 mj.mj_setState(m,d,np.array(pref['audit']['integration_state']),mj.mjtState.mjSTATE_INTEGRATION);wid=m.body('right_wrist_yaw_link').id
 assert np.array_equal(d.qpos,x['initial']['qpos'])
 ph=[]
 for i,r in enumerate(x['physics']):
  d.ctrl[-29:]=r['torque'];mj.mj_forward(m,d);d.xfrc_applied[wid]=r['applied_contact_wrench'];mj.mj_step(m,d)
  rows=[];load_n=np.zeros(2)
  for j,c in enumerate(d.contact[:d.ncon]):
   bodies=[m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]]
   if 0 not in bodies:continue
   body=int(max(bodies));b=m.body(body).name
   if 'ankle' not in b and 'foot' not in b:continue
   f=np.zeros(6);mj.mj_contactForce(m,d,j,f);side=int('right' in b);load_n[side]+=abs(f[0])
   rows.append(dict(foot=side,body=body,geom_pair=[int(c.geom1),int(c.geom2)],normal_n=abs(f[0])))
  for k,a in [('qpos',d.qpos),('qvel',d.qvel),('foot_normal_force_n',load_n)]:
   if not np.array_equal(a,r[k]):raise ValueError((n,i,k,float(np.max(abs(a-np.array(r[k]))))))
  ph.append(rows);d.xfrc_applied[wid]=0;mj.mj_forward(m,d);mj.mj_forward(m,d)
  if i%4==3:mj.mj_forward(m,d)
 out=s/'low01';out.mkdir(exist_ok=True)
 with gzip.open(out/(n+'.json.gz'),'wt') as f:json.dump(ph,f)
 print(n,len(ph),'exact',flush=True)
if __name__=='__main__':
 for n in names:run(n)
