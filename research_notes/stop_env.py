"""Real partner braking; input-only force stress. Original BFM/PD/reward unchanged."""
from pathlib import Path
import json,gzip,copy
import numpy as np
PLAN=json.loads(Path(__file__).with_name('stop_protocol.json').read_text())
def braking(spec,t):
 u=np.clip((spec['stop_time']-spec['onset'])/spec['ramp'],0.,1.)
 d0=spec['distance']*(3*u*u-2*u*u*u);v0=spec['distance']*6*u*(1-u)/spec['ramp'];duration=spec['deceleration'];dt=np.clip(t-spec['stop_time'],0.,duration)
 return d0+v0*(dt-dt*dt/(2*duration)),v0*(1-dt/duration)
def phase(spec,t):
 if t<spec['onset']:return 'pre'
 if t>=spec['release']:return 'recover'
 if 'stop_time' in spec:
  return 'move' if t<spec['stop_time'] else 'decelerate' if t<spec['stop_time']+spec['deceleration'] else 'hold'
 return 'move' if t<spec['onset']+spec['ramp'] else 'hold'
def install():
 import clean_env
 if getattr(clean_env,'STOP_INSTALLED',False):return
 from m10_contact import install_partner as original_partner
 from humanoidverse.force_compliance.interaction import spring_force
 from m11_history import FastHistory
 original_task=clean_env.Task
 def partner(tether,spec,anchor):
  original_partner(tether,spec,anchor)
  if 'stop_time' not in spec:return
  original=tether.measure
  def measure():
   point,force=original()
   if tether.time<spec['stop_time']:return point,force
   d,v=braking(spec,tether.time);tether.anchor=anchor+np.array([d,0,0]);tether.partner_velocity=np.array([v,0,0])
   velocity=tether.jac@tether.data.qvel
   tether.force=spring_force(tether.anchor,point,tether.partner_velocity,velocity,tether.cfg.stiffness,tether.cfg.damping,tether.cfg.max_force) if tether.active else np.zeros(3)
   return point,tether.force.copy()
  tether.measure=measure
 class StopTask(original_task):
  def reset(self,*args,**kwargs):
   self.sensor=FastHistory();self.sensor_queue=[];self.sensor_last=np.zeros(3);self.sensor_rng=np.random.default_rng(getattr(self,'sensor_seed',707));self.sensor_log=[]
   return super().reset(*args,**kwargs)
  def prepare(self):
   prepared=self.prepared;result=super().prepare()
   if not prepared:
    mode=getattr(self,'sensor_mode','ideal');cfg=PLAN['sensing'][mode];f=self.e.tether.force.copy();R=self.e.sim.data.xmat[1].reshape(3,3)
    self.sensor_queue.append(f.copy());delay=cfg.get('delay_samples',0);past=self.sensor_queue[-1-delay] if len(self.sensor_queue)>delay else np.zeros(3)
    if mode=='ideal':measured=f
    else:
     # Delay/drop/noise in sensor (body) coordinates; retain world history with measured orientation.
     if self.sensor_rng.random()>=cfg['drop_probability']:
      self.sensor_last=R@(R.T@past+np.array(cfg['bias'])+self.sensor_rng.normal(0,cfg['noise_std'],3))
     measured=self.sensor_last.copy()
    self.sensor.append(measured,'right_wrist_yaw_link',[0,0,0],R)
    if self.record:self.sensor_log.append(dict(time=float(self.e.tether.time),true=f.tolist(),measured=measured.tolist()))
   return result
  def sample_spec(self):
   spec=super().sample_spec();cfg=PLAN['teacher_stop_training']
   if self.rng.random()<cfg['probability']:
    t=float(self.rng.uniform(*cfg['stop_time']));d=float(self.rng.uniform(*cfg['deceleration']));hold=float(self.rng.uniform(*cfg['hold']));spec.update(stop_time=t,deceleration=d,release=t+d+hold,end=t+d+hold+6)
   return spec
  def save(self,out):
   result=super().save(out);p=out/(self.label+'.json.gz')
   with gzip.open(p,'rt') as f:data=json.load(f)
   s=self.spec
   if 'stop_time' in s:data['phases']={'pre':[3,s['onset']],'move':[s['onset'],s['stop_time']],'decelerate':[s['stop_time'],s['stop_time']+s['deceleration']],'hold':[s['stop_time']+s['deceleration'],s['release']],'recover':[s['release'],s['end']]}
   data['sensor_mode']=getattr(self,'sensor_mode','ideal');data['sensor_log']=self.sensor_log
   with gzip.open(p,'wt') as f:json.dump(data,f)
   result['phases']={}
   for name,(a,b) in data['phases'].items():
    rows=[r for r in self.physics if a<=r['time']<b]
    if not rows:continue
    E=np.array([np.array(r['anchor'])-r['wrist'] for r in rows]);F=np.array([r['force'] for r in rows]);root=np.array([r['root'] for r in rows])
    result['phases'][name]=dict(error_cm=float(100*np.sqrt((E*E).sum(1).mean())),signed_xyz_cm=(100*E.mean(0)).tolist(),force_rms=float(np.sqrt((F*F).sum(1).mean())),force_xyz_rms=np.sqrt((F*F).mean(0)).tolist(),cap_fraction=float((np.linalg.norm(F,axis=1)>=4.999).mean()),root_displacement=(root[-1]-root[0]).tolist())
   return result
 clean_env.install_partner=partner;clean_env.Task=StopTask;clean_env.STOP_INSTALLED=True
