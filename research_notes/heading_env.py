"""Continuous real partner curvature change; no task clock enters Direct policy."""
import math,numpy as np

def path(s,k):
 return np.array([s,0.,0.]) if abs(k)<1e-12 else np.array([math.sin(k*s)/k,(1-math.cos(k*s))/k,0.])

def pose(spec,t):
 u=np.clip((t-spec['onset'])/spec['ramp'],0.,1.);s=spec['distance']*(3*u*u-2*u*u*u);v=spec['distance']*6*u*(1-u)/spec['ramp'] if 0<u<1 else 0.;k=spec['curvature']
 if 'change_time' not in spec or t<=spec['change_time']:return path(s,k),v*np.array([math.cos(k*s),math.sin(k*s),0.])
 uc=np.clip((spec['change_time']-spec['onset'])/spec['ramp'],0.,1.);sc=spec['distance']*(3*uc*uc-2*uc*uc*uc);theta=k*sc;c,sn=math.cos(theta),math.sin(theta);R=np.array([[c,-sn,0],[sn,c,0],[0,0,1]]);kk=spec['after_curvature'];d=s-sc
 return path(sc,k)+R@path(d,kk),v*np.array([math.cos(theta+kk*d),math.sin(theta+kk*d),0.])

def install():
 import clean_env,json
 from pathlib import Path
 if getattr(clean_env,'HEADING_INSTALLED',False):return
 old_partner=clean_env.install_partner;Original=clean_env.Task
 def partner(tether,spec,anchor):
  old_partner(tether,spec,anchor)
  if 'change_time' not in spec:return
  original=tether.measure
  def measure():
   point,force=original();d,v=pose(spec,tether.time);tether.anchor=anchor+d;tether.partner_velocity=v
   from humanoidverse.force_compliance.interaction import spring_force
   velocity=tether.jac@tether.data.qvel
   tether.force=spring_force(tether.anchor,point,v,velocity,tether.cfg.stiffness,tether.cfg.damping,tether.cfg.max_force) if tether.active else np.zeros(3)
   return point,tether.force.copy()
  tether.measure=measure
 class HeadingTask(Original):
  def save(self,out):
   result=super().save(out)
   moving=[r for r in self.physics if r['active'] and np.linalg.norm(r['anchor_velocity'][:2])>.02]
   e=[math.atan2(math.sin(math.atan2(np.array(r['pelvis_matrix'])[1,0],np.array(r['pelvis_matrix'])[0,0])-math.atan2(r['anchor_velocity'][1],r['anchor_velocity'][0])),math.cos(math.atan2(np.array(r['pelvis_matrix'])[1,0],np.array(r['pelvis_matrix'])[0,0])-math.atan2(r['anchor_velocity'][1],r['anchor_velocity'][0]))) for r in moving]
   result['moving_heading_rms_deg']=float(np.degrees(np.sqrt(np.mean(np.square(e))))) if e else None
   return result
  def sample_spec(self):
   plan=json.loads(Path(__file__).with_name('heading_protocol.json').read_text());spec=dict(list(plan['heading_sources'].values())[int(self.rng.integers(len(plan['heading_sources'])))])
   import os
   if os.environ.get('NARROW_DOMAIN')=='neighborhood':
    spec=dict(plan['single']);spec.update(curvature=float(self.rng.uniform(-.23,.3)),distance=float(self.rng.uniform(.78,.84)),ramp=float(self.rng.uniform(7.7,8.1)),onset=float(self.rng.uniform(4.15,4.4)))
    spec['release']=spec['onset']+spec['ramp']+2;spec['end']=spec['release']+6
    if self.rng.random()<.35:spec.update(after_curvature=spec['curvature'],curvature=0.,change_time=float(self.rng.uniform(7.8,8.2)))
   return spec
 clean_env.Task=HeadingTask;clean_env.install_partner=partner;clean_env.HEADING_INSTALLED=True
