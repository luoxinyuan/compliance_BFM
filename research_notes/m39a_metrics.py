"""Physical no-demand and recovery metrics, never released-anchor task error."""
import math,numpy as np
from geometry_env import wrap
def extra(task):
 rr=task.physics;initial=task.initial
 def yaw(r,key):
  v=np.array(r[key]);return math.atan2(v[1,0],v[0,0])
 def angle(r,key):return math.degrees(wrap(yaw(r,key)-yaw(initial,key)))
 recovery=[r for r in rr if r['force_time']>=task.spec['release']]
 return dict(torso_excursion_deg=max(abs(angle(r,'torso_matrix')) for r in rr),pelvis_excursion_deg=max(abs(angle(r,'pelvis_matrix')) for r in rr),final_torso_deg=angle(rr[-1],'torso_matrix'),final_pelvis_deg=angle(rr[-1],'pelvis_matrix'),recovery_root_displacement_cm=float(100*np.linalg.norm(np.array(recovery[-1]['root'])-recovery[0]['root'])) if recovery else 0.,qpd_max_deg=max(c['q_PD_step_max_rad'] for c in task.controls)*180/math.pi)
def cost(r):
 if not r['condition']['enabled']:return .5*r['root_max_displacement']+.02*r.get('torso_excursion_deg',0)+20*bool(r['boundary'])
 return r.get('sum_force_norm_rms',r['force_rms'])+3*r['error_cm']/100+r['cap_fraction']+.04*(r['torso_heading_rms_deg'] if r['condition'].get('geometry')=='torso_pair' else r['moving_heading_rms_deg'])+20*bool(r['boundary'])
