"""Independent actual-force, timing, deployable-input and episode provenance audit."""
import json,gzip
from pathlib import Path
import numpy as np
from m8_analysis import RUN,read

def audit(e,causal_code):
 rows=e['physics'];cs=e['control'];f=np.array([r['force'] for r in rows]);w=np.array([r['applied_contact_wrench'] for r in rows]);active=np.array([r['active'] for r in rows]);t=np.array([r['time'] for r in rows]);errors=[];norm=[];code=[]
 assert np.max(abs(f-w[:,:3]))==0
 assert np.linalg.norm(f,axis=1).max()<=5.000001
 assert np.max(abs(f[~active]),initial=0)==0
 for c in cs:
  if 'student_input' not in c:continue
  h=np.array(c['student_input']);measured=np.array(c['measured_force_xyz']);assert len(h)==1430
  errors.append(float(abs(h[1422:1425]*5-measured).max()))
  z=np.array(c['executed_z']);norm.append(float(abs(np.linalg.norm(z)-16)))
  if causal_code:
   touched=abs(h[465:1425]).max()>1e-8;s=e['condition'];b=s['body']=='torso_link';expected=np.array([float(not b),float(b),*s['local_point']])*touched;code.append(float(abs(h[-5:]-expected).max()))
  prior=np.flatnonzero(t<=c['time']+1e-7)
  if len(prior):assert abs(f[prior[-1]]-measured).max()<1e-6
 if errors:assert max(errors)<2e-6 and max(norm)<1e-4
 if code:assert max(code)<1e-7
 return dict(label=e['label'],physics_frames=len(rows),control_frames=len(cs),boundary=e.get('boundary'),applied_force_exact=True,force_history_error=max(errors,default=0),z_norm_error=max(norm,default=0),causal_location_code_checked=causal_code,location_error=max(code,default=0),support_loss_frames=int(((np.array([r['foot_normal_force_n'] for r in rows])<1).all(1)).sum()),joint_limit_max=max(r['joint_limit_excess_max_rad'] for r in rows),torque_saturation=max(r['torque_saturation'] for r in rows),undesired_contacts=max(r['undesired_ground_contacts'] for r in rows))

def main():
 ledger=read(RUN/'compute-ledger.json');res=[]
 for a in ledger['attempts']:
  for p in sorted((Path(a['path'])/'outputs').glob('*.json.gz')):
   e=read(p)
   if not e.get('physics'):continue
   x=audit(e,a['stage'].startswith('collect') or a['stage']=='final');x['path']=str(p);x['stage']=a['stage'];res.append(x)
 # New native body expert and its legal projected execution should exactly agree.
 def episode(stage,name):
  aa=[a for a in ledger['attempts'] if a['stage']==stage and a.get('result_status')=='completed'];return read(Path(aa[0]['path'])/'outputs'/name)
 projection={}
 for name,stage in [('down_dev','contact'),('down_dev2','expert')]:
  a=episode(stage,name+'__teacher.json.gz');b=episode('expert',name+'__projected.json.gz');diff={k:float(abs(np.array([r[k] for r in a['physics']])-np.array([r[k] for r in b['physics']])).max()) for k in ['qpos','qvel','force','wrist']};assert max(diff.values())==0;projection[name]=diff
 result=dict(episodes=res,body_projected_exact_physical_parity=projection,complete=sum(x['boundary'] is None for x in res),partial=sum(x['boundary'] is not None for x in res),recorded_simulated_seconds=sum(x['physics_frames'] for x in res)/200,actual_simulated_seconds_including_prefix=sum(x['physics_frames'] for x in res)/200+3*len(res),scope='Original-force sensors and frozen BFMs; no real-robot qualification. Initial contact/expert teacher files precede causal location-code preprocessing; training build masks these before learning.')
 (RUN/'analysis/trace-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))
if __name__=='__main__':main()
