"""Read saved physical records only, no model execution."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
from contact_gate_summary import summarize

def analyze(out):
 contract=json.loads((out/'action-contract.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
 scale=np.asarray(contract['q_PD_scale']);offset=np.asarray(contract['q_PD_offset'])
 results={};audits={};raw={}
 for ctl in protocol['conditions'][0]['controllers']:
  label='up30__'+ctl;p=out/(label+'.json')
  if not p.exists():continue
  e=json.loads(p.read_text());raw[ctl]=e
  if e['boundary'] is not None:
   results[label]={'boundary':e['boundary'],'physics_samples':len(e['physics'])};continue
  results[label]=summarize(e,{})
  cs=e['control'];rs=e['physics'];a=np.asarray([c['action'] for c in cs]);q=np.asarray([c['q_PD'] for c in cs]);f=np.asarray([r['force'] for r in rs]);w=np.asarray([r['applied_wrist_wrench'][:3] for r in rs])
  observed=np.asarray([c['measured_force_xyz'] for c in cs]);expected=np.vstack([np.zeros(3),f[3:-1:4]])
  z=np.asarray([c['executed_z'][0] for c in cs])
  audit=dict(raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),common_state_sha256=e['common_state_sha256'],control_count=len(cs),physics_count=len(rs),
    qPD_recompute_max_rad=float(abs(q-(offset+np.clip(a*5,-contract['clip'],contract['clip'])*scale/5)).max()),
    applied_force_max_abs=float(abs(f-w).max()),sensor_delay_max_abs=float(abs(observed-expected).max()),
    latent_norm_max_error=float(abs(np.linalg.norm(z,axis=1)-16).max()),latent_angle_max_deg=max(c['z_angle_deg'] for c in cs),latent_step_max_deg=max(c['z_step_deg'] for c in cs))
  assert audit['qPD_recompute_max_rad']<1e-6 and audit['applied_force_max_abs']<1e-10 and audit['sensor_delay_max_abs']<1e-10 and audit['latent_norm_max_error']<1e-4
  # Force terminal release must not count as low-error tracking.
  assert results[label]['phases']['recovery']['tracking_rms_cm'] is None
  audits[label]=audit
  # At each phase report all axes and movement, including vertical overshoot/negative force.
  for phase,span in [('move',(3.8,7.8)),('hold',(7.8,8.8)),('all_contact',(3.8,8.8))]:
   rows=[r for r in rs if r['active'] and (phase=='all_contact' or span[0]-1e-8<=r['force_time']<span[1]-1e-8)]
   fz=np.asarray([r['force'][2] for r in rows]);ph=results[label]['phases'][phase]
   ph['negative_force_z_fraction']=float((fz<-.1).mean())
   ph['vertical_error_min_max_cm']=[100*min(r['anchor'][2]-r['wrist'][2] for r in rows),100*max(r['anchor'][2]-r['wrist'][2] for r in rows)]
 if raw:
  base=raw['standing'];allanchors=np.asarray([r['anchor'] for r in base['physics']])
  assert len({e['common_state_sha256'] for e in raw.values()})==1
  for e in raw.values():
   assert np.array_equal(np.asarray([r['anchor'] for r in e['physics']]),allanchors[:len(e['physics'])])
  # Common t3 initial state is exact; learned A2 can emit tiny nonzero alpha before contact. Report it rather than force trajectory equality.
  samekeys=['qpos','qvel','force','wrist','root']
  for ctl,e in raw.items():
   for k in samekeys:
    x=np.asarray([r[k] for r in e['physics'] if r['force_time']<3.8-1e-8]);y=np.asarray([r[k] for r in base['physics'] if r['force_time']<3.8-1e-8])
    audits['up30__'+ctl].setdefault('precontact_diff_vs_standing',{})[k]=float(abs(x-y).max())
    if ctl!='A2':assert np.array_equal(x,y),(ctl,k)
 checks={};gains={}
 if len(results)==4 and all(e.get('boundary') is None for e in results.values()):
  b=results['up30__standing']['phases']['all_contact'];r=results['up30__raise_clock']['phases']['all_contact']
  for c in ['A2','forward_clock','raise_clock']:
   v=results['up30__'+c]['phases']['all_contact'];gains[c]={'error_reduction_vs_standing':1-v['tracking_rms_cm']/b['tracking_rms_cm'],'force_reduction_vs_standing':1-v['force_rms_n']/b['force_rms_n']}
  target=results['up30__raise_clock'];whole=target['phases']['whole']
  checks=dict(raise_error_improves_10pct=gains['raise_clock']['error_reduction_vs_standing']>=.1,raise_force_not_worse_5pct=r['force_rms_n']<=1.05*b['force_rms_n'],
   raise_safe=whole['failed_samples']==0 and whole['samples_without_either_foot_load']==0 and whole['undesired_contacts_max']==0,
   raise_recovery=target['root_speed_release_plus_2s_mps']<=.02)
 # Observability comparison uses pre-existing +X standing record only, not a new trial.
 old=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/issue21-skill-compat-20261031A/pairs-attempt-01/outputs/large__standing.json')
 observability={}
 if old.exists() and 'standing' in raw:
  other=json.loads(old.read_text())
  for name,e in [('old_plusX',other),('new_plusZ',raw['standing'])]:
   selected=[r for r in e['physics'] if 4.0-1e-8<=r['force_time']<4.3-1e-8]
   fs=np.asarray([r['force'] for r in selected]);mean=fs.mean(0)
   observability[name]=dict(force_xyz_mean_n=mean.tolist(),mean_force_direction=(mean/np.linalg.norm(mean)).tolist(),force_xyz_rms_n=np.sqrt((fs**2).mean(0)).tolist(),window=[4.0,4.3])
  observability['cosine']=float(np.dot(observability['old_plusX']['mean_force_direction'],observability['new_plusZ']['mean_force_direction']))
 return dict(episodes=results,audits=audits,gains=gains,raise_usefulness_checks=checks,raise_usefulness_passed=bool(checks) and all(checks.values()),observability=observability)
if __name__=='__main__':
 out=Path(sys.argv[1]);dest=Path(sys.argv[2]);r=analyze(out);dest.write_text(json.dumps(r,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in r.items() if k not in ['episodes','audits']},indent=2))
