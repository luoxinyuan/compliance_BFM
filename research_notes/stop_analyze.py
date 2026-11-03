"""Saved data only; never selects a Student on final results."""
from pathlib import Path
import json,gzip,collections,hashlib
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/stop-sensing-20261103';out=R/'analysis';out.mkdir(exist_ok=True)
rows=[];training=[]
for job in sorted(R.glob('*/job')):
 if (job/'summaries.json').exists():
  for r in json.loads((job/'summaries.json').read_text()):r['job']=job.parent.name;rows.append(r)
 for f in job.glob('*-training.json'):training.append(dict(path=str(f),**json.loads(f.read_text())))
summary={}
for r in rows:
 p=R/r['job']/'job'/(r['label']+'.json.gz')
 if not p.exists():continue
 with gzip.open(p,'rt') as f:ep=json.load(f)
 phy=ep['physics'];ctrl=ep['control'];active=[x for x in phy if x['active']];summ=dict(r)
 if not phy:continue
 for phase,(a,b) in ep['phases'].items():
  rr=[x for x in phy if a<=x['time']<b];cc=[x for x in ctrl if a<=x['time']<b]
  if not rr:continue
  F=np.array([x['force'] for x in rr]);E=np.array([np.array(x['anchor'])-x['wrist'] for x in rr]);Q=np.array([x['qpos'][7:] for x in rr]);root=np.array([x['root'] for x in rr]);w=np.array([x['wrist'] for x in rr]);shoulder=np.array([x['shoulder_xyz'] for x in rr]);qpd=np.array([x['q_PD'] for x in cc]);support=np.array([x['support'] for x in rr]);valid=np.array([x['active'] for x in rr]);q={}
  q.update(error_cm=100*float(np.sqrt((E*E).sum(1).mean())),signed_xyz_cm=(E.mean(0)*100).tolist(),force_rms=float(np.sqrt((F*F).sum(1).mean())),force_xyz_rms=np.sqrt((F*F).mean(0)).tolist(),mean_force_xyz=F.mean(0).tolist(),cap=float((np.linalg.norm(F,axis=1)>=r.get('max_force_N',5.)-1e-3).mean()),active_samples=int(valid.sum()),root_delta=(root[-1]-root[0]).tolist(),wrist_delta=(w[-1]-w[0]).tolist(),wrist_relative_root_delta=((w[-1]-root[-1])-(w[0]-root[0])).tolist(),shoulder_delta=(shoulder[-1]-shoulder[0]).tolist(),tilt_max=max(x['tilt_deg'] for x in rr),min_height=min(x['height'] for x in rr),no_foot_support_fraction=float((support.sum(1)==0).mean()),single_support_fraction=float((support.sum(1)==1).mean()),joint_limit=max(x['joint_limit_excess_max_rad'] for x in rr),torque_saturation=max(x['torque_saturation'] for x in rr),joint_step_max_deg=float(np.rad2deg(np.abs(np.diff(Q,axis=0))).max()) if len(Q)>1 else 0,qpd_step_max_deg=float(np.rad2deg(max(x['q_PD_step_max_rad'] for x in cc))) if cc else 0)
  summ.setdefault('detailed_phases',{})[phase]=q
 summ['recovery_curve']=[]
 release=ep['condition']['release']
 for a in np.arange(release,ep['condition']['end'],.5):
  rr=[x for x in phy if a<=x['time']<a+.5]
  if rr:summ['recovery_curve'].append(dict(start=float(a),root_speed_rms=float(np.sqrt(np.mean([np.dot(x['qvel'][:3],x['qvel'][:3]) for x in rr]))),wrist_mean=np.mean([x['wrist'] for x in rr],axis=0).tolist()))
 summary[r['job']+'/'+r['label']]=summ
(out/'physical-summary.json').write_text(json.dumps(summary,indent=2));(out/'training-summary.json').write_text(json.dumps(training,indent=2))
for r in rows:
 h=r.get('phases',{}).get('hold',{});d=r.get('phases',{}).get('decelerate',{})
 print(r['label'],round(r['error_cm'],3),round(r['force_rms'],3),'hold',round(h.get('error_cm',0),3),[round(x,2) for x in h.get('signed_xyz_cm',[])],r['boundary'])
