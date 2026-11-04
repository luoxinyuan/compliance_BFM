"""Saved-data only force and backward-encoding diagnostic. No policy/physics."""
import json,gzip,time,math
from pathlib import Path
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/transition-curriculum-20261104';D=R/'C-final-02/job';start=time.monotonic()
rows=json.loads((D/'summaries.json').read_text());small=[]
for r in rows:
 small.append({k:r[k] for k in ['label','condition','error_cm','sum_force_norm_rms','cap_fraction','actual_torso_turn_deg','actual_pelvis_turn_deg','boundary','qpd_max_deg','phase_metrics']})
bs=json.loads((D/'backward-diagnostic.json').read_text());ref=next(x for x in bs if x['label']=='cap5_stand_walk_T30');ref=np.array([s['B'] for s in ref['samples'] if 6<=s['time']<=11]);ref/=np.linalg.norm(ref,axis=-1,keepdims=True)
reports=[]
for ep in bs:
 data=json.loads(gzip.decompress((D/(ep['label']+'.json.gz')).read_bytes()));trace=data['physics'];vals=[]
 for s in ep['samples']:
  row=min(trace,key=lambda r:abs(r['time']-s['time']))
  if not row['active']:continue
  b=np.array(s['B']);b/=np.linalg.norm(b);distance=float(1-np.max(ref@b))
  err=float(np.sqrt(np.mean([np.sum((np.array(c['anchor'])-c['point'])**2) for c in row['contacts']])))
  vals.append([distance,err,float(np.linalg.norm(row['qvel'][:2])),row['sum_force_norm']])
 a=np.array(vals)
 if len(a)>3:
  corr=np.corrcoef(a.T)
  reports.append(dict(label=ep['label'],samples=len(a),nearest_motion_sequence_cosine_distance_mean=float(a[:,0].mean()),physical_error_mean_cm=float(100*a[:,1].mean()),root_speed_mean=float(a[:,2].mean()),correlation_distance_error=float(corr[0,1]) if np.isfinite(corr[0,1]) else None,correlation_distance_speed=float(corr[0,2]) if np.isfinite(corr[0,2]) else None))
parity={k:max(s['input_parity'].get(k,0) for ep in bs for s in ep['samples']) for k in bs[0]['samples'][0]['input_parity']}
result=dict(scope='saved trajectories only; exploratory B diagnostic not preregistered success criterion',runtime='8d1a02754807eda84fc2842921c9577c56a9dafb',episodes=small,B=dict(reference='actual T30 5N standing-to-walk moving sequence t6..11; nearest sequence cosine, NOT single-pose goal distance',reconstruction_max_abs=parity,episodes=reports,limitation='Associations are descriptive across behavior/contact domains; no probability calibration, no causal proof and no B shaping used'),cpu_analysis_seconds=time.monotonic()-start)
(R/'prepare-results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
for r in small:print(r['label'],round(r['error_cm'],3),round(r['sum_force_norm_rms'],3),round(r['actual_torso_turn_deg'],2))
print('B parity',parity,'CPU seconds',result['cpu_analysis_seconds'])
