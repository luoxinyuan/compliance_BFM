"""Offline exact-label, closed-loop divergence, and actual same-state diagnostics."""
from pathlib import Path
import gzip,json,math
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/m38-rotation-20261103'
def read(p):return json.load(gzip.open(p,'rt')) if str(p).endswith('.gz') else json.loads(p.read_text())
def first(t,v,th,n):
 for i in range(len(v)-n+1):
  if np.all(v[i:i+n]>th):return dict(time=float(t[i]),value=float(v[i]))
 return None
out={};baseline=read(R/'A-goal-04/job/T30_straight.json.gz')
for name,path in [('r1',R/'A-goal-02/job/retained_straight.json.gz'),('r2',R/'A-goal-03/job/retained_straight.json.gz'),('r2_T30prefix',R/'A-goal-04/job/T30_prefix_straight.json.gz')]:
 e=read(path);a=baseline['physics'];b=e['physics'];assert len(a)==len(b);t=np.array([r['time'] for r in a]);q=np.degrees(np.abs(np.array([r['qpos'][7:] for r in a])-np.array([r['qpos'][7:] for r in b]))).max(1);root=100*np.linalg.norm(np.array([r['root'] for r in a])-np.array([r['root'] for r in b]),axis=1);ca=baseline['control'];cb=e['control'];ct=np.array([r['time'] for r in ca]);pd=np.degrees(np.abs(np.array([r['q_PD'] for r in ca])-np.array([r['q_PD'] for r in cb]))).max(1)
 out[name]=dict(qPD_over1deg_for100ms=first(ct,pd,1,5),actual_q_over1deg_for100ms=first(t,q,1,20),root_over1cm_for100ms=first(t,root,1,20),caveat='Descriptive thresholds; diverged closed-loop paths no longer same-state. Same-state counterqueries logged separately.')
# Exact elementwise verification avoids CUDA cdist cancellation near zero.
old=P/'bridge-v2-owned/runs/contact-geometry-20261103/C-goal-10/job';a=read(old/'source_pair_left-labels.json.gz')[0];b=read(old/'retain_pair_no_contact-labels.json.gz')[0];c=read(R/'A-goal-03/job/causal_pair_left-labels.json.gz')[0];pairs={}
for name,x,y in [('old_goal_vs_T30',a,b),('causal_source_vs_T30',c,b)]:
 pairs[name]=dict(input_max_abs=float(np.max(np.abs(np.array(x['x'])-y['x']))),bfm_obs_max_abs=max(float(np.max(np.abs(np.array(x['obs'][k])-y['obs'][k]))) for k in x['obs']),target_z_max_abs=float(np.max(np.abs(np.array(x['z'])-y['z']))),target_angle_deg=math.degrees(math.acos(np.clip(np.dot(x['z'],y['z'])/256,-1,1))))
out['exact_label_consistency']=pairs
(R/'diagnostics.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
# Existing source state versus actual r2 visited state: individual row stats, not causal proof.
for lab in ['T30_straight','T30_prefix_straight','sealed_left_r2','sealed_right_r2']:
 p=R/'A-goal-04/job';same=read(p/(lab+'-same-state.json'));dist=read(p/(lab+'-distribution.json'));onset=read(p/(lab+'.json.gz'))['condition']['onset'];summary={'contact_onset':onset}
 for phase,sel in [('pre',lambda r:r['time']<onset),('later',lambda r:r['time']>=onset)]:
  ix=[i for i,r in enumerate(same) if sel(r)]
  if not ix:continue
  summary[phase]={k:float(np.mean([same[i]['relative_to_T30']['r2'][k] for i in ix])) for k in ['z_angle_deg','qpd_rms_deg','first_tick_qpd_rms_deg']};summary[phase]['nearest_train_input_rms']=float(np.mean([dist[i]['nearest_normalized_rms'] for i in ix]))
 out[lab+'_same_state']=summary
(R/'diagnostics.json').write_text(json.dumps(out,indent=2))
