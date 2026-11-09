"""Read-only segment and prediction-realization audit; no fitting or physical queries."""
import sys,json,gzip,time
from pathlib import Path
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/source-coverage-m1-20261109';N=S/'repo/research_notes';sys.path.insert(0,str(N))
from native_transfer import metrics
out=[];gaps=[]
def load(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def measure(ep,a,b):
 rr=[r for r in ep['physics'] if a-1e-7<=r['force_time']<b-1e-7];cc=[r for r in ep['control'] if a-1e-7<=r['time']<b-1e-7]
 if not rr:return None
 m=metrics(rr,cc);m['loaded_slip_peak_mps']=max(r['loaded_slip_peak_mps'] for r in rr);m['loaded_slip_p95_mps']=float(np.quantile([r['loaded_slip_peak_mps'] for r in rr],.95));m['yaw_final_deg']=float(np.degrees(np.arctan2(rr[-1]['pelvis_matrix'][1][0],rr[-1]['pelvis_matrix'][0][0])));return m
for name in ['standing-r2','goal_arm5-r2','motion_walk_task-r2','exact-standing','exact-arm']:
 d=S/'runs'/name;qs=json.loads((d/'searches.json').read_text())
 for wp in d.glob('*-window.json'):
  w=json.loads(wp.read_text());cell=wp.name.removesuffix('-window.json');real=load(d/(cell+'_rolling.json.gz'));bc=load(d/(cell+'_BC.json.gz'));a,b=w['start'],w['end'];end=max(r['time'] for r in real['physics'])+1e-5
  out.append(dict(run=name,cell=cell,Teacher_window=[a,b],Teacher_window_metrics=measure(real,a,b),BC_same_window_metrics=measure(bc,a,b),BC_after_Teacher_metrics=measure(real,b,end),BC_uninterrupted_same_after_window=measure(bc,b,end),roles=[dict(start=a,end=b,controller='actual rolling Teacher first commands'),dict(start=b,end=end,controller='frozen BC500 aftermath')]))
 for q in qs:
  if q['label']!='Student_visited_first':continue
  cell=q['cell'];real=load(d/(cell+'_best_sequence.json.gz'));i=q['selected'];pred=load(d/f"query-{q['query']:03d}/candidate-{i}-prediction.json.gz");a=q['time'];b=a+2.4
  ar=[r for r in real['physics'] if a-1e-7<=r['force_time']<b-1e-7];pr=pred['physics'];nr=min(len(ar),len(pr));ac=[c for c in real['control'] if a-1e-7<=c['time']<b-1e-7];pc=pred['control'];nc=min(len(ac),len(pc))
  gaps.append(dict(run=name,cell=cell,query=q['query'],selected=i,prediction=q['candidates'][i]['physical'],actual_same_24commands=measure(real,a,b),anchor_prediction_error_rms_cm=float(100*np.sqrt(np.mean(np.sum((np.array([r['anchor'] for r in ar[:nr]])-np.array([r['anchor'] for r in pr[:nr]]))**2,axis=1)))),first_actual_qpd_vs_prediction_max_deg=float(np.degrees(np.max(np.abs(np.array(ac[0]['q_PD'])-pc[0]['q_PD'])))),target_sequence_equal=np.allclose([c['executed_z'] for c in ac[:nc]],[c['executed_z'] for c in pc[:nc]],rtol=0,atol=1e-5),note='Identical planned high-level sequence; BFM observations evolve under different future partner motion. Actual full episode subsequently uses BC500.'))
(N/'source_coverage_m1_artifacts/segments.json').write_text(json.dumps(dict(rolling_windows=out,prediction_to_actual=gaps),indent=2,allow_nan=False))
print('segments',len(out),'same-plan actual comparisons',len(gaps))
