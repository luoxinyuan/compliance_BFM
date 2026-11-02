"""Read-only evidence extraction; no model update or physical rollout."""
from pathlib import Path
import ast,json,gzip,hashlib
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/direct-transfer-20261103';D=P/'maintenance/research-handoff-20261102/research_notes'
# Reuse the recorded phase metric implementation without executing old experiment globals.
tree=ast.parse((D/'evaluate_clean_saved.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='metrics');scope={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'saved_metrics','exec'),scope);base_metrics=scope['metrics']
def metrics(ep):
 result=base_metrics(ep)
 for name,(lo,hi) in ep['phases'].items():
  rows=[r for r in ep['physics'] if lo<=r['time']<hi]
  if not rows:continue
  m=result['phases'][name];m['body_delta']={k:(np.array(rows[-1]['body_positions'][k])-rows[0]['body_positions'][k]).tolist() for k in rows[0]['body_positions']}
  q=np.array([r['qpos'][7:] for r in rows]);m['actual_joint_step_peak_deg_200Hz']=float(np.rad2deg(np.abs(np.diff(q,axis=0))).max()) if len(q)>1 else 0.
  for part in ['pelvis','torso']:
   rot=np.array([r[part+'_matrix'] for r in rows]);m[part+'_tilt_peak_deg']=float(np.rad2deg(np.arccos(np.clip(rot[:,2,2],-1,1))).max())
 return result
def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def divergence(teacher,student):
 a=teacher['physics'];b=student['physics'];n=min(len(a),len(b));assert np.max(np.abs(np.array([x['time'] for x in a[:n]])-np.array([x['time'] for x in b[:n]])))<1e-5
 w=np.linalg.norm(np.array([r['wrist'] for r in a[:n]])-np.array([r['wrist'] for r in b[:n]]),axis=1);root=np.linalg.norm(np.array([r['root'] for r in a[:n]])-np.array([r['root'] for r in b[:n]]),axis=1)
 idx=next((i for i in range(n-19) if np.all(w[i:i+20]>.01)),None)
 return dict(first_wrist_gap_gt1cm_for100ms=None if idx is None else a[idx]['time'],gap_at_contact_cm=float(w[np.argmin(abs(np.array([r['time'] for r in a[:n]])-teacher['condition']['onset']))]*100),max_root_gap_cm=float(root.max()*100),scope='paired trajectory divergence; not causal attribution by itself')
def main():
 result={'student':{},'teacher_training':{},'physical':{},'divergence':{},'models':{}}
 jobs=list(R.glob('A-prepare-*/job'));assert len(jobs)==1;job=jobs[0]
 for f in job.glob('*-training.json'):result['student'][f.stem]=json.loads(f.read_text())
 for f in job.glob('*-recovery.json'):result['student'][f.stem]=json.loads(f.read_text())
 for f in job.glob('*.pt'):result['models'][f.name]=dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
 result['summaries']=json.loads((job/'summaries.json').read_text())
 result['visited_same_state']={}
 for f in job.glob('*-same-state.json'):
  rows=json.loads(f.read_text());result['visited_same_state'][f.stem]={k:dict(mean=float(np.mean([r[k] for r in rows])),p95=float(np.percentile([r[k] for r in rows],95)),max=max(r[k] for r in rows)) for k in ['z_angle_deg','first_tick_qpd_rms_deg','endpoint_action_rms']}

 for f in job.glob('*.json.gz'):
  if f.name.endswith('-queries.json.gz'):continue
  ep=read(f);result['physical'][f.stem]=metrics(ep)
 for key in ['initial','ppo15']:
  for condition in ['single','slower','faster','heldout_slow','heldout_fast','early_stop']:
   tf=job/f'{key}__source_{condition}.json.gz'
   if not tf.exists():tf=job/f'{key}__teacher_final_{condition}.json.gz'
   for rnd in ['r0','final']:
    sf=job/f'{key}__student_{rnd}_{condition}.json.gz'
    if sf.exists() and tf.exists():result['divergence'][sf.stem]=divergence(read(tf),read(sf))
 for run in sorted(R.glob('B*-train-*')):
  p=run/'job';data={}
  if (p/'training.jsonl').exists():data['curve']=[json.loads(x) for x in (p/'training.jsonl').read_text().splitlines()]
  data['dev']={q.parent.name:json.loads(q.read_text()) for q in p.glob('dev-*/summary.json')}
  if (p/'selection.json').exists():data['selection']=json.loads((p/'selection.json').read_text())
  result['teacher_training'][run.name]=data
 for f in R.glob('B*-final-*/job/evaluation/*.json.gz'):result['physical'][f.parent.parent.parent.name+'__'+f.stem]=metrics(read(f))
 for repair in [R/'A-final-04/job']:
  result['corrected_student']={f.stem:json.loads(f.read_text()) for f in repair.glob('*-training.json')}
  result['validated_queries']={f.stem:json.loads(f.read_text()) for f in repair.glob('*-validated-queries.json')}
  if (repair/'summaries.json').exists():result['corrected_summaries']=json.loads((repair/'summaries.json').read_text())
  for f in repair.glob('*.pt'):result['models']['corrected__'+f.name]=dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
  for f in repair.glob('*.json.gz'):
   if f.name.endswith('-queries.json.gz'):continue
   ep=read(f);result['physical']['corrected__'+f.stem]=metrics(ep)
   key=ep['label'].split('__')[0];condition=ep['label'].split('_final_')[-1];tf=job/f'{key}__source_{condition}.json.gz'
   if not tf.exists():tf=job/f'{key}__teacher_final_{condition}.json.gz'
   if tf.exists():result['divergence']['corrected__'+f.stem]=divergence(read(tf),ep)
 output=R/'analysis';output.mkdir(exist_ok=True);(output/'evidence.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 print('student summaries',len(result['summaries']),'physical episodes',len(result['physical']))
 for key,v in result['divergence'].items():print(key,v)
if __name__=='__main__':main()
