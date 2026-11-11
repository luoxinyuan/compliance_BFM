"""CPU verification of complete real continuation against archived unmodified paths."""
import argparse,gzip,json,time
from pathlib import Path
from collections import Counter
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');O=P/'experiments/frl-v2-fb-repair-20261110';run=O/'runs/frl-formal-B-01'
def read(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else p.open() as f:return json.load(f)
rows=read(a.output/'outcomes.json');checks=[];stat=read(run/'rollout-65-stats.json');trainph=read(run/'physics-65.json.gz');trainco=read(run/'native-controls-65.json.gz')
for state in sorted(set(r['state_id'] for r in rows)):
 rr=[r for r in rows if r['state_id']==state];is_train=rr[0]['continuation']!='mean';kind='archived_sample' if is_train else 'original_mean';row=next(r for r in rr if r['candidate']==kind or kind in r['duplicate_sources']);ep=read(a.output/(f'state{state:02d}-'+row['candidate']+'.json.gz'));meta=ep['meta'];ph=ep['physics'];co=ep['control']
 if is_train:
  j=meta['index'];pi=sum(s['physics_steps'] for s in stat[:j]);ci=sum(s['control_steps'] for s in stat[:j]);oldph=trainph[pi:pi+len(ph)];oldco=trainco[ci:ci+len(co)]
 else:
  k=dict(BC500=0,B64=64,B128=128)[row['state_owner']];old=read(run/f'dev-{k}'/(row['parent_id'][:12]+'-main.json.gz'));pi=next(i for i,r in enumerate(old['physics']) if abs(r['force_time']-row['time'])<1e-7);ci=next(i for i,r in enumerate(old['control']) if abs(r['time']-row['time'])<1e-7);oldph=old['physics'][pi:pi+len(ph)];oldco=old['control'][ci:ci+len(co)]
 assert len(oldph)==len(ph) and len(oldco)==len(co)
 diffs={}
 for key in ['qpos','qvel','force','error','support','stage_loaded_slip_mps','pelvis_matrix','torso_matrix','torque']:
  aa=np.asarray([r[key] for r in ph]);bb=np.asarray([r[key] for r in oldph]);assert aa.shape==bb.shape;diffs[key]=float(np.max(np.abs(aa.astype(float)-bb.astype(float))));assert np.array_equal(aa,bb),('archived physical continuation differs',state,key,diffs[key])
 for key in ('executed_z','action','q_PD'):
  aa=np.asarray([r[key] for r in co]);bb=np.asarray([r[key] for r in oldco]);diffs[key]=float(np.max(abs(aa-bb)));assert np.array_equal(aa,bb),(state,key,diffs[key])
 checks.append(dict(state_id=state,compared_candidate=kind,physical_rows=len(ph),controls=len(co),archived_continuation_exact=True,max_errors=diffs))
# Terminal tails are not folded into legacy c.reward_integral by the archived executor.
# Preserve original outputs and separately audit actual optimized scores including tails.
corrected=[]
for row in rows:
 ep=read(a.output/(f"state{row['state_id']:02d}-"+row['candidate']+'.json.gz'))
 tail=sum(.99**((cc['time']-row['time'])/.1)*cc.get('absorbing_failure_penalty',0.) for cc in ep['control'])
 corrected.append(dict(state_id=row['state_id'],candidate=row['candidate'],old_return=row['scores']['old_discounted_return']-tail,v3_return=row['scores']['v3_discounted_return']-tail,terminal_tail=tail))
(a.output/'optimized-score-audit.json').write_text(json.dumps(dict(rows=corrected,terminal_cases=sum(r['terminal_tail']!=0 for r in corrected),original_scores_affected=sum(r['terminal_tail']!=0 for r in corrected),scope='original outputs retained; termination tail explicitly audited'),indent=2))
sem=read(a.output/'load-semantics.json');bands=Counter();bodies=Counter()
for e in sem:
 n=e['normal_N'];bands['<=1N' if n<=1 else '1-10N' if n<=10 else '10-100N' if n<=100 else '>100N']+=1;bodies[e['body']]+=1
norm=[e['normal_N'] for e in sem]
gate=read(a.output/'gate.json');result=dict(status='completed',states=len(checks),all_archived_continuations_bitwise=True,physics_rows=sum(r['physical_rows'] for r in checks),controls=sum(r['controls'] for r in checks),prefix_checks=checks,load_semantics=dict(overlimit20ms_events=len(sem),normal_bands=dict(bands),bodies=dict(bodies),normal_quantiles_N=dict(zip(['p50','p90','p99','max'],map(float,np.quantile(norm,[.5,.9,.99,1])))) if norm else {},scope='one peak tick per overlimit20ms event; all loads counted using unchanged1e-6N; bands descriptive only'),P2_allowed=gate['P2_allowed'],job_s=time.monotonic()-t)
(a.output/'counterfactual-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='prefix_checks'},indent=2))
