"""Read-only Issue53 pairing/group audit; no physical or F recomputation."""
from pathlib import Path
import json,gzip,time,hashlib,collections
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');O=P/'experiments/interactive-contact-pilot-20261108';S=P/'experiments/ppo-retention-audit-20261109';out=S/'runs/B-eligibility';out.mkdir(exist_ok=False);beg=time.monotonic()
allq=[]
for sub in ['diagnose-01','repair-short-01','moving-cost-01']:
 p=O/'runs'/sub/'searches.json'
 if not p.exists():
  matches=list((O/'runs').glob('*short*/searches.json'));p=matches[0] if sub=='repair-short-01' and len(matches)==1 else p
 if p.exists():
  for row in json.loads(p.read_text()):allq.append(dict(folder=str(p.parent),**row))
assert len(allq)==96,len(allq)
pairs=[];states=[]
for q in allq:
 if 'shadow'not in q:continue
 # All20 states are taken along4 actual frozen Student episodes, not20 independent prefixes.
 group=q['label'].removeprefix('shadow_').rsplit('_',1)[0]
 ep=O/'runs/diagnose-01'/('S4000_'+group+'.json.gz')
 with gzip.open(ep,'rt') as f:traj=json.load(f)
 cs=[c for c in traj['control'] if abs(c['time']-q['time'])<1e-6];assert len(cs)==1
 c=cs[0];active=bool(next(r for r in traj['physics'] if abs(r['force_time']-q['time'])<1e-6)['active'])
 fs=q['shadow']['F']['all_F_scores'];ex=q['shadow']['F']['all_available_extra'];assert len(q['candidates'])==9
 for r in q['candidates']:
  j=r['index'];fi=j if j<7 else 7+next(i for i,e in enumerate(ex) if e==r['proposal'])
  pairs.append(dict(group=group,label=q['label'],active=active,candidate=j,F=fs[fi],physical_reward=r['score'],unsafe=bool(r['boundary']),BFM_obs_available='bfm_obs'in c,previous_z_available=len(q['previous_z'])==256,privileged_available=len(q['privileged_input'])==1088,restricted_available=len(q['restricted_input'])==1639,candidate_z_explicit=False,candidate_action_explicit=False,proposal=r['proposal'],reconstruction='first z must reconstruct original command and spherical(previous_z,first,.2); action requires frozen BFM act on this exact bfm_obs; not existing explicit per-candidate tensors'))
 states.append(dict(group=group,label=q['label'],time=q['time'],active=active,rows=9,source_trajectory=str(ep),paired_BFM_state=c['bfm_obs'].keys().__iter__().__class__.__name__ if False else list(c['bfm_obs'])))
contact_groups=sorted({s['group'] for s in states if s['active']});groups=sorted({s['group'] for s in states});n=sum(len(q['candidates']) for q in allq)
assert n==712 and len(pairs)==180 and len(states)==20
result=dict(status='closed_at_data_gate',all_queries=len(allq),all_candidate_rows=n,explicit_F_physical_pairs=len(pairs),paired_states=len(states),paired_contact_states=sum(s['active'] for s in states),paired_precontact_states=sum(not s['active'] for s in states),independent_episode_groups=groups,independent_contact_groups=contact_groups,independent_contact_group_count=len(contact_groups),shared_standing_prefix='four episodes may share initial standing prefix; group count4 is an upper bound, never20/180/712 independent groups',gate_minimum_contact_groups=10,fit_allowed=False,reason='at most4 independent contact episodes <10; no calibrator fitting and no extra F recomputation',other_candidates=n-len(pairs),other_candidates_audit='No saved F values. Some original post-reset physical candidate results fall within Issue53 known affected callback interval; none silently admitted. Reconstructing tensors cannot create new independent episodes.',complete_explicit_tensor_pair_rows=0,reconstructible_shadow_pair_rows=180,existing_only_metrics_pairs=180,extra_F_computations=0,new_physics=0,Teacher_queries=0,states=states,pairs=pairs,seconds=time.monotonic()-beg)
(out/'eligibility.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['pairs','states']},indent=2))
