"""Read-only final M1 gate/asset audit; no fitting, selection or rerunning cases."""
import json,hashlib,time,collections,gzip,math
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/source-coverage-m1-20261109';R=S/'repo';N=R/'research_notes';OUT=N/'source_coverage_m1_artifacts';start=time.monotonic();prot=json.loads((N/'source_coverage_m1_protocol.json').read_text());gate=prot['gate'];rows=[];searches=[];counts=collections.Counter();missing=[];no=[];assets=[];extra_controls=[]
for source in ['standing','goal_arm5','motion_walk_task']:
 d=S/'runs'/(source+'-r2')
 if (d/'counts.json').exists():counts.update(json.loads((d/'counts.json').read_text()))
 if (d/'summaries.json').exists():rows+=json.loads((d/'summaries.json').read_text())
 if (d/'searches.json').exists():searches+=json.loads((d/'searches.json').read_text())
 for p in d.glob('query-*/summary.json'):
  q=json.loads(p.read_text());prefix=p.parent;files=[prefix/'state.pt',prefix/'plans.pt']+[prefix/f'candidate-{i}-{suffix}' for i in range(7) for suffix in ['F.pt','prediction.json.gz']]
  absent=[str(x) for x in files if not x.is_file()];assert not absent,absent
  assets.append(dict(group=q['group'],split=q['split'],cell=q['cell'],query=q['query'],state=q['state_fingerprint'],path=str(prefix),files=[dict(name=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files],candidate_rows=7,actual_query_time=q['time'],runtime=json.loads((d/'launch.json').read_text())['runtime']))
def verdict(row,bc,cell):
 a=row['all_quality'];b=bc['all_quality'];req=cell['request'];fail=[]
 for key,limit in [('error_cm',gate['error_cm'][req]),('force_rms',gate['force_N'][req]),('cap_fraction',gate['cap_fraction']),('slip_peak_mps',gate['loaded_slip_peak_mps'])]:
  if a[key] is None or a[key]>limit:fail.append(key)
 if abs(a['yaw_final_deg'])>gate['release_abs_yaw_deg']:fail.append('release_yaw')
 for key in ['error_cm','force_rms']:
  if a[key] is None or a[key]>1.05*b[key]+1e-9:fail.append('relative_'+key)
 if row['boundary'] is not None or not row['complete']:fail.append('boundary_or_incomplete')
 if req=='stop':
  h=row.get('holding',{});
  if h.get('error_cm',math.inf)>gate['stop_hold_error_cm'] or h.get('force_rms',math.inf)>gate['stop_hold_force_N']:fail.append('stop_holding')
 return dict(passed=not fail,failures=fail,metrics={k:a[k] for k in ['error_cm','force_rms','cap_fraction','slip_peak_mps','legacy_allcontact_slip_peak_mps','yaw_final_deg']},holding=row.get('holding'),actual_role=row['identity'])
cells=[]
for c in prot['pilot']['cells']:
 rr=[r for r in rows if r['cell']==c['id']];bc=next((r for r in rr if r['identity']=='actual_Student_frozen_BC500'),None);nat=next((r for r in rr if r['identity']=='actual_native_continuation_contact_control'),None);branches=[r for r in rr if r['identity'] in ['actual_selected_plan24_then_BC_aftermath','actual_Teacher20_firstcommands_then_BC_aftermath']]
 if bc is None:
  cells.append(dict(cell=c,usable=False,missing='no actual BC reference / source stratum unavailable'));continue
 vv={r['identity']:verdict(r,bc,c) for r in branches};cells.append(dict(cell=c,BC=bc['all_quality'],native=nat['all_quality'] if nat else None,arms=vv,usable=any(v['passed'] for v in vv.values()),labels_scope='Only actual rolling first commands from a passed rolling arm qualify; selected openplan commands not assumed closed-loop expert labels. Full episode includes BC aftermath.'))
for r in rows:
 if r['cell']=='native_no_contact':
  a=r['all_quality'];passed=r['complete'] and r['boundary'] is None and a['force_rms']==0 and a['min_height']>=.45 and a['max_tilt']<=30
  no.append(dict(source=r['source'],passed=passed,metrics=a))
usable=sum(c['usable'] for c in cells);by_source={s:sum(c['usable'] and c['cell']['source']==s for c in cells) for s in ['standing','goal_arm5','motion_walk_task']};gatepass=usable>=8 and by_source['goal_arm5']>=1 and by_source['motion_walk_task']>=1
qualified_query_pairs=[]
for c in cells:
 if c.get('arms',{}).get('actual_Teacher20_firstcommands_then_BC_aftermath',{}).get('passed'):
  src=c['cell']['source'];win=json.loads((S/'runs'/(src+'-r2')/(c['cell']['id']+'-window.json')).read_text());qualified_query_pairs += [dict(source=src,cell=c['cell']['id'],query=q) for q in win['queries']]
manifest=dict(protocol_sha=hashlib.sha256((N/'source_coverage_m1_protocol.json').read_bytes()).hexdigest(),cells=cells,no_contact_native=no,usable=usable,by_source=by_source,pilot_gate_pass=gatepass,independent_native_prefix_groups=len(set(q['group'] for q in searches)),evaluation_units=len(cells),split_actual=dict(train_groups=len(set(q['group'] for q in searches)),dev_groups=0,sealed_groups=0),split_target=prot['independence']['split_target'],counts=dict(counts),assets=assets,qualified_actual_teacher_labels=qualified_query_pairs,decision='生成能力不足' if not gatepass else '数据资格不足',decision_reason='predeclared pilot gate failed; no expansion' if not gatepass else 'pilot passed but only3 common-prefix families; cannot claim requested independent coverage',M2_started=False,training_started=False,full_episode_success_not_inferred_from_prediction=True)
# Supplemental exact original-course controls never alter12-cell gate.
for name in ['exact-standing','exact-arm']:
 d=S/'runs'/name
 if (d/'summaries.json').exists():extra_controls+=json.loads((d/'summaries.json').read_text())
 if (d/'counts.json').exists():counts.update(json.loads((d/'counts.json').read_text()))
 if (d/'searches.json').exists():
  qq=json.loads((d/'searches.json').read_text());searches+=qq
  for q in qq:
   prefix=d/f"query-{q['query']:03d}";files=[prefix/'state.pt',prefix/'plans.pt']+[prefix/f'candidate-{i}-{suffix}' for i in range(7) for suffix in ['F.pt','prediction.json.gz']]
   assert all(f.is_file() for f in files)
   assets.append(dict(group=q['group'],split=q['split'],cell=q['cell'],query=q['query'],state=q['state_fingerprint'],path=str(prefix),files=[dict(name=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files],candidate_rows=7,actual_query_time=q['time'],runtime=json.loads((d/'launch.json').read_text())['runtime'],scope='supplemental original-domain control not gate rescue'))
manifest['exact_original_controls']=extra_controls;manifest['counts']=dict(counts);manifest['assets']=assets
# Per state paired original-F/physics ranks, with exact asset provenance; no calibration.
def ranks(v):
 return [sum(x<y for x in v)+(sum(x==y for x in v)-1)/2 for y in v]
def corr(x,y):
 ax=sum(x)/len(x);ay=sum(y)/len(y);d=sum((a-ax)**2 for a in x)*sum((b-ay)**2 for b in y)
 return sum((a-ax)*(b-ay) for a,b in zip(x,y))/d**.5 if d else None
paired=[]
for q in searches:
 cc=q['candidates'];fs=[c['F_score'] for c in cc];ps=[c['physical_score'] for c in cc];rank=sorted(range(7),key=lambda i:fs[i],reverse=True);best=max(range(7),key=lambda i:ps[i]);paired.append(dict(group=q['group'],cell=q['cell'],query=q['query'],F_spearman=corr(ranks(fs),ranks(ps)),Ftop4_preserves_best=best in rank[:4],best_physical=best,Ftop1=rank[0],regret=ps[best]-ps[rank[0]],F_contract='first interpolated action',physics_contract='24 control commands from same complete state',predicted_best=cc[best]['physical'],same_state_candidates=7))
manifest['paired_ranking']=paired
OUT.mkdir(exist_ok=True);(OUT/'m1-evidence.json').write_text(json.dumps(manifest,indent=2,allow_nan=False));print(json.dumps(dict(decision=manifest['decision'],usable=usable,by_source=by_source,counts=dict(counts),assets=len(assets),qualified_actual_teacher_labels=len(qualified_query_pairs),analysis_seconds=time.monotonic()-start)))
