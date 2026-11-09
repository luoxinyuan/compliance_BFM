"""Archive-only episode/phase counts and bounded M1 acquisition ledger."""
import json,time,collections,hashlib
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/source-coverage-m1-20261109';O=S/'repo/research_notes/source_coverage_m1_artifacts'
c=json.loads((O/'coverage5403.json').read_text());joint=collections.defaultdict(lambda:dict(episode_ids=[],complete_episodes=0,partial_episodes=0,decisions=0));outcomes=[]
for e in c['episodes']:
 s=e['spec'];key=(s['source'],s['demand_family'])
 for ph,n in e['phases'].items():
  j=joint['|'.join((*key,ph))];j['episode_ids'].append(e['index']);j['complete_episodes']+=int(e['complete']);j['partial_episodes']+=int(not e['complete']);j['decisions']+=n
 # Historical episodes lack complete slip/yaw. Partial qualification deliberately cannot mean full pass.
 d=s['distance'];bounds=(10,6.5) if d>=.5 else ((8,7) if .3<=d<=.45 else ((4,4) if d<=.15 else None))
 contact=e['active']>0
 errs=[]
 if bounds and contact:
  if e['error_cm']>bounds[0]:errs.append('contact_error')
  if e['force_rms']>bounds[1]:errs.append('actual_force')
  if e['cap_fraction']>.15:errs.append('saturation')
 if e['boundary']:errs.append('boundary')
 outcomes.append(dict(episode=e['index'],source=s['source'],request=s['demand_family'],complete=e['complete'],known_failed_metrics=errs,full_quality_pass=None,reason='saved train logs lack full slip/yaw; 15-30cm amplitudes have no user absolute bin',metric_bounds=bounds))
(O/'coverage5403-phase-episodes.json').write_text(json.dumps(dict(joint=dict(joint),outcomes=outcomes,phase_assignment='100ms decision start; crossing decisions not relabeled as exact physics ticks',full_quality_success_count='not recoverable from archived logs'),indent=2))
jobs=[]
for p in sorted((S/'jobs').glob('*.json')):
 j=json.loads(p.read_text());assert j['status']!='running',p;jobs.append(j)
files=[p for p in (S/'runs').rglob('*') if p.is_file()];qs=list((S/'runs').glob('*/query-*'));complete=[p for p in qs if (p/'summary.json').exists()]
ledger=dict(jobs=jobs,supervised_job_seconds=sum(j['elapsed_s'] for j in jobs),output_bytes=sum(p.stat().st_size for p in files),query_attempts=len(qs),completed_queries=len(complete),original7_candidate_rows=7*len(complete),extra_FB_branches=0,RL_decisions=0,training_updates=0,F_fits=0,archive_analysis_seconds='separate analysis script runtimes recorded in final ledger',limits=dict(job_seconds=18000,output_GiB=8,full_queries_stage=400,full_queries_pilot=96,extra_FB=80),split_scope='3 native prefix families; all train; no dev/sealed used',replay=json.loads((S/'runs/asset-replay/result.json').read_text()),timestamp=time.time())
(O/'resource-ledger.json').write_text(json.dumps(ledger,indent=2));print(json.dumps({k:ledger[k] for k in ['supervised_job_seconds','output_bytes','query_attempts','completed_queries','original7_candidate_rows']}))
