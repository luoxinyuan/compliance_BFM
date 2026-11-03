from pathlib import Path
import json,hashlib,shutil
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/heading-transfer-20261103';OUT=P/'maintenance/research-handoff-20261102/research_notes/heading_artifacts';OUT.mkdir(exist_ok=True)
d=json.loads((R/'compute-ledger.json').read_text());assert all(x['status']=='finished' and not x.get('remaining_processes') for x in d['attempts'])
result=dict(ledger=d,cost_seconds=sum(x['wall_seconds'] for x in d['attempts']),output_bytes=sum(p.stat().st_size for p in R.rglob('*') if p.is_file()),frozen=json.loads((R/'frozen-identities.json').read_text()),heading_selection=json.loads((R/'heading-final-selection.json').read_text()),heading_final=json.loads((R/'B-final-06/job/summaries.json').read_text()),student=json.loads((R/'A-prepare-05/job/summaries.json').read_text()),coverage=json.loads((R/'A-prepare-05/job/coverage.json').read_text()),student_training={},heading_development={})
for name in ['visit_low','visit_high']:
 x=json.loads((R/f'A-prepare-05/job/{name}-training.json').read_text());assert hashlib.sha256((R/f'A-prepare-05/job/{name}.pt').read_bytes()).hexdigest()==x['sha256'];result['student_training'][name]=x
for step in [0,5,10,15,20]:result['heading_development'][step]=json.loads((R/f'B-train-04/job/dev-{step:04d}/summary.json').read_text())
rows=[json.loads(s) for s in (R/'B-train-04/job/training.jsonl').read_text().splitlines()];result['rl']=dict(decisions=rows[-1]['rl_decisions'],physical_steps=sum(x['physical_steps'] for x in rows),control_steps=sum(x['control_steps'] for x in rows),prefill_physics_steps=sum(x['prefill_physics_steps'] for x in rows),logged_completed_episodes=sum(len(x['episodes']) for x in rows),logged_episode_boundaries=sum(bool(e['boundary']) for x in rows for e in x['episodes']),curve=rows)
result['recovery']={f.name:json.loads(f.read_text()) for f in (R/'A-prepare-05/job').glob('*-recovery.json')}
result['source_attempts']={k:json.loads((R/k/'job/summaries.json').read_text()) for k in ['B-prepare-02','B-prepare-03']}
(OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
for src,dest in [('A-prepare-05/job/physical-metrics-v2.json','student-physics.json'),('B-final-06/job/physical-metrics-v2.json','heading-physics.json'),('B-final-06/job/causal-response-v2.json','causal-response.json'),('videos-01/manifest.json','video-manifest.json')]:shutil.copyfile(R/src,OUT/dest)
print('cost',result['cost_seconds'],'GiB',result['output_bytes']/1024**3,'RL', {k:v for k,v in result['rl'].items() if k!='curve'})
for x in result['student']:
 if x['label'].startswith(('S0_','visit_low_','visit_high_')):print(x['label'],round(x['error_cm'],3),round(x['force_rms'],3))
print('models',[(k,v['sha256']) for k,v in result['student_training'].items()])
