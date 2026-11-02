from pathlib import Path
import sys,json,gzip,hashlib
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/clean-contact-expert-20261102'
sys.path.insert(0,str(Path(__file__).resolve().parent));from evaluate_clean_saved import metrics
result={}
for p in sorted(R.glob('*-final-*/job/evaluation/summary.json')):
 launch=json.loads((p.parents[1]/'launch.json').read_text());resume=launch['args']['resume'];name=p.parents[2].name
 result[name]={'source_checkpoint':resume,'checkpoint_sha256':hashlib.sha256(Path(resume).read_bytes()).hexdigest(),'summary':json.loads(p.read_text()),'physical':{}}
 for q in p.parent.glob('*.json.gz'):
  with gzip.open(q,'rt') as f:e=json.load(f)
  result[name]['physical'][q.stem]=metrics(e)
(R/'evaluation-analysis/final-evidence.json').write_text(json.dumps(result,indent=2,allow_nan=False))
for name,r in result.items():
 for row in r['summary']:
  if row['label'].endswith('__learned'):
   ph=next(x['phases'] for k,x in r['physical'].items() if '__learned' in k)
   print(name,'err',row['error_cm'],'force',row['force_rms'],'cap',row['cap_fraction'],'recovery',ph['recover']['end_root_speed_rms'],'move',ph['move']['root_delta'],'tilt',max(p['tilt_peak'] for p in ph.values()),'qPD',max(p.get('qpd_step_peak_deg',0) for p in ph.values()),'supportloss',max(p['no_support_fraction'] for p in ph.values()))
