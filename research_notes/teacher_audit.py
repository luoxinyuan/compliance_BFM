from pathlib import Path
import json,statistics,hashlib
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/stop-sensing-20261103');p=R/'B1-train-02/job';rows=[json.loads(x) for x in (p/'training.jsonl').read_text().splitlines()];out={}
for k in ['physical_steps','control_steps','prefill_physics_steps','sampling_seconds','optimization_seconds']:out[k]=sum(r[k] for r in rows)
out['episodes_completed']=sum(len(r['episodes']) for r in rows);out['updates']=rows[-1]['optimizer_updates'];out['decisions']=rows[-1]['rl_decisions']
for k in ['old_GAE_value_EV','updated_value_EV','pre_logprob_max_error','pre_ratio_max_error','raw_mean_change_rms','std_mean']:
 a=[r['optimization'][k] for r in rows];out[k]=dict(first=a[0],last=a[-1],min=min(a),max=max(a))
for k in ['joint_gaussian_KL','clip_fraction']:
 a=[x[k] for r in rows for x in r['optimization']['epochs']];out[k]=dict(min=min(a),max=max(a),mean=statistics.mean(a))
out['selection']=json.loads((p/'selection.json').read_text());out['extension']=json.loads((p/'extension-decision.json').read_text());out['checkpoints']={f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in ['initial.pt','best.pt','last.pt']}
(R/'analysis/teacher-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
for d in sorted(p.glob('dev-*')):
 if not (d/'summary.json').exists():continue
 z=json.loads((d/'summary.json').read_text());print(d.name,str(z)[:100])
