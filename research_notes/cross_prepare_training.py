"""A one-time evidence gate and immutable data manifest; no model execution."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNS=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs')
RUN=RUNS/'issue21-alpha-cross-20261031A'
read=lambda p:json.loads(Path(p).read_text())
p=read(ROOT/'research_notes/alpha_cross_protocol.json')
diagnosis=read(RUN/'analysis/diagnostic-analysis.json')
probe=read(RUN/'diagnostic-attempt-01/outputs/cross-frozen-probe.json')
effects=diagnosis['factor_effects'];cells=diagnosis['results']
old=read(RUNS/'issue21-alpha-repair-20261031A/analysis/repair-results.json')
checks=dict(all_physics_safe=diagnosis['all_safe'],teacher_operational_pass=all(cells[c['name']+'__contact']['behavior_pass'] for c in p['matrix']),
 force_history_reconstructs=all(v['force_fifo_reconstruction_max']==0 for v in probe.values()),
 same_state_teacher_mapping_error=any(v['alpha_rms']>.05 for k,v in probe.items() if '__contact__' in k),
 condition_effect_beyond_numerical_noise=any(max(abs(v[k]) for k in ['late_minus_early_short','late_minus_early_long','long_minus_short_early','long_minus_short_late'])>.01 for v in effects.values()),
 prior_matched_gap_mapping_learned=all(old['episodes'][c+'__'+ctl]['shadow_alpha_rms_gap_resume1s']<.01 for c in ['short02','long08'] for ctl in ['A','B']),
 usable_real_task_benefit=all(cells[c['name']+'__'+ctl]['post_reconnect_gain']>0 for c in p['matrix'] for ctl in ['A','B']))
decision=dict(coverage_gate_passed=all(checks.values()),max_paired_fits=1,checks=checks,evidence=effects,
 interpretation='Matched old gaps learned, missing crossed combinations fail with teacher-state mapping residual and sufficient measured-force suffix. This supports a single coverage intervention, not proof coverage is the only cause. Teacher400ms remains operational, task benefit separate.')
target=RUN/'analysis/training-decision.json';assert not target.exists();target.write_text(json.dumps(decision,indent=2));print(json.dumps(decision,indent=2))
if not decision['coverage_gate_passed']:raise SystemExit('STOP: coverage gate did not pass; no fit')
for c in p['matrix']:
 for ctl,count in [('contact',16),('A',8),('B',8)]:
  f=Path(c['reference_dir'])/(c['name']+'__'+ctl+'.json');ep=read(f)
  assert not ep['condition']['name'].startswith('sealed_')
  p['training_sources'].append(dict(name='cross_'+c['name']+'__'+ctl,path=str(f.relative_to(RUNS)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),kind='interruption',label='teacher' if ctl=='contact' else 'shadow',batch_count=count))
assert sum(s['batch_count'] for s in p['training_sources'])==256
assert sum(len(read(RUNS/s['path'])['control']) for s in p['training_sources'])==p['training_samples']==10020
f=ROOT/'research_notes/alpha_cross_training_protocol.json';assert not f.exists();f.write_text(json.dumps(p,indent=2)+'\n')
