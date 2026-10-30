"""Fixed final criteria separated into operational semantics, safety and task value."""
import json
from pathlib import Path
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/issue21-alpha-cross-20261031A')
p=json.loads(Path(__file__).with_name('alpha_cross_protocol.json').read_text())
f=json.loads((R/'analysis/final-analysis.json').read_text());r=f['results'];checks={}
for c in p['conditions']:
 for k in ['A','B']:
  label=c['name']+'__'+k;v=r[label];d=dict(safe=v['safe'])
  if c['enabled']:d['recovery']=v['recovery_pass']
  if 'disconnect_start' in c:d.update(operational_behavior=v['behavior_pass'],post_reconnect_gain_positive=v['post_reconnect_gain']>0)
  elif not c['enabled']:d['no_false_contact']=v['negative_pass']
  elif c['name']!='pulse':
   d.update(normal_error_gain=v['error_gain']>=.3,normal_force_gain=v['force_gain']>=.05,no_material_regression=max(f['regression'][label]['error_change'],f['regression'][label]['force_change'])<=.05)
  checks[label]=d
sealed={k:all(all(checks[c['name']+'__'+k].values()) for c in p['conditions'] if c['split']=='sealed_final') for k in ['A','B']}
old={k:all(all(checks[c['name']+'__'+k].values()) for c in p['conditions'] if c['split']!='sealed_final') for k in ['A','B']}
v=dict(checks=checks,sealed_pass=sealed,regression_pass=old,all_new_physics_safe=f['all_safe'],note='Operational thresholds unchanged; not a claim that400mswithdrawal optimizes human interaction. No requirement to erase bounded imitation residual before considering separate interface-feasibility research.')
(R/'analysis/final-qualification.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
