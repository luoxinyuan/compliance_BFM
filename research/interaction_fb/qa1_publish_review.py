"""Public-artifact review; sanitize sidecar and add paired diagnostic limitations."""
import hashlib,json
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/r2-g17-qa1-20261115';R=S/'runtime-cfix';N=R/'research_notes'
p=N/'r2_ei_qa1_complete_results.json';d=json.loads(p.read_text())
d['C_job']={k:d['C_job'][k] for k in ('name','runtime','limit_s','wall_s','exit_code','physics_counts','status')}
raw=json.loads((S/'runs/C-repaired-04/qa1-result.json').read_text())
def keyed(a):return {(r['round'],r['a'],r['b']):r for r in raw['retrospective'][a]['nodes']['512']['comparisons'] if r['eligible']}
u,c=keyed('U'),keyed('C');assert u.keys()==c.keys() and len(u)==58
paired=dict(C_correct_U_wrong=sum(c[k]['correct'] and not u[k]['correct'] for k in u),
 U_correct_C_wrong=sum(u[k]['correct'] and not c[k]['correct'] for k in u),both_correct=sum(u[k]['correct'] and c[k]['correct'] for k in u),
 both_wrong=sum(not u[k]['correct'] and not c[k]['correct'] for k in u))
d['paired_512_retrospective']=paired;p.write_text(json.dumps(d,indent=2))
assert '/mnt/' not in p.read_text() and 'command' not in d['C_job']
report=N/'R2_EI_G17_QA1_COMPLETE_RESULTS.md';text=report.read_text();needle='完整核对456批实际累计父组'
text=text.replace(needle,f"配对58对中：C对/U错{paired['C_correct_U_wrong']}，U对/C错{paired['U_correct_C_wrong']}，共同对{paired['both_correct']}、共同错{paired['both_wrong']}。C中途256组从32/58回落到29/58，未表现单调稳定改善；单seed且仅已暴露分支，不能声称统计显著或新父组泛化。\n\n"+needle)
report.write_text(text)
original=P/'experiments/r2-execution-20261114/runtime/research/interaction_fb/approved_learning.py';current=R/'research/interaction_fb/approved_learning.py'
assert original.read_bytes()==current.read_bytes()
print(json.dumps(dict(paired=paired,original_Q_update_module_unchanged=True,sha256=hashlib.sha256(current.read_bytes()).hexdigest(),public_sidecar_sanitized=True),indent=2))
