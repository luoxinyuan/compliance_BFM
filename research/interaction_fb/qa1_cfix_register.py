"""Register C repair under the existing QA1/R2 budgets; no optimization/physics."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/r2-g17-qa1-20261115';O=P/'experiments/r2-v11-qentry-20261114';R=S/'runtime-cfix'
for name in ('pointer_layout.so','pointer_layout.json'):
    source=O/'prep'/name;target=S/'prep'/name
    expected=json.loads((O/'prep/dependency-preflight.json').read_text())['sha256'][name]
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    if not target.exists():shutil.copyfile(source,target)
    assert hashlib.sha256(target.read_bytes()).hexdigest()==expected
ledger=json.loads((S/'ledger.json').read_text());old=json.loads((O/'ledger.json').read_text())
assert ledger['status']=='STOP_AWAIT_DECISION' and ledger['diagnostic_Q_groups']==dict(U=456,C=0)
record=dict(name='QA1-C-isolation-repair-preparation',category='G17-QA1',wall_s=480,cost_type='conservative_upper_bound',
    includes='isolated worktree materialization, code review, CPU Adam fixture, library hash verification, IO and approval archive',physics_ticks=0,scientific_Q_groups=0)
ledger['jobs'].append(record);ledger['charged_upper_s']+=480;ledger['status']='C_REPAIR_AUTHORIZED_PRELAUNCH'
old['job_entries'].append(record);old['total_charged_upper_s']+=480;old['status']='G17_QA1_C_REPAIR_AUTHORIZED'
assert ledger['charged_upper_s']+1200+300<=3600
(S/'ledger.json').write_text(json.dumps(ledger,indent=2));(O/'ledger.json').write_text(json.dumps(old,indent=2))
print(json.dumps(dict(qa1_used=ledger['charged_upper_s'],job_limit=1200,publication_reserve=300,
    projected_subtotal=ledger['charged_upper_s']+1500,hard_cap=3600,prior_reserved_ticks=ledger['physics_counts']['reserved'])))
