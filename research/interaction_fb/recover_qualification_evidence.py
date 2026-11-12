import pathlib,json,sys,hashlib,time
import numpy as np,torch
S=pathlib.Path(sys.argv[1]);R=S/'runtime';sys.path[:0]=[str(R),str(R/'research_notes')]
from research.interaction_fb.journal import load_state
from diagnostic_fb_resume import tree_digest
from f1_mjdata_contract import canonical,exact
L=json.loads((R/'research_notes/adapter_envelope_lock_v1.json').read_text());cp=load_state(pathlib.Path(L['model_checkpoint']))
f=S/'runs/qualification-B19-01/envelope-sample/qualification-trace.pt';t=torch.load(f,map_location='cpu',weights_only=False)
ca=canonical(t['initial'],cp['mj_metadata']);cb=canonical(cp['physical'],cp['mj_metadata'])
checks=dict(canonical_mjdata=exact(ca['mjdata_arrays'],cb['mjdata_arrays']),BFM_history=exact(t['initial']['history'],cp['physical']['history']),rng=tree_digest(t['rng'])==tree_digest(cp['rng']))
print('checks',checks,'initialkeys',t['initial'].keys());assert all(checks.values())
j=dict(status='completed_3s' if len(t['physics'])==600 else 'incomplete',error=None,mode='envelope',sample=True,case=L['qualification_states'][0],initial_digest=tree_digest(ca),rng_digest=tree_digest(t['rng']),model_digest=tree_digest(cp['agent']),trace_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),ticks=len(t['physics']),controls=len(t['controls']),updates=cp['agent']['updates'],exact_execution_replay_qPD=True,path_audit=None,path_audit_status='FAILED_harness_inherited_outer_no_grad',provenance='posthoc read-only summary from trace saved after frozen-state and replay/qPD exact assertions; original job remains exit1',restore_checks=checks)
assert len(t['physics'])==600 and len(t['rows'])==len(t['controls'])==len(t['actions'])==150
(S/'qualification-B19-01-partial-evidence.json').write_text(json.dumps(j,indent=2));print(json.dumps(j,indent=2))
