"""CPU-only post-hoc byte-sensitive trajectory digest check; no physics execution."""
import time,json
from pathlib import Path
start=time.monotonic();here=Path(__file__).resolve();S=here.parents[2] if here.parent.name=='research_notes' else here.parents[1]
import sys
sys.path[:0]=[str(S/'runtime'),str(S/'runtime/research_notes')]
import torch
from f1_array_audit import digest
A=S/'runs/array-audit-03'; rows=json.loads((A/'interventions.json').read_text());refs={};checked=[]
for r in rows:
 sid=r['state_id']
 if sid not in refs:refs[sid]=digest(torch.load(A/(sid+'-reference.pt'),map_location='cpu',weights_only=False))
 got=digest(torch.load(A/r['path'],map_location='cpu',weights_only=False))
 checked.append(dict(state_id=sid,arrays=r['arrays'],mode=r['mode'],reference_digest=refs[sid],result_digest=got,exact=refs[sid]==got))
 assert checked[-1]['exact'],r
result=dict(status='completed',checks=checked,elapsed_s=time.monotonic()-start,training=0,physics=0)
(A/'byte-digest-validation.json').write_text(json.dumps(result,indent=2))
(S/'jobs/cpu-byte-digest.json').write_text(json.dumps(dict(status='completed',elapsed_s=result['elapsed_s'],kind='CPU only; read saved trajectories',count=len(rows)),indent=2));print(len(rows),result['elapsed_s'])
