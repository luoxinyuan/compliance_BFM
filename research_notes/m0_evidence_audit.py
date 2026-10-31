"""Read-only audit of M0 raw physical records, no model or simulator execution."""
import hashlib,json
from pathlib import Path
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/m0-lpp-20261031A'
read=lambda p:json.loads(p.read_text())
ledger=read(R/'compute-ledger.json');audits={};source_sha=set();weights=set()
for attempt in ledger['attempts']:
    if attempt.get('result_status')!='completed':continue
    out=Path(attempt['path'])/'outputs';result=read(out/'result.json');assert result['original_bfm_unchanged']
    contract=read(out/'action-contract.json');weights.add(contract['model_sha256'])
    scale=np.asarray(contract['q_PD_scale']);offset=np.asarray(contract['q_PD_offset'])
    for path in out.glob('*__*.json'):
        ep=read(path)
        if 'physics' not in ep:continue
        cs=ep['control'];rs=ep['physics'];a=np.asarray([c['action'] for c in cs]);q=np.asarray([c['q_PD'] for c in cs])
        recomputed=offset+np.clip(a*contract['normalize_ratio'],-contract['clip'],contract['clip'])*scale/contract['normalize_ratio']
        qerr=float(abs(q-recomputed).max());assert qerr<1e-6
        wrench=float(max(np.max(abs(np.asarray(r['force'])-np.asarray(r['applied_wrist_wrench'][:3]))) for r in rs));assert wrench<1e-10
        forceerr=float(max(np.max(abs(np.asarray(c['measured_force_xyz'])-np.asarray(rs[4*i-1]['force']))) for i,c in enumerate(cs) if i));assert forceerr<1e-10
        assert max(abs(np.diff([c['time'] for c in cs])-.02))<1e-10
        assert max(abs(np.diff([r['time'] for r in rs])-.005))<1e-10
        gatezero=None
        if 'executed_z' in cs[0] and ep['controller'] in ['P0','alignment','distill','rule']:
            standing=np.asarray(read(P/'bridge-v2-owned/runs/issue19-wrist-anchor-20261028T062503Z/attempt-02/outputs/standing_latent.json'),dtype=np.float32).reshape(1,256)
            z=np.asarray([c['executed_z'][0] for c in cs]);angles=np.rad2deg(np.arccos(np.clip((z*standing).sum(-1)/256,-1,1)))
            assert angles.max()<=np.rad2deg(np.arctan(.3))+.001
            ix=np.asarray([c['gate']==0 for c in cs]);gatezero=float(abs(z[ix]-standing).max()) if ix.any() else None
            if gatezero is not None:assert gatezero==0
        source_sha.add(ep.get('common_state_sha256','half old common state'))
        audits[ep['label']]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),q_PD_recompute_max_rad=qerr,wrench_force_error=wrench,previous_physics_force_error=forceerr,zero_gate_standing_error=gatezero,control_samples=len(cs),physics_samples=len(rs),complete=ep['boundary'] is None)
assert len(weights)==1
result=dict(records=audits,model_sha256=list(weights)[0],common_states=list(source_sha),total_simulated_seconds=sum(v['physics_samples']/200 for v in audits.values()),resources=ledger,scope='Branch physics excludes repeated3second prefixes; simulator reset internals are not counted. No reexecution or model forward.')
(R/'analysis/evidence-audit.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps({k:result[k] for k in ['common_states','total_simulated_seconds']},indent=2))
