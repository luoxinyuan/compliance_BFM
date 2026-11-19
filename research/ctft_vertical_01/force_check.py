from pathlib import Path
import gzip,json,pickle
import numpy as np
s=Path.cwd();checks=[];sha=json.loads((s/'soft/model-identities.json').read_text())['U']
for p in sorted((s/'soft').glob('[0-9][0-9]*.gz')):
    def load(k):
        with gzip.open(k,'rb') as f:return pickle.load(f)
    x=load(p);a=load(s/'reference'/p.name)
    assert x['model_sha']==a['model_sha']==sha and x['spec']==a['spec'] and np.array_equal(x['initial_qpos'],a['initial_qpos'])
    assert all(np.array_equal(c['z'],a['controls'][0]['z']) for c in x['controls'])
    raw=np.array([r['ctft_raw_force'] for r in x['physics']])
    applied=np.array([r['force'] for r in x['physics']]);diff=np.max(abs(applied-raw*np.minimum(1,10/np.maximum(1e-12,np.linalg.norm(raw,axis=1)))[:,None]));assert diff<1e-12
    at0=np.array(a['physics'][0]['ctft_raw_force']);at0[2]*=.25
    assert np.allclose(raw[0],at0,rtol=0,atol=1e-12)
    checks.append(dict(series=p.stem,max_force_mapping_residual_N=float(diff),ticks=len(raw)))
Path('force-contract.json').write_text(json.dumps(checks,indent=2))
