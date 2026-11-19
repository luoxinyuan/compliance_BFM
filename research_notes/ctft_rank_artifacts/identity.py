from pathlib import Path
import hashlib,gzip,json,pickle
import numpy as np
s=Path(__file__).resolve().parent;e=s.parent
def read(p):
 with gzip.open(p) as f:return pickle.load(f)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for p in sorted((e/'ctft_d1_68/forward/reference').glob('*.state.gz')):
 q=e/'ctft_d2_69/reference'/p.name
 assert sha(q)==sha(p)
 d=read(p)['mjdata_arrays']
 rows.append(dict(state=p.name,sha=sha(p),qpos=sha(p) if False else hashlib.sha256(np.asarray(d['qpos']).tobytes()).hexdigest(),qvel=hashlib.sha256(np.asarray(d['qvel']).tobytes()).hexdigest()))
p=e/'ctft_quiet_01/paired/common-source.gz';q=e/'ctft_quiet_01/resumed/common-source.gz'
for k in ['qpos','qvel']:assert np.array_equal(read(p)['mjdata_arrays'][k],read(q)['mjdata_arrays'][k])
(s/'source_checks.json').write_text(json.dumps(dict(shared_original_snapshots=rows,quiet_qpos_qvel_equal=True),indent=2))
