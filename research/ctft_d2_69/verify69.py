from pathlib import Path
import numpy as np,gzip,pickle,json,hashlib
s=Path.cwd();rd=lambda p:json.loads(p.read_bytes())
def load(p):
    with gzip.open(p,'rb') as f:return pickle.load(f)
cmd=np.load(s/'d2/command.npy');out=[];manifest={}
for spec in rd(s/'lock/lock.json')['picks']:
    case=spec['source']['case'];branchdata={}
    for b in spec['branches']:
        key=f'{case:02d}-{b}';p=s/'d2'/(key+'-walk.gz');x=load(p);a=load(s/'reference'/(key+'.gz'))
        assert x['spec']==a['spec'] and np.array_equal(x['initial_qpos'],a['initial_qpos']) and x['model_sha']==a['model_sha'] and x['start']==a['start']
        for i,c in enumerate(x['controls']):
            expect=cmd if i<50 else a['controls'][0]['z']
            assert np.array_equal(c['z'].reshape(-1),expect.reshape(-1))
        if b=='no_force':assert not any(np.any(v['force']) or v['active'] for v in x['physics'])
        branchdata[b]=x;manifest[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    cut=int(round(spec['branches']['stop']['distance_m']/spec['branches']['stop']['plateau_speed_mps']/.005))
    ref=branchdata['continue']['physics'];st=branchdata['stop']['physics'];n=min(len(ref),len(st),cut)
    for field in ['qpos','qvel','anchor','force','foot_normal_force_n']:
        assert np.array_equal([v[field] for v in ref[:n]],[v[field] for v in st[:n]])
    out.append([case,n])
(s/'verification.json').write_text(json.dumps(dict(identity_and_command_verified=len(manifest),common_prefix_ticks=out,raw_sha=manifest),indent=2))
