"""Issue 68 archive-only preflight. Explicit train-lineage file allowlist."""
from pathlib import Path
import hashlib, json, gzip, pickle, csv, time, sys
import numpy as np
import torch
s=Path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(exist_ok=True); start=time.monotonic()
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,v):(out/name).write_text(json.dumps(v,indent=2,allow_nan=False))
train=s/'runs/formal-U-v1a-01'; native=s/'runs/endpoint-evaluation-02'
srcs=read(s/'locks/source-reset-domain-v1.json')['train_sources']
eps=read(train/'train-only-check-131072.json')['source_episodes']
assert len(srcs)==12 and sum(e['controls'] for e in eps)==131072
man=[]; events=[]
for src in srcs:
    case=src['case']; lineage=src['ancestor']; group=[e for e in eps if e['spec']['source']['case']==case]
    m={}
    for a in ['B0','U','T']:
        pref=f'ctft-native-regression-{case:02d}-{a}-native_continuation.pkl.gz'
        p=native/pref; met=read(native/(pref+'.metrics.json'))
        with gzip.open(p,'rb') as f:x=pickle.load(f)
        assert x['spec']['mode']==0 and x['spec']['source']['case']==case and x['model']==a
        ph=x['physics'];cs=x['controls']
        m[a]=dict(trajectory_sha=sha(p),actor_sha=x['model_sha'],hard=met['hard'],end_s=met['terminal_time'],rows=len(ph),root_path=met['root_path'],yaw=met['yaw_abs_deg'],reason=met['hard_reason'])
        if a!='U':continue
        for typ,ok in [('fail',False),('normal',True)]:
            options=[v for v in met['stances'] if v['measured'] and (v['p95']<=.05 and v['path']<=.02)==ok]
            if not options:continue
            st=min(options,key=lambda v:(v['start'],v['foot']))
            k=st['start']; c=cs[k//4]; r=ph[k]; scale=np.asarray(x['contract']['q_PD_scale']).reshape(-1); off=np.asarray(x['contract']['q_PD_offset']).reshape(-1);act=c['action'].reshape(-1)
            events.append(dict(case=case,ancestor=lineage,domain='g0_native',kind=typ,tick=k,tick_end=st['end_exclusive'],foot=st['foot'],p95=st['p95'],material_path=st['path'],applied_force_N=float(np.linalg.norm(r['force'])),load_N=np.asarray(r['foot_normal_force_n']).tolist(),qpos=np.asarray(r['qpos']).tolist(),qvel=np.asarray(r['qvel']).tolist(),action=act.tolist(),qPD=(off+np.clip(5*act,-x['contract']['clip'],x['contract']['clip'])*scale/5).tolist(),wrist=np.asarray(r['wrist']).tolist(),root=np.asarray(r['root']).tolist(),aux=c.get('reward_components'),raw_slip=None))
    man.append(dict(case=case,lineage=lineage,original_native_source_sha=src['source_sha256'],native=m,training={b:dict(episodes=sum(e['spec']['branch']==b for e in group),controls=sum(e['controls'] for e in group if e['spec']['branch']==b)) for b in ['native_continuation','no_force','continue','stop']}))
save('train-manifest.json',man);save('native-events.json',events)
groups=[]
for arm in ['U','T']:
    d=s/f'runs/formal-{arm}-v1a-01';ps=sorted(d.glob('actual-rows-*.pt'))
    # Check field availability in every private training block. Raw individual
    # components may exist in other archives, so this finding is block-specific.
    for p in ps:
        ts=int(p.stem.split('-')[-1]); v=torch.load(p,weights_only=False,map_location='cpu')
        with gzip.open(d/f'physics-{ts}.pkl.gz','rb') as f:ph=pickle.load(f)
        keys=v['aux_rewards']; parts=v['ctft_reward_components']
        groups.append(dict(arm=arm,end=ts,count=len(ph),controls=len(v['action']),material=sum('ctft_material_contacts' in row for row in ph),individual_slip='penalty_slippage' in keys,individual_parts=sorted(parts),total_identity=bool(torch.equal((parts['native_aux']+parts['contact_aux']).float(),keys['ctft_total_native']))))
save('block-inventory.json',groups)
save('cost.json',dict(seconds=time.monotonic()-start,branches=0,ticks=0,updates=0,train_replay_rows_added=0))
