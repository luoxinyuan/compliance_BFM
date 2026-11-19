from pathlib import Path
import sys,json,hashlib,statistics,math
s=Path(sys.argv[1])
def read(p):return json.loads(p.read_bytes())
def median(v):return round(statistics.median(v),4) if v else None
arms={};cov={};results=read(s/'comparisons.json')
for folder in ['reference','replay','d1','d2']:
    arms[folder]=read(s/folder/'cost.json')
    cov[folder]=read(s/folder/'complete.json')
rows=[]
for p in sorted((s/'reference').glob('[0-9][0-9]-*.json')):
    ref=read(p); alt=s/'d2'/p.name.replace('.json','-walk.json')
    if not alt.exists():continue
    walk=read(alt)
    if 'released' not in ref.get('stages',{}):continue
    def stats(a):
        return {k:a.get(k) for k in ['root_forward','root_path','yaw_abs_deg','force_rms','lag_rms','stopped','release_root_path','release_yaw','slip_result','hard']}
    rows.append(dict(branch=ref['branch'],case=p.name[:2],standing=stats(ref),walking=stats(walk)))
reps={}
for r in results:
    if r['status']=='MATCHED' and r['case'] not in reps:reps[r['case']]=r
overview=dict(cost=arms,replay_exact=len(cov['replay']),checks=read(s/'replay/slip-recompute.json'),cases=sorted({r['case'] for r in results}),compare=results,reps=reps,full=rows,projection=read(s/'projection.json'),hard={k:[r for r in vs if r['hard']] for k,vs in cov.items()})
(s/'summary.json').write_text(json.dumps(overview,indent=2))
for stage in ['D1','D2']:
    from collections import Counter
    print(stage,Counter(r['status'] for r in results if r['test']==stage))
print('H_R',[(k,v['test'],v['proxy_weak_or_reversed'],v['aux_prefers_clean']) for k,v in reps.items()])
for branch in ['continue','stop','no_force']:
    sub=[r for r in rows if r['branch']==branch]
    for a in ['standing','walking']:
        print(branch,a,{k:median([r[a][k] for r in sub if r[a][k] is not None]) for k in ['root_forward','root_path','yaw_abs_deg','force_rms','lag_rms','release_root_path','release_yaw']})
print('hard',{k:len(v) for k,v in overview['hard'].items()},'cost',arms)
