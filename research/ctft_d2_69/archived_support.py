import json,csv,numpy as np
from pathlib import Path
s=Path.cwd();old=s.parent/'ctft_d1_68'
ids={0,1,2,4,5,6,7,8}
def read(p):return json.loads(p.read_bytes())
native=[{'case':r['case'],**r['native']['U']} for r in read(old/'d0/train-manifest.json') if r['case'] in ids]
(s/'audit/g0-native.json').write_text(json.dumps(native,indent=2))
segs=[]
for p in sorted((s/'reference').glob('[0-9][0-9]-*.json')):
    j=read(p);a=np.load(s/'audit'/(p.stem+'.npz'))
    for v in j['stances']:
        lo,hi=v['start'],v['end_exclusive'];foot=v['foot'];d=a['loads'][lo:hi];c=a['native_penalty_slippage'][lo//4:(hi+3)//4]
        segs.append(dict(series=p.stem,**v,previous_load_N=float(a['loads'][lo-1,foot]) if lo else None,loaded_mean_N=d[:,foot].mean(),other_foot_mean_N=d[:,1-foot].mean(),next_load_N=float(a['loads'][hi,foot]) if hi<len(a['loads']) else None,native_slip_2x=2*c.mean(),qpd_rms=float(np.sqrt(np.mean(a['qPD20'][lo//4:(hi+3)//4]**2))),native_aux=a['native_aux'][lo//4:(hi+3)//4].mean(),contact_aux=a['contact_aux'][lo//4:(hi+3)//4].mean()))
with (s/'audit/stance-support.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(segs[0]));w.writeheader();w.writerows(segs)
