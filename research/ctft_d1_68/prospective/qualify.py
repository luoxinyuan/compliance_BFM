"""Fail closed before authorizing post-reference phases."""
from pathlib import Path
import sys,json,gzip,pickle,math,shutil
import numpy as np
s=Path(sys.argv[1]);ref=s/'reference';lock=json.loads((s/'lock/lock.json').read_bytes())
ls=json.loads((ref/'complete.json').read_bytes())
assert len(ls)==3*len(lock['picks'])
def get(p):
    with gzip.open(p,'rb') as f:return pickle.load(f)
num=[]
for p in lock['picks']:
    cid=p['source']['case']; a=get(ref/f'{cid:02d}-continue.gz');b=get(ref/f'{cid:02d}-stop.gz')
    cut=b['spec']['partner']['distance_m']/b['spec']['partner']['plateau_speed_mps']
    n=min(len(a['physics']),len(b['physics']),round(cut/.005))
    assert n>0
    for k in ['qpos','qvel','force','foot_normal_force_n','anchor','anchor_velocity']:
        assert np.array_equal([r[k] for r in a['physics'][:n]],[r[k] for r in b['physics'][:n]]),(cid,k)
    num.append([cid,n])
paths=sorted(ref.glob('*.json'),key=lambda p:p.stat().st_mtime)
samples=[p for p in paths if p.name[:2].isdigit()];t=np.array([p.stat().st_mtime for p in samples])
duration=np.diff(t[:20]);p90=float(np.quantile(duration,.9));total_ticks=sum(r['ticks'] for r in ls)
model_cold=float(t[0]-json.loads((ref/'launch.json').read_text())['wall']); allowance=model_cold*1.25+60
forecast=dict(p90_seconds=p90,initial_load_plus_first_branch_s=model_cold,replay_plus_d1_s=2*allowance+1.25*p90*(len(ls)+3*len(lock['picks'])),d2_s=allowance+1.25*p90*len(ls),observed_ticks=total_ticks,phase_prefix_ticks=num)
assert forecast['replay_plus_d1_s']<7100 and forecast['d2_s']<14300
assert total_ticks*4.5+50000<1000000
assert sum(p.stat().st_size for p in ref.iterdir() if p.is_file())*6<min(96*2**30,shutil.disk_usage(s).free-100*2**30)
(s/'projection.json').write_text(json.dumps(forecast,indent=2))
