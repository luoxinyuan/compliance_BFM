import numpy as np,gzip,json,pickle,csv
from pathlib import Path
p=Path(__file__).resolve().parent;e=p.parent
def aux(n):
 with gzip.open(e/'ctft_positive50_01/aux03'/(n+'.json.gz'),'rt') as f:a=json.load(f)
 return np.array([r['contact']+sum(r['native'].values()) for r in a])[50:]
def quiet(v,b):
 with gzip.open(e/'ctft_quiet_01'/('paired' if v=='B0' else 'resumed')/(v+'-'+b+'.gz')) as f:x=pickle.load(f)
 return np.array([sum(r['reward_components'].values()) for r in x['controls']])
rows=[]
for group,a,b,L in [('50_MPC-C32',aux('rolling_continue'),aux('C32__speed0.15'),340),('quiet_continue_U-B0',quiet('U','continue'),quiet('B0','continue'),275),('quiet_stop_U-B0',quiet('U','stop'),quiet('B0','stop'),200)]:
 for gamma in (.98,1.):
  times=np.arange(L);cum=np.cumsum((a-b)[:L]*gamma**times)
  ii=np.flatnonzero(cum<=0)
  after=('never' if len(ii) and ii[-1]==L-1 else float((ii[-1]+2 if len(ii) else 1)*.02))
  rows.append(dict(pair=group,gamma=gamma,contact_s=L*.02,first_second=float(cum[49]),second_second=float(cum[99]-cum[49]),rest=float(cum[-1]-cum[99]),gap_end=float(cum[-1]),time_positive_thereafter=after))
with (p/'cost_timeline.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
