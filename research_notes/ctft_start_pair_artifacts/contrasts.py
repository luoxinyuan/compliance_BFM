import csv
from pathlib import Path
import numpy as np
s=Path(__file__).resolve().parent
r=list(csv.DictReader((s/'branches.csv').open()))
p=list(csv.DictReader((s/'phase_returns.csv').open()))
idx={(a['case'],a['branch'],a['cmd']):a for a in r}
wins={(a['case'],a['branch'],a['cmd'],a['window']):a for a in p}
rr=[]
for case in sorted({a['case'] for a in r}):
 for cmd in ('early','late'):
  for branch in ('continue','stop','no_force'):
   a=idx[case,branch,cmd];b=idx[case,branch,'unchanged']
   delta=lambda k:float(a[k])-float(b[k]) if a[k] and b[k] else None
   w=wins.get((case,branch,cmd,'contact'));q=wins.get((case,branch,'unchanged','contact'))
   rr.append(dict(case=case,branch=branch,cmd=cmd,root_delta_cm=delta('root_move_cm'),exchange_new=a['exchange_s'],exchange_old=b['exchange_s'],force_delta_N=delta('force_N'),error_delta_cm=delta('error_cm'),slip_old=b['slip'],slip_new=a['slip'],return98_delta=float(w['return98'])-float(q['return98']) if w and q else None,total_delta=float(w['sum'])-float(q['sum']) if w and q else None,release_path_delta_m=delta('release_path_m')))
with (s/'contrasts.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
