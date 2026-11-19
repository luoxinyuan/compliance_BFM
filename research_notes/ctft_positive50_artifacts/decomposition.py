import json,gzip,csv,math
from pathlib import Path
s=Path(__file__).resolve().parent
rows=[]
for p in sorted((s/'aux03').glob('*.gz')):
 a=json.load(gzip.open(p));n=p.name.split('.json')[0];hold=8.5 if n=='rolling_stop' else 9.8;rel=hold+2
 for stage,start,end in [('move',5,hold),('first_second',5,6),('hold',hold,rel),('full_contact',5,rel),('suffix',4,rel+4)]:
  rr=a[round((start-4)*50):round((end-4)*50)]
  for gamma in [.98,1.]:
   weights=[gamma**i for i in range(len(rr))]
   r={k:-sum(w*b['parts'][k] for b,w in zip(rr,weights)) for k in rr[0]['parts']}
   native=sum(w*sum(b['native'].values()) for b,w in zip(rr,weights))
   rows.append(dict(model=n,stage=stage,gamma=gamma,**r,native=native,total=sum(r.values())+native))
with (s/'decomposition.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for a in rows:
 if a['model'] in ('rolling_continue','C32__speed0.15','I__speed0.15') and a['stage'] in ('move','full_contact'):
  print(a['model'],a['stage'],a['gamma'],*[round(a[k],1) for k in ['response','burden','vertical','idle','total']])
