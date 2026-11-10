"""Per-tick differences and suffix-only recovery metrics on actual exposed traces."""
import csv,gzip,json,math,time
from pathlib import Path
import numpy as np
from dual_track_high import write
from native_transfer import metrics

def load(p):return json.load(gzip.open(p,'rt'))

def analyze(folder,out):
 out.mkdir(exist_ok=False);prefix='3ffd2fc10497';a=load(folder/(prefix+'-i0.json.gz'));b=load(folder/(prefix+'-i48.json.gz'))
 rows=[]
 for x,y in zip(a['physics'],b['physics']):
  assert abs(x['time']-y['time'])<1e-8
  ci=max(0,sum(c['time']<=x['force_time']+1e-8 for c in a['control'])-1);cj=max(0,sum(c['time']<=y['force_time']+1e-8 for c in b['control'])-1)
  qa=np.asarray(a['control'][ci]['q_PD']);qb=np.asarray(b['control'][cj]['q_PD'])
  rows.append(dict(time=x['time'],force_time=x['force_time'],q_max_deg=float(np.rad2deg(np.abs(np.asarray(x['qpos'][7:])-y['qpos'][7:])).max()),
   dq_max_rad_s=float(np.abs(np.asarray(x['qvel'][6:])-y['qvel'][6:]).max()),qpd_max_deg=float(np.rad2deg(np.abs(qa-qb)).max()),
   root_dx_cm=100*(y['root'][0]-x['root'][0]),root_vx0=x['qvel'][0],root_vx48=y['qvel'][0],
   wrist_shoulder0=(np.asarray(x['wrist'])-x['shoulder_xyz']).tolist(),wrist_shoulder48=(np.asarray(y['wrist'])-y['shoulder_xyz']).tolist(),
   support0=x['support'],support48=y['support'],foot_normal0=x['foot_normal_force_n'],foot_normal48=y['foot_normal_force_n'],
   slip0=x['stage_loaded_slip_mps'],slip48=y['stage_loaded_slip_mps'],force0=x['force'],force48=y['force'],
   error0_cm=100*x['error'],error48_cm=100*y['error'],active0=x['active'],active48=y['active']))
 with (out/'physical-first-divergence.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 events={}
 for name,fn in [('first_actual_joint_difference',lambda r:r['q_max_deg']>1e-8),('first_support_difference',lambda r:r['support0']!=r['support48']),
   ('first_qpd_difference_gt1deg',lambda r:r['qpd_max_deg']>1),('first_slip48_gt_point2',lambda r:r['slip48']>.2)]:
  events[name]=next((r for r in rows if fn(r)),None)
 # Thresholds are explicitly descriptive; retain complete values for independent reading.
 events['first_sustained_qpd_difference_gt1deg_three_ticks']=next((rows[i] for i in range(len(rows)-2) if all(rows[j]['qpd_max_deg']>1 for j in range(i,i+3))),None)
 recovery=[]
 for branch in json.loads((folder/'interventions.json').read_text()):
  ni=0 if branch['source_model']==0 else 1;t=branch['at'];kind=branch['kind'];p=folder/f'{prefix}-from{ni}-at{round(t*10)}-{kind}.json.gz';d=load(p);base=a if ni==0 else b
  row={k:branch[k] for k in ['source_model','to_model','at','kind','prefix_exact','prefix_fingerprint']}
  row['windows']={}
  for key,end in [('first1s',t+1),('first2s',t+2),('all_remaining',d['condition']['end'])]:
   def calc(v):
    rr=[r for r in v['physics'] if t-1e-8<=r['force_time']<end-1e-8];cc=[c for c in v['control'] if t-1e-8<=c['time']<end-1e-8]
    return dict(**metrics(rr,cc),slip_peak=max(r['stage_loaded_slip_mps'] for r in rr))
   row['windows'][key]=dict(branch=calc(d),continuation=calc(base))
  recovery.append(row)
 write(out/'events.json',events);write(out/'recovery-suffix.json',recovery)
 return events

if __name__=='__main__':
 import sys
 analyze(Path(sys.argv[1]),Path(sys.argv[2]))
