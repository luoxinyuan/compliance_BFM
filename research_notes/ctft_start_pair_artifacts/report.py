import csv,gzip,pickle,json,itertools,sys
from pathlib import Path
import numpy as np
s=Path(__file__).resolve().parent;d=s/(sys.argv[1] if len(sys.argv)>1 else 'paired')
rows=[];events=[];prefix=[]
def read(p):
 with gzip.open(p,'rb') as f:return pickle.load(f)
def write(p,rr):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
sets={}
for j in sorted(d.glob('??-*-*.json')):
 n=j.stem;case,b,c=n.split('-');p=read(j.with_suffix('.gz'));m=json.loads(j.read_text());ph=p['physics'];cs=p['controls'];q0=p['initial_qpos'];par=p['spec']['partner']
 arr=lambda k:np.array([r[k] for r in ph])
 t=(arr('force_time')-p['start']).round(6);f=arr('force');wr=arr('wrist');root=arr('root');err=np.linalg.norm(arr('anchor')-wr,axis=1);loads=arr('foot_normal_force_n');yaw=np.unwrap(np.arctan2(arr('pelvis_matrix')[:,1,0],arr('pelvis_matrix')[:,0,0]))*180/np.pi
 hold=par['distance_m']/par['plateau_speed_mps']+par['accel_s'];rel=hold+par['hold_s'];end=rel+4
 aux=np.array([sum(cc['reward_components'].values()) for cc in cs if 'reward_components' in cc])
 contact=t<rel;moving=t<hold;holding=(t>=hold)&contact;released=t>=rel;v_m=t[max(np.flatnonzero(moving))]
 force=lambda mask:float(np.mean(np.sum(f[mask]**2,axis=1))**.5) if mask.any() else None
 rms=lambda mask:float(np.mean(err[mask]**2)**.5*100) if mask.any() and b!='no_force' else None
 sides=loads>=.05*344.45207502000005;support=sides[:,0].astype(int)-sides[:,1].astype(int)
 stable=[];offset=0
 for side,it in itertools.groupby(support):
  length=sum(1 for _ in it)
  if side and length>=20 and t[offset+19]<rel:
   stable.append((side,offset))
  offset+=length
 swap=None
 for i,(sd,k) in enumerate(stable[1:],1):
  if any(a!=sd for a,_ in stable[:i]) and root[k+19,0]-q0[0]>=.02:
   swap=t[k+19];break
 stance=m['stances']
 firstbad=next(iter(sorted([r for r in stance if r['measured'] and (r['p95']>.05 or r['path']>.02)],key=lambda r:r['start'])),None)
 rr=dict(case=case,ancestor=p['spec']['parent'],branch=b,cmd=c,hard=m['hard'],complete=m.get('complete_release',False),root_move_cm=(root[moving][-1,0]-q0[0])*100,wrist_move_cm=(wr[moving][-1,0]-wr[0,0])*100,exchange_s=swap,loaded_bouts=len(stable),slip=m['slip_result'],worst_p95=max((r['p95'] for r in stance if r['p95'] is not None),default=None),worst_path_cm=max((r['path']*100 for r in stance if r['path'] is not None),default=None),first_bad_stance_start_s=firstbad['start']*.005 if firstbad else None,force_N=force(contact),error_cm=rms(contact),hold_force_N=force(holding),hold_error_cm=rms(holding),stop=m.get('stopped'),release_path_m=m.get('release_root_path'),release_yaw=m.get('release_yaw'),root_path_m=m['root_path'],yaw_abs_deg=m['yaw_abs_deg'])
 rows.append(rr)
 for label,a,z in [('pre-command',0,.5),('early-only',.5,1.5),('late-only',1.5,2.5),('common-continuation',2.5,rel),('contact',0,rel),('release',rel,end)]:
  sel=(t>=a)&(t<z);idx=np.arange(len(aux));keep=(idx>=round(a*50))&(idx<round(z*50))
  if sum(keep)!=round((z-a)*50):continue
  events.append(dict(case=case,branch=b,cmd=c,window=label,return98=float(.98**np.arange(keep.sum())@aux[keep]),sum=float(aux[keep].sum()),force_N=force(sel),err_cm=rms(sel) if label!='release' else None))
 sets.setdefault((case,c),{})[b]=(t,np.array([r['qpos'] for r in ph]),np.array([r['qvel'] for r in ph]),f,np.asarray([cc['action'] for cc in cs]))
for (case,c),ss in sets.items():
 if not {'continue','stop'}<=ss.keys():continue
 a,b=ss['continue'],ss['stop'];k=min(300,len(a[0]),len(b[0]))
 assert all(np.array_equal(aa[:k],bb[:k]) for aa,bb in zip(a[:4],b[:4])) and np.array_equal(a[4][:k//4],b[4][:k//4])
 prefix.append(dict(case=case,cmd=c,pair='continue/stop',verified_prefix_ticks=k))
write(s/'branches.csv',rows);write(s/'phase_returns.csv',events)
if prefix:write(s/'prefix.csv',prefix)
