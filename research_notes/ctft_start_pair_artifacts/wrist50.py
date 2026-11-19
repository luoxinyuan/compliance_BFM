from pathlib import Path
import numpy as np,json,gzip,csv,itertools
p=Path(__file__).resolve().parent;ex=p.parent
def read(n):
 with gzip.open(n,'rt') as f:return json.load(f)
def write(n,rs):
 with (p/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
res=[];events=[];series=[]
for name in ['rolling_continue','C32__speed0.15']:
 stage=ex/'native-rolling-h100-20261107/runs'/('rolling-02' if name=='rolling_continue' else 'qualification-03')
 x=read(stage/(name+'.json.gz'));a=read(ex/'ctft_positive50_01/aux03'/(name+'.json.gz'))
 ph=x['physics'];cs=x['control'];t=np.round([v['time']-5 for v in ph],6)
 arr=lambda k:np.array([v[k] for v in ph])
 wrist=arr('wrist')[:,0];sh=arr('shoulder_xyz')[:,0];root=arr('root')[:,0];rel=wrist-sh
 loads=arr('foot_normal_force_n')>=.05*344.45207502000005;single=loads[:,0].astype(int)-loads[:,1].astype(int);adv=[];zero=np.flatnonzero(t>=0)[0];k=zero
 for side,it in itertools.groupby(single[zero:]):
  l=sum(1 for _ in it)
  if l>=20 and side:adv.append((k,k+l-1,side))
  k+=l
 exchanges=[k+19 for j,(k,end,side) in enumerate(adv) if any(o!=side for _,_,o in adv[:j]) and root[k+19]-root[zero]>=.02]
 first=t[adv[0][0]] if adv else None;swap=t[exchanges[0]] if exchanges else None
 tc=np.round([v['time']-5 for v in cs],6);vw=np.array([v['post_velocity'] for v in cs]);vp=arr('anchor_velocity')[3::4]
 f=arr('force')[3::4];filtered=np.zeros(3);refs=[]
 for force in f:
  filtered+=(-np.expm1(-.02/.05))*(force-filtered);h=filtered[:2];norm=np.linalg.norm(h);ref=.025*max(0,norm-.5)*h/(norm+1e-8);refs.append(ref*min(1,.25/max(np.linalg.norm(ref),1e-12)))
 refs=np.array(refs)
 for lo,hi in [(0,.5),(.5,1),(1,1.5),(1.5,2),(2,2.5),(2.5,3),(3,4.8)]:
  mask=(tc>=lo)&(tc<hi);ids=np.flatnonzero(mask);pi=np.flatnonzero((t>lo)&(t<=hi));end=pi[-1];start=pi[0]-1
  res.append(dict(model=name,from_s=lo,to_s=hi,vx_mean=vw[mask,0].mean(),vx_min=vw[mask,0].min(),vx_max=vw[mask,0].max(),horizontal_speed_p95=np.percentile(np.linalg.norm(vw[mask,:2],axis=1),95),vx_reward_ref_mean=refs[mask,0].mean(),vx_partner_mean=vp[mask,0].mean(),root_cm=100*(root[end]-root[start]),wrist_cm=100*(wrist[end]-wrist[start]),relative_cm=100*(rel[end]-rel[start]),force_rms=np.mean((f[mask]**2).sum(1))**.5,error_rms_cm=100*np.mean(np.sum((arr('anchor')[pi]-arr('wrist')[pi])**2,axis=1))**.5,response_cost_avg=np.mean([a[j]['parts']['response'] for j in ids]),lateral_response_cost=2*np.mean((vw[mask,1]-refs[mask,1])**2)/.25**2))
 for lo,hi,desc in [(0,3,'first_3s'),(0,first,'before_first_sustained_single_support'),(first,swap,'first_single_to_advancing_exchange')]:
  if lo is None or hi is None or hi<=lo:continue
  idx=np.flatnonzero((t>=lo)&(t<=hi))
  for frame,xx in [('world',wrist),('shoulder_relative',rel)]:
   peak=xx[idx[0]];pk=tail=peakidx=idx[0];drop=0.
   for j in idx:
    if xx[j]>peak:peak=xx[j];peakidx=j
    if peak-xx[j]>drop:drop=peak-xx[j];pk=peakidx;tail=j
   events.append(dict(model=name,interval=desc,coord=frame,retraction_cm=100*drop,peak_s=t[pk],trough_s=t[tail],root_delta_cm=100*(root[tail]-root[pk]),shoulder_delta_cm=100*(sh[tail]-sh[pk]),first_single_s=first,exchange_confirm_s=swap))
 for j in range(zero,len(t),4):
  if t[j]>4.8:break
  series.append(dict(model=name,time=t[j],wrist=100*(wrist[j]-wrist[zero]),root=100*(root[j]-root[zero]),rel=100*(rel[j]-rel[zero]),force_x=arr('force')[j,0],support=int(single[j])))
write('wrist_phases.csv',res);write('retraction.csv',events);write('wrist_curve.csv',series)
