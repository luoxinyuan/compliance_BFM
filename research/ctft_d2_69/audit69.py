from pathlib import Path
import csv,json,gzip,pickle,time,hashlib,shutil
import numpy as np
s=Path(__file__).resolve().parent
old=s.parent/'ctft_d1_68/forward'
def rd(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
t=time.monotonic()
hashes=rd(old/'evidence-hashes.json')
lock=rd(old/'lock/lock.json')
assert [v['source']['case'] for v in lock['picks']]==[0,1,2,4,5,6,7,8]
assert sha(old/'lock/walking.npy')==lock['walking']['z_file_sha256']
rep=rd(old/'replay/complete.json')
assert len(rep)==24 and sum(v['ticks'] for v in rep)==66432
o=s/'audit'
o.mkdir()
rows=[];sizes=[];manifest={}
for p in sorted((old/'reference').glob('[0-9][0-9]-*.gz')):
    if '.state.' in p.name:continue
    for f in [p,p.with_suffix('.state.gz'),p.with_suffix('.json')]:
        k=str(f.relative_to(old));assert sha(f)==hashes[k];manifest[k]=hashes[k]
    with gzip.open(p,'rb') as stream:x=pickle.load(stream)
    ph=x['physics'];cs=x['controls'];met=rd(p.with_suffix('.json'))
    def a(k):return np.asarray([r[k] for r in ph])
    pp=x['spec']['partner'];ang=np.deg2rad(pp['direction_deg']);h=np.array([np.cos(ang),np.sin(ang)])
    force=a('force');fr=np.c_[force[:,:2]@h,force[:,:2]@[-h[1],h[0]],force[:,2]]
    wrist=a('wrist');anchor=a('anchor');q=a('qpos');dq=a('qvel');loads=a('foot_normal_force_n')
    active=a('active');tt=a('force_time')-x['start'];branch=x['spec']['branch']
    dec=pp['distance_m']/pp['plateau_speed_mps'];hold=dec+pp['accel_s'];release=hold+pp['hold_s']
    masks={'first2s':np.arange(len(ph))<400}
    if branch=='no_force':masks['never_loaded']=np.ones(len(ph),bool)
    else:masks.update(moving=active&(tt<dec),decelerating=active&(tt>=dec)&(tt<hold),holding=active&(tt>=hold),release4s=~active&(tt>=release-1e-8))
    for good in [False,True]:
        segs=sorted([v for v in met['stances'] if v['measured'] and (v['p95']<=.05 and v['path']<=.02)==good],key=lambda v:(v['start'],v['foot']))
        if segs:
            m=np.zeros(len(ph),bool);v=segs[0];m[v['start']:v['end_exclusive']]=True;masks['first_pass' if good else 'first_fail']=m
    acts=np.asarray([v['action'].reshape(-1) for v in cs]);pd=x['contract']
    target=np.array(pd['q_PD_offset']).reshape(-1)+np.clip(5*acts,-pd['clip'],pd['clip'])*np.array(pd['q_PD_scale']).reshape(-1)/5
    np.savez_compressed(o/(p.stem+'.npz'),force_time=tt,force_axis_lateral_up=fr,root=q[:,:3],wrist=wrist,anchor=anchor,loads=loads,q=q,dq=dq,action20=acts,qPD20=target,**{'native_'+k:np.array([v['native_terms'][k] for v in cs]) for k in cs[0]['native_terms']},**{k:np.array([v['reward_components'][k] for v in cs]) for k in cs[0]['reward_components']})
    for phase,m in masks.items():
        if not m.any():continue
        lo,hi=np.flatnonzero(m)[[0,-1]];f=fr[m];mag=np.linalg.norm(f,axis=1);loaded=mag>=1;valid=f[loaded];forward=(valid[:,0]>0)&(valid[:,0]>=abs(valid[:,1]))&(valid[:,0]>=abs(valid[:,2]))
        rows.append(dict(series=p.stem,phase=phase,ticks=int(m.sum()),forward_N=f[:,0].mean(),lateral_N=f[:,1].mean(),up_N=f[:,2].mean(),forward_dominant_fraction=forward.mean() if loaded.any() else '',classification=('FORWARD_DOMINANT' if forward.mean()>.5 else 'NOT_FORWARD_DOMINANT') if loaded.any() else 'NO_MEASURABLE_LOAD',height_m=(wrist-anchor)[m,2].mean(),lag_m=((anchor-wrist)[m,:2]@h).mean() if active[m].any() else '',saturation=(mag>=10-1e-8).mean(),root_progress_m=(q[hi,:2]-(q[lo-1,:2] if lo else np.asarray(x['initial_qpos'])[:2]))@h,load_L_N=loads[m,0].mean(),load_R_N=loads[m,1].mean()))
    sizes.append(p.stat().st_size)
with (o/'phase.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
proj=rd(old/'projection.json')
forecast=dict(seconds_with_cold=proj['d2_s'],trajectory_bytes_p90=float(np.quantile(sizes,.9)),rollout_bytes=int(1.25*24*np.quantile(sizes,.9)),additional_allowance_bytes=3*2**30,prior_p90_s=proj['p90_seconds'],prior_job=rd(old/'d2/cost.json'),reference=manifest,seconds=time.monotonic()-t)
assert forecast['seconds_with_cold']<3.5*3600 and forecast['rollout_bytes']+3*2**30<min(32*2**30,shutil.disk_usage(s).free-100*2**30)
(o/'identity_and_projection.json').write_text(json.dumps(forecast,indent=2))
