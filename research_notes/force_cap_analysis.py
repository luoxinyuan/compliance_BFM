from pathlib import Path
import json,gzip
import numpy as np
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/stop-sensing-20261103');out=R/'analysis';jobs=list(R.glob('F-final-*/job'));job=jobs[-1];rows=[];episodes={}
for r in json.loads((job/'summaries.json').read_text()):
 with gzip.open(job/(r['label']+'.json.gz'),'rt') as f:e=json.load(f)
 episodes[r['label']]=e;rr=e['physics'];on=e['condition']['onset'];cap=r['max_force_N'];times=np.array([x['time'] for x in rr]);v=np.array([x['qvel'][0] for x in rr]);start=None
 for i in range(len(rr)-40):
  if times[i]>=on and np.all(v[i:i+40]>.05):start=float(times[i]);break
 result=dict(label=r['label'],cap_N=cap,error_cm=r['error_cm'],force_rms=r['force_rms'],cap_fraction=r['cap_fraction'],peak_force=r['peak_force_N'],first_sustained_forward_005_mps=start,boundary=r['boundary'],segments={})
 windows={**e['phases'],'early':[on,6.],'walk_early':[6.,8.],'walk_late':[8.,9.45],'late_hold':[13.,14.35]}
 for label,(a,b) in windows.items():
  ss=[x for x in rr if a<=x['time']<b];cc=[x for x in e['control'] if a<=x['time']<b]
  if not ss:continue
  E=np.array([np.array(x['anchor'])-x['wrist'] for x in ss]);F=np.array([x['force'] for x in ss]);w=np.array([x['wrist'] for x in ss]);root=np.array([x['root'] for x in ss]);feet=np.array([x['foot_positions'] for x in ss]) if 'foot_positions' in ss[0] else None
  dt=np.diff([x['time'] for x in ss]);dp=np.diff(w,axis=0);work=float(np.sum((.5*(F[:-1]+F[1:])*dp)))
  result['segments'][label]=dict(error_cm=100*float(np.sqrt((E*E).sum(1).mean())),signed_xyz_cm=(100*E.mean(0)).tolist(),force_rms=float(np.sqrt((F*F).sum(1).mean())),force_xyz_rms=np.sqrt((F*F).mean(0)).tolist(),cap_fraction=float((np.linalg.norm(F,axis=1)>=cap-.001).mean()),root_delta_cm=(100*(root[-1]-root[0])).tolist(),wrist_delta_cm=(100*(w[-1]-w[0])).tolist(),force_work_J=work,tilt_max=max(x['tilt_deg'] for x in ss),min_height=min(x['height'] for x in ss),no_support_fraction=float(np.mean([not any(x['support']) for x in ss])),qpd_peak_deg=float(np.rad2deg(max([x['q_PD_step_max_rad'] for x in cc],default=0))))
 rows.append(result)
parity=[]
for name,source in [('B505','A-prepare-03'),('N505','C-final-05'),('N506','C-final-05')]:
 if name+'_cap5' not in episodes:continue
 with gzip.open(R/source/'job'/(name+'_stop_unseen_late.json.gz'),'rt') as f:old=json.load(f)
 new=episodes[name+'_cap5'];parity.append(dict(model=name,shapes=[len(old['physics']),len(new['physics'])],qpos_max_diff=float(np.abs(np.array([x['qpos'] for x in old['physics']])-np.array([x['qpos'] for x in new['physics']])).max())))
(out/'force-cap-summary.json').write_text(json.dumps(dict(rows=rows,parity=parity),indent=2))
for r in rows:print(r['label'],'error/force/cap',*[round(r[x],3) for x in ['error_cm','force_rms','cap_fraction']],'walktime',r['first_sustained_forward_005_mps'],'hold',round(r['segments']['hold']['error_cm'],3),'rootmove',round(r['segments']['move']['root_delta_cm'][0],2),'recover',round(r['segments']['recover']['root_delta_cm'][0],2))
print('PARITY',parity)
