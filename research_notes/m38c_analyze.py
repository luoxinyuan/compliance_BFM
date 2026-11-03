"""Time-matched physical contact contrasts and first-fork evidence."""
import json,gzip,sys
from pathlib import Path
import numpy as np
import m38b_analyze as old
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m38c-history-20261103')
old.R=R
read=old.read

def curve(path):
 e=read(path);rr=e['physics'];initial=e['initial'];s=e['condition'];out=[]
 for t in sorted(set([3.,4.2,4.5,5.,5.5,6.,7.,8.,10.,12.05,14.,14.05,16.,19.,s['onset'],s['onset']+s['ramp'],s['release'],s['end']-.005])):
  r=min(rr,key=lambda r:abs(r['force_time']-t));T=np.rad2deg(old.wrap(old.yaw(r,'torso_matrix')-old.yaw(initial,'torso_matrix')));P=np.rad2deg(old.wrap(old.yaw(r,'pelvis_matrix')-old.yaw(initial,'pelvis_matrix')))
  out.append(dict(time=r['force_time'],torso_deg=float(T),pelvis_deg=float(P),waist_relative_deg=float(old.wrap(np.deg2rad(T-P))*180/np.pi),root_delta_cm=(100*(np.array(r['root'])-initial['root'])).tolist(),active=r['active'],force_xyz=[c['force'] for c in r['contacts']],moment_com=r['net_yaw_moment_com'],sum_force=r['sum_force_norm']))
 return out

def forks(a,b):
 e,f=read(a),read(b);rows=[]
 for i,(x,y) in enumerate(zip(e['decisions'],f['decisions'])):
  cc=e['control'][5*i:5*i+5];dd=f['control'][5*i:5*i+5];xx=np.array(x['obs']);yy=np.array(y['obs']);p=e['physics'][min(i*20,len(e['physics'])-1)];q=f['physics'][min(i*20,len(f['physics'])-1)]
  angle=lambda a,b:float(np.degrees(np.arccos(np.clip(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))
  obs={k:float(np.sqrt(np.mean((np.array(cc[0]['bfm_obs'][k])-dd[0]['bfm_obs'][k])**2))) for k in cc[0]['bfm_obs']}
  rows.append(dict(time=cc[0]['time'],previous_z_angle=angle(xx[565:821],yy[565:821]),privileged_extra_rms=float(np.sqrt(np.mean((xx[825:]-yy[825:])**2))),bfm_obs_rms=obs,qpd_five_rms_deg=float(np.degrees(np.sqrt(np.mean((np.array([c['q_PD'] for c in cc])-np.array([c['q_PD'] for c in dd]))**2)))),actual_q_rms_deg=float(np.degrees(np.sqrt(np.mean((np.array(p['qpos'][7:])-q['qpos'][7:])**2)))),dq_rms=float(np.sqrt(np.mean((np.array(p['qvel'][6:])-q['qvel'][6:])**2))),actual_force_xyz=[c['force'] for c in p['contacts']],source_force_xyz=[c['force'] for c in q['contacts']],root_difference_cm=float(100*np.linalg.norm(np.array(p['root'])-q['root']))))
 first={}
 for key,thr in [('previous_z_angle',2),('qpd_five_rms_deg',.2),('actual_q_rms_deg',.2),('root_difference_cm',.1)]:
  first[key]=next((r['time'] for j,r in enumerate(rows[:-2]) if all(v[key]>thr for v in rows[j:j+3])),None)
 return dict(first_sustained_3decisions=first,thresholds_are_descriptive_not_success_gates=True,rows=rows)

def main():
 old.main();dest=Path(sys.argv[1]);allcurves={};allforks={};contrasts=[]
 for p in sorted(R.glob('*/job/summaries.json')):
  names={r['label']:r for r in read(p)}
  for name in names:allcurves[name]=curve(p.parent/(name+'.json.gz'))
  for model in ['C2','T30','source','D0','D1','D2']:
   for sign in ['left','right']:
    k=model+'_pair_'+sign;off=model+'_'+sign+'_noforce';zero=model+'_zero'
    if k in allcurves and off in allcurves:
     for a,b in zip(allcurves[k],allcurves[off]):
      contrasts.append(dict(model=model,direction=sign,time=a['time'],contact_torso=a['torso_deg'],no_force_torso=b['torso_deg'],increment_torso=a['torso_deg']-b['torso_deg'],contact_waist=a['waist_relative_deg'],increment_waist=a['waist_relative_deg']-b['waist_relative_deg'],contact_pelvis=a['pelvis_deg'],increment_pelvis=a['pelvis_deg']-b['pelvis_deg'],force_xyz=a['force_xyz'],moment_com=a['moment_com']))
  if 'C2_pair_left' in names and 'source_pair_left' in names:allforks['C2_vs_source_left']=forks(p.parent/'C2_pair_left.json.gz',p.parent/'source_pair_left.json.gz')
 for name,obj in [('time-curves',allcurves),('contact-contrasts',contrasts),('first-forks',allforks)]: (dest/(name+'.json')).write_text(json.dumps(obj,indent=2,allow_nan=False))
if __name__=='__main__':main()
