"""Saved-data regression localization and feature-coverage distances. No model forward or fit."""
import gzip,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m3-continuous-body-20261031A')
def read(p):return json.loads(gzip.decompress(p.read_bytes()))
def main():
 d=torch.load(R/'train-attempt-01/outputs/dataset.pt',map_location='cpu',weights_only=True);state=torch.load(R/'train-attempt-01/outputs/projected-z-final.pt',map_location='cpu',weights_only=True)['state_dict'];scale=state['scale'].numpy();h=d['h'].numpy();records=[]
 for u,times in [('500',[12.4,13.2,13.6,14.4,15.,19.4,23.9]),('625',[13.15,15.15,17.95])]:
  e=read(R/'final-attempt-01/outputs'/f'u{u}__m3_student.json.gz');cs=e['control'];rows=e['physics']
  for t in times:
   c=min(cs,key=lambda c:abs(c['time']-t));p=min(rows,key=lambda r:abs(r['time']-t));v=np.asarray(c['student_input']);dx=(h-v)/scale;dist=np.sqrt(np.mean(dx*dx,axis=-1));ix=np.argsort(dist)[:5]
   records.append(dict(u=u,time=c['time'],coords=c['native_coords'],force_xyz=c['measured_force_xyz'],wrist_delta_xyz_cm=((np.asarray(p['wrist'])-e['initial']['wrist'])*100).tolist(),query_valid=c['oracle_query_valid'],student_query_arm_rms_deg=float(np.degrees(np.sqrt(np.mean((np.asarray(c['q_PD'])[22:]-c['oracle_q_PD'][22:])**2)))),nearest=[dict(index=int(i),metadata=d['metadata'][i],standardized_input_rms_distance=float(dist[i]),proprio_rms_distance=float(np.sqrt(np.mean(dx[i,:465]**2))),force_history_rms_distance=float(np.sqrt(np.mean(dx[i,465:]**2)))) for i in ix]))
 e=read(R/'final-attempt-01/outputs/u500__m3_student.json.gz');rows=e['physics'];on=e['condition']['onset'];rel=e['condition']['release'];z0=e['initial']['wrist'][2]
 first=next(r['time'] for r in rows if r['time']>=on and r['wrist'][2]<z0-.05)
 result=dict(records=records,first_wrist_z_drop_5cm_time=first,threshold_is_locator_not_changed_test_criterion=True,scope='CPU saved-array statistics only, no BFM/Student forward, no fitting. Distances do not prove non-identifiability or establish a universal OOD threshold. No final samples enter training.')
 (R/'analysis/regression-diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
 sources=[('M3 contact',R/'final-attempt-01/outputs/u500__m3_student.json.gz'),('M3 no contact',R/'final-attempt-01/outputs/u500__m3_student__free.json.gz'),('Selected teacher',R/'qualification-attempt-01/outputs/u500__local_teacher.json.gz'),('Native contact',R/'qualification-attempt-01/outputs/u500__native_hold.json.gz')]
 for label,path in sources:
  x=read(path);rr=x['physics'];t=np.array([r['time'] for r in rr]);w=np.array([r['wrist'] for r in rr]);active=np.array([r['active'] for r in rr]);mask=t>=10.4;zref=w[np.argmin(abs(t-on)),2]
  ax[0,0].plot(t[mask],(w[mask,2]-zref)*100,label=label)
  if active.any():ax[0,1].plot(t[active],np.linalg.norm(np.array([r['anchor'] for r in rr])-w,axis=-1)[active]*100,label=label)
 t=np.array([r['time'] for r in rows]);f=np.array([r['force'] for r in rows]);mask=t>=10.4
 for i,n in enumerate('XYZ'):ax[1,0].plot(t[mask],f[mask,i],label='F'+n)
 cs=[c for c in e['control'] if c['native_coords'] is not None]
 for i,n in enumerate(['u','a','r']):ax[1,1].plot([c['time'] for c in cs],[c['native_coords'][i] for c in cs],label=n)
 for a,title,y in zip(ax.flat,['Known u=.5 regression: actual wrist','Active contact error','Actual 3D force','Student command coordinates'],['Wrist Z relative onset (cm)','3D error (cm)','Force (N)','Native coordinate']):
  a.set(title=title,xlabel='Physical time (s)',ylabel=y);a.axvline(on,ls=':',color='black');a.axvline(rel,ls='--',color='black');a.grid(alpha=.2);a.legend(fontsize=8)
 fig.suptitle('Preserved M3 failure: no retraining or test selection');fig.savefig(R/'analysis/known-midpoint-failure.png',dpi=170);plt.close(fig)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
