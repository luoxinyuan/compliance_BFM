"""Plot preserved final traces; no model loading."""
import gzip,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m3-continuous-body-20261031A')
S=R/'final-attempt-01/outputs';O=R/'analysis'
def read(c,free=False):return json.loads(gzip.decompress(((R/'contact_ablation-attempt-01/outputs' if c.endswith('_z_replay') else S)/('u625__'+c+('__free' if free else '')+'.json.gz')).read_bytes()))
names=dict(native_hold='Native hold',teacher='Goal teacher',local_teacher='Pose-base teacher',rule='Proprio rule',distill='M1 projected-z',alignment='M1 alignment',m3_student='M3 projected-z',m3_student_z_replay='M3 no-contact z replay')
colors=dict(zip(names,['#777777','#009E73','#56B4E9','#E69F00','#CC79A7','#0072B2','#D55E00','#000000']))
fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for c in names:
 e=read(c);rows=e['physics'];t=np.array([r['time'] for r in rows]);mask=np.array([r['active'] for r in rows]);w=np.array([r['wrist'] for r in rows]);anchor=np.array([r['anchor'] for r in rows]);force=np.linalg.norm([r['force'] for r in rows],axis=-1);rel=e['condition']['release'];on=e['condition']['onset'];recovery=(t>=rel)&(t<=rel+8)
 kw=dict(color=colors[c],label=names[c],lw=1.7 if c=='m3_student' else 1,alpha=1 if c=='m3_student' else .85)
 ax[0,0].plot(t[mask]-on,np.linalg.norm(anchor-w,axis=-1)[mask]*100,**kw);ax[0,1].plot(t[mask]-on,force[mask],**kw)
 ax[1,0].plot(t[recovery]-rel,np.linalg.norm([r['qvel'][:3] for r in rows],axis=-1)[recovery],**kw)
 ax[1,1].plot(t[recovery]-rel,(w[recovery,2]-w[np.argmin(abs(t-rel)),2])*100,**kw)
for a,title,x,y in zip(ax.flat,['Real contact tracking','Real force','Complete recovery','Wrist vertical recovery'],['Time since contact (s)']*2+['Time since release (s)']*2,['3D error (cm)','Force (N)','Root speed (m/s)','Wrist Z change (cm)']):a.set(title=title,xlabel=x,ylabel=y);a.grid(alpha=.2)
ax[0,0].legend(fontsize=8,ncol=2);fig.suptitle('M3 sealed u=0.625, onset=13.15 s; same frozen BFM and physical protocol');fig.savefig(O/'final-contact-recovery.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(3,2,figsize=(12,10),layout='constrained');base=read('native_hold',True);bw=np.array([r['wrist'] for r in base['physics']]);br=np.array([r['root'] for r in base['physics']]);rd=base['control_ready_s']
for c in ['native_hold','rule','distill','alignment','m3_student']:
 e=read(c,True);rows=e['physics'];t=np.array([r['time'] for r in rows]);mask=t>=rd;w=np.array([r['wrist'] for r in rows]);r=np.array([r['root'] for r in rows]);kw=dict(color=colors[c],label=names[c],lw=1.7 if c=='m3_student' else 1)
 ax[0,0].plot(t[mask]-rd,np.linalg.norm(w-bw,axis=-1)[mask]*100,**kw);ax[0,1].plot(t[mask]-rd,np.linalg.norm(r-br,axis=-1)[mask]*100,**kw)
 cs=[c for c in e['control'] if c['time']>=rd and c['native_coords'] is not None]
 if cs:
  ax[1,0].plot([c['time']-rd for c in cs],[c['native_coords'][0] for c in cs],**kw);ax[1,1].plot([c['time']-rd for c in cs],[c['native_coords'][1] for c in cs],**kw)
  ax[2,0].plot([c['time']-rd for c in cs],[c['native_coords'][2] for c in cs],**kw)
 zz=np.asarray([c['executed_z'][0] for c in e['control']]);bz=np.asarray([c['executed_z'][0] for c in base['control']]);ct=np.asarray([c['time'] for c in e['control']]);cm=ct>=rd;angle=np.degrees(np.arccos(np.clip(np.sum(zz*bz,axis=-1)/(np.linalg.norm(zz,axis=-1)*np.linalg.norm(bz,axis=-1)),-1,1)));ax[2,1].plot(ct[cm]-rd,angle[cm],**kw)
ax[1,0].axhline(.625,color='black',ls=':',label='prepared u=.625');ax[1,1].axhline(0,color='black',ls=':')
for a,title,y in zip(ax.flat,['No-contact wrist departure from native','No-contact root departure from native','Student native pose coordinate','Student goal progress','Student return coordinate','Actual latent departure from native'],['Wrist difference (cm)','Root difference (cm)','u','a','r','Angle (degrees)']):a.set(title=title,xlabel='Time since fixed ready (s)',ylabel=y);a.grid(alpha=.2)
ax[0,0].legend(fontsize=8,ncol=2);fig.suptitle('Actual no-contact physical branches; native body remains free to move');fig.savefig(O/'final-no-contact.png',dpi=170);plt.close(fig)
