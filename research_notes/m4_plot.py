"""Plot saved M4 real physics; no model/simulation."""
import gzip,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs');RUN=R/'m4-continuous-feedback-20261031A'
def read(p):return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
def main():
 out=RUN/'analysis';s=read(out/'physical-summary.json');fig,ax=plt.subplots(3,2,figsize=(13,10),sharex=True)
 for ctl in ['native_hold','teacher','local_teacher','rule','m3_student','distill']:
  key='u250__'+ctl
  if key not in s['episodes']:continue
  e=read(Path(s['episodes'][key]['source']));rr=e['physics'];t=np.array([r['time'] for r in rr]);f=np.array([r['force'] for r in rr]);w=np.array([r['wrist'] for r in rr]);root=np.array([r['root'] for r in rr]);err=np.linalg.norm(np.array([r['anchor'] for r in rr])-w,axis=1)*100;active=np.array([r['active'] for r in rr]);err[~active]=np.nan;start=np.argmin(abs(t-12.4));q=np.array([c['q_PD_step_max_rad'] for c in e['control']]);ct=[c['time'] for c in e['control']]
  for a,y,title in [(ax[0,0],err,'Active 3D error [cm]'),(ax[0,1],np.linalg.norm(f,axis=1),'True force [N]'),(ax[1,0],(w[:,0]-w[start,0])*100,'Wrist X from onset [cm]'),(ax[1,1],(w[:,2]-w[start,2])*100,'Wrist Z from onset [cm]'),(ax[2,0],(root[:,0]-root[start,0])*100,'Root X from onset [cm]')]:a.plot(t,y,label=ctl,lw=1);a.set_title(title)
  ax[2,1].plot(ct,np.degrees(q),label=ctl,lw=1);ax[2,1].set_title('q_PD largest per-tick change [deg]')
 for a in ax.flat:a.axvline(12.4,color='gray',ls=':');a.axvline(19.4,color='gray',ls=':');a.set_xlim(10.4,27.4);a.grid(alpha=.2)
 ax[0,0].legend(fontsize=8);fig.suptitle('M4 continuous forward guidance, stop and full recovery (frozen controllers)');fig.tight_layout();fig.savefig(out/'continuous-response.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots(3,3,figsize=(14,10),sharex=True);old=read(R/'m3-continuous-body-20261031A/final-attempt-01/outputs/u500__m3_student.json.gz')
 for row,st in enumerate([13.2,13.6,14.4]):
  episodes=[('M3 continuation',old)]
  for ctl in ['recover_teacher','recover_projected']:
   key=f'u500__{ctl}__at{st:.1f}'
   if key in s['episodes']:episodes.append((ctl,read(Path(s['episodes'][key]['source']))))
  for name,e in episodes:
   rr=e['physics'];t=np.array([r['time'] for r in rr]);w=np.array([r['wrist'] for r in rr]);err=np.linalg.norm(np.array([r['anchor'] for r in rr])-w,axis=1)*100;err[~np.array([r['active'] for r in rr])]=np.nan
   f=np.array([r['force'] for r in rr]);start=np.argmin(abs(t-12.4))
   for a,y,title in [(ax[row,0],err,f'Switch {st:.1f}s: active 3D error [cm]'),(ax[row,1],(w[:,2]-w[start,2])*100,'Wrist Z from onset [cm]'),(ax[row,2],np.linalg.norm(f,axis=1),'True force [N]')]:a.plot(t,y,label=name,lw=1);a.set_title(title);a.axvline(st,color='gray',ls=':');a.axvline(17.4,color='gray',ls=':');a.set_xlim(12.4,25.4);a.grid(alpha=.2)
 ax[0,0].legend(fontsize=8);fig.suptitle('M4 exact matched-state expert and legal-latent recovery (not Student training)');fig.tight_layout();fig.savefig(out/'real-recovery.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
