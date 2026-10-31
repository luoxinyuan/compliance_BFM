"""Publication-independent evidence plot of fixed M2 saved records."""
import gzip,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m2-unseen-body-20261031A')
def main():
 fig,ax=plt.subplots(3,2,figsize=(13,10),sharex=True);axes=ax.ravel()
 labels={'native_hold':'Native pose hold','teacher':'Privileged teacher','rule':'Continuous proprio rule','distill':'Frozen projected-z','alignment':'Frozen action alignment'}
 colors=['#888888','#6a3d9a','#e69f00','#0072b2','#d55e00']
 for (name,label),color in zip(labels.items(),colors):
  p=R/f'comparison-attempt-01/outputs/midpoint__{name}.json.gz'
  if not p.exists():continue
  e=json.loads(gzip.decompress(p.read_bytes()));rs=e['physics'];cs=e['control'];t=np.array([r['time'] for r in rs]);ct=np.array([c['time'] for c in cs]);active=np.array([r['active'] for r in rs]);force=np.array([r['force'] for r in rs]);err=np.array([r['error'] for r in rs])*100;err[~active]=np.nan;w=np.array([r['wrist'] for r in rs]);v=np.linalg.norm(np.array([r['qvel'][:3] for r in rs]),axis=-1);T0=np.array(e['initial']['torso_matrix']);Ts=np.array([r['torso_matrix'] for r in rs]);ang=np.rad2deg(np.arccos(np.clip((np.einsum('nij,ij->n',Ts,T0)-1)/2,-1,1)))
  curves=[(t,err),(t,np.linalg.norm(force,axis=-1)),(t,(w[:,0]-e['initial']['wrist'][0])*100),(t,v),(t,ang),(ct,np.rad2deg([c['q_PD_step_max_rad'] for c in cs]))]
  for a,(xx,yy) in zip(axes,curves):a.plot(xx,yy,color=color,label=label,lw=1.3)
 titles=['Active-contact 3D error (cm)','Actual force norm (N)','Wrist world X from contact start (cm)','Root speed (m/s)','Torso rotation from contact start (deg)','Largest joint target step / 20 ms (deg)']
 for a,title in zip(axes,titles):
  a.set_title(title);a.axvline(12.4,color='k',ls=':',lw=.8);a.axvline(17.4,color='k',ls='--',lw=.8);a.axvspan(16.4,17.4,color='grey',alpha=.12);a.set_xlim(10.4,23.4);a.grid(alpha=.2)
 axes[1].axhline(5,color='k',lw=.5,ls=':');axes[3].axhline(.02,color='grey',lw=.6,ls=':',label='Historical recovery reference (not safety veto)')
 for a in axes[-2:]:a.set_xlabel('Physical time (s); ready 10.4, contact 12.4, release 17.4')
 handles,ls=axes[1].get_legend_handles_labels();fig.legend(handles,ls,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.975));fig.suptitle('M2: frozen policies, naturally entered u=0.5, real +X 15 cm contact',y=.998);fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(R/'analysis/m2-physical-evidence.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
