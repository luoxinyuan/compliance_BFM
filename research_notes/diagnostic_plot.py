"""Small actual-trajectory figure, not reference/q_PD animation."""
import gzip,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(folder,out):
 files=[('Initialization','3ffd2fc10497-i0.json.gz'),('Batch48','3ffd2fc10497-i48.json.gz'),
  ('Init to48 at5.3s','3ffd2fc10497-from0-at53-sustained.json.gz'),('48 toInit at5.3s','3ffd2fc10497-from1-at53-sustained.json.gz')]
 fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
 for name,file in files:
  d=json.load(gzip.open(folder/file,'rt'));r=d['physics'];t=np.array([v['force_time'] for v in r]);active=np.array([v['active'] for v in r]);start=r[0]['root'][0]
  ax[0,0].plot(t,100*(np.array([v['root'][0] for v in r])-start),label=name)
  ax[0,1].plot(t,np.where(active,100*np.array([v['error'] for v in r]),np.nan),label=name)
  ax[1,0].plot(t,np.where(active,np.linalg.norm([v['force'] for v in r],axis=1),np.nan),label=name)
  ax[1,1].plot(t,[v['stage_loaded_slip_mps'] for v in r],label=name)
 for a,title,y in [(ax[0,0],'Real forward root displacement','cm'),(ax[0,1],'Active-contact 3D error','cm'),(ax[1,0],'Actual contact force','N'),(ax[1,1],'Loaded-contact slip','m/s')]:
  a.set(title=title,xlabel='Physical time (s)',ylabel=y);a.axvline(5.2,color='grey',linestyle=':',linewidth=1);a.axvline(10.069369,color='grey',linestyle='--',linewidth=1);a.grid(alpha=.2)
 ax[0,0].legend(fontsize=8);fig.suptitle('Issue57: exposed62cm condition; complete-prefix physics handoffs')
 fig.savefig(out,dpi=150);plt.close(fig)
if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
