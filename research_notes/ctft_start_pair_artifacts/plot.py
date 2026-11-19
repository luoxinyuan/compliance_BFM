import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
data=list(csv.DictReader((p/'wrist_curve.csv').open()));fig,axs=plt.subplots(2,1,sharex=True,figsize=(8.8,5.6),layout='constrained')
for ax,n,title in zip(axs,['rolling_continue','C32__speed0.15'],['Rolling feedback (#50)','Frozen C32']):
 r=[v for v in data if v['model']==n and float(v['time'])<=3];t=[float(v['time']) for v in r]
 for k,label,color in [('wrist','Wrist (world)','#087e86'),('rel','Wrist relative to shoulder','#bc6c25'),('root','Pelvis (world)','#415b9e')]:
  ax.plot(t,[float(v[k]) for v in r],label=label,color=color,lw=2.)
 ax.set_title(title,loc='left');ax.set_ylabel('Forward change (cm)');ax.grid(alpha=.22);ax.set_ylim(-4,44)
axs[0].axvspan(1.255,1.5,color='gray',alpha=.2);axs[0].axvline(1.505,ls=':',color='k');axs[0].axvline(2.22,ls=':',color='k')
axs[0].text(1.43,32,'Wrist retracts',ha='right',fontsize=9);axs[0].text(1.54,24,'Single support',fontsize=9);axs[0].text(2.24,34,'Exchange confirmed',fontsize=9)
axs[1].legend(loc='upper left');axs[1].set_xlabel('Seconds after partner motion onset');fig.savefig(p/'startup.png',dpi=175);plt.close(fig)
