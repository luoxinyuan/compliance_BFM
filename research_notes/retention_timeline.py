from pathlib import Path
import os,json,gzip,time
S=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/ppo-retention-audit-20261109');os.environ['MPLCONFIGDIR']=str(S/'.matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=S/'runs/A0-timeline';out.mkdir(exist_ok=False);fig,axes=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
for role,folder,color in [('s5401b4','A0-eval-01','tab:blue'),('s5401b7','A0-eval-02','tab:red')]:
 with gzip.open(S/'runs'/folder/(role+'_normal72.json.gz'),'rt') as f:e=json.load(f)
 rr=e['physics'];t=np.array([r['force_time'] for r in rr]);root=np.array([r['root'] for r in rr]);F=np.array([r['force'] for r in rr]);w=np.array([np.array(r['wrist'])-r['shoulder_xyz'] for r in rr]);load=np.array([r['foot_normal_force_n'] for r in rr]);yaw=np.degrees(np.unwrap([np.arctan2(r['pelvis_matrix'][1][0],r['pelvis_matrix'][0][0]) for r in rr]))
 series=[100*(root[:,0]-root[0,0]),100*(root[:,1]-root[0,1]),F[:,1],yaw,100*w[:,0],load[:,0]]
 titles=['Root forward displacement (cm)','Root lateral displacement (cm)','Actual lateral force Y (N)','Pelvis yaw (deg)','Wrist relative to shoulder X (cm)','Left foot normal load (N)']
 for ax,y,title in zip(axes.flat,series,titles):
  ax.plot(t,y,color=color,label=role,lw=1);ax.axvline(6.585,color='gray',ls=':',lw=1);ax.axvline(e['condition']['release'],color='black',ls='--',lw=.8);ax.set(title=title,xlabel='physical time (s)');ax.grid(alpha=.2)
axes[0,0].legend();fig.suptitle('Matched retrospective real rollouts: early amplification, then lateral mismatch\nDotted: first support difference; dashed: actual release');fig.savefig(out/'retention_divergence.png',dpi=160);plt.close(fig)
(out/'manifest.json').write_text(json.dumps(dict(source='saved actual qpos/force/support, no new simulation',roles=['s5401b4','s5401b7'],source_folders=['A0-eval-01','A0-eval-02'],new_queries=0,new_RL=0),indent=2));print('timeline saved')
