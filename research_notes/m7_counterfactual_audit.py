"""Posthoc physical partner counterfactual; fixed model, no fitting or simulation."""
import json,os
from pathlib import Path
import numpy as np
from m7_analysis import RUN,read,summarize_transition

def main():
 a=read(RUN/'final-attempt-01/outputs/sealed_mid__distill.json.gz')
 b=read(RUN/'counterfactual-attempt-01/outputs/posthoc_mid_hold__distill.json.gz')
 fork=a['condition']['transition']['restart'];diff={}
 for domain,fields,clock in [('physics',['qpos','qvel','wrist','force','anchor'],'force_time'),('control',['student_input','executed_z','q_PD'],'time')]:
  aa=[r for r in a[domain] if r[clock]<fork-1e-8];bb=[r for r in b[domain] if r[clock]<fork-1e-8]
  assert len(aa)==len(bb)
  for field in fields:diff[domain+'/'+field]=float(abs(np.array([r[field] for r in aa])-np.array([r[field] for r in bb])).max())
 assert max(diff.values())==0,diff
 out=dict(fork=fork,prefix_max_difference=diff,interpretation='Posthoc physical partner intervention; unchanged checkpoint, not blind generalization and no training',reverse=summarize_transition(a),hold=summarize_transition(b))
 (RUN/'analysis/counterfactual.json').write_text(json.dumps(out,indent=2,allow_nan=False))
 summary=RUN/'analysis/all-summary.json'
 if summary.exists():
  combined=read(summary);entry=dict(out['hold'],source=str(RUN/'counterfactual-attempt-01/outputs/posthoc_mid_hold__distill.json.gz'));combined['counterfactual-attempt-01/posthoc_mid_hold__distill']=entry
  summary.write_text(json.dumps(combined,indent=2,allow_nan=False))
 os.environ['MPLCONFIGDIR']=str(RUN/'analysis/mpl-cache')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(4,1,figsize=(10,12),sharex=True)
 for e,label,color in [(a,'Anchor reverses','teal'),(b,'Anchor remains forward','darkorange')]:
  rows=e['physics'];cs=e['control'];t=np.array([r['force_time'] for r in rows]);ct=np.array([c['time'] for c in cs]);origin=e['initial']['wrist'][0]
  axes[0].plot(t,[c['force'][0] for c in rows],label=label,color=color)
  axes[1].plot(t,[(r['wrist'][0]-origin)*100 for r in rows],color=color)
  axes[1].plot(t,[(r['anchor'][0]-origin)*100 if r['active'] else np.nan for r in rows],ls=':',color=color)
  axes[2].plot(ct,[c['native_coords'][0] for c in cs],color=color)
  axes[3].plot(t,np.linalg.norm(np.array([r['qvel'][:3] for r in rows]),axis=1),color=color)
 for ax,l in zip(axes,['Actual Fx (N)','Wrist / dotted anchor X (cm)','Executed native progress','Root speed (m/s)']):
  ax.set_ylabel(l);ax.axvline(fork,ls='--',color='black');ax.axvline(a['condition']['release'],ls=':',color='grey');ax.grid(alpha=.2)
 axes[0].legend();axes[-1].set_xlabel('Physical episode time (s)');fig.suptitle('M7 posthoc: identical prefix, different real partner demand, same frozen PZ')
 fig.tight_layout();fig.savefig(RUN/'analysis/m7-partner-counterfactual.png',dpi=140)
 for k in ['reverse','hold']:
  v=out[k]['transition']['reverse'];print(k,{s:v[s] for s in ['error_cm','force_rms_n','force_xyz_mean','alpha_first','alpha_last','wrist_delta_xyz_cm','root_delta_xyz_cm']})
 print('EXACT_MATCHED_PREFIX',diff)
if __name__=='__main__':main()
