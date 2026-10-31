"""Final-only diagnostic labels never enter training. Actual qPD and physical response timing."""
import json
from pathlib import Path
import numpy as np
from m7_analysis import RUN,read

def main():
 out={}
 for a in read(RUN/'compute-ledger.json')['attempts']:
  if a['stage']!='final' or a.get('result_status')!='completed':continue
  for p in sorted((Path(a['path'])/'outputs').glob('*__*.json.gz')):
   e=read(p);cs=e['control'];s=e['condition'];tr=s.get('transition');ts=np.array([c['time'] for c in cs]);q=np.array([c['q_PD'] for c in cs]);dq=np.diff(q,axis=0);i,j=np.unravel_index(abs(dq).argmax(),dq.shape)
   v=dict(source=str(p),peak_target=dict(time=cs[i+1]['time'],joint=int(j),step_deg=float(np.rad2deg(dq[i,j])),next20ms_actual_deg=float(np.rad2deg(cs[min(i+2,len(cs)-1)]['pre_qpos'][7+j]-cs[i+1]['pre_qpos'][7+j]))),phase_feedback={})
   if tr:
    alpha=np.array([c['native_coords'][0] if c['native_coords'] is not None else c['alpha'] if e['controller']=='teacher' else c['rule_record']['alpha'] if e['controller'] in ['force_rule','signed_rule'] else 0. for c in cs]);ix=lambda t:min(len(ts)-1,int(np.searchsorted(ts,t-1e-8)))
    v['timing']=dict(alpha_at_unload=float(alpha[ix(tr['unload'])]),alpha_at_restart=float(alpha[ix(tr['restart'])]),first_alpha_below_half_after_unload=next((c['time'] for c,x in zip(cs,alpha) if c['time']>=tr['unload']-1e-8 and x<.5),None),first_negative_force_after_restart=next((r['force_time'] for r in e['physics'] if r['force_time']>=tr['restart']-1e-8 and r['active'] and r['force'][0]<-.1),None))
   if cs and 'oracle_q_PD' in cs[0]:
    target=np.array([c['oracle_q_PD'] for c in cs]);base=np.array([c['oracle_base_q_PD'] for c in cs]);err=q-target;desired=target-base;student=q-base
    spans={'all':(3,s['end']),'recovery':(s['release'],s['end'])}
    if tr:spans.update(forward=(s['onset'],tr['unload']),reverse=(tr['restart'],s['release']),reverse_hold=(tr['reverse_end'],s['release']))
    v['query_valid_fraction']=float(np.mean([c['oracle_query_valid'] for c in cs]))
    for name,(lo,hi) in spans.items():
     mask=(ts>=lo-1e-8)&(ts<hi-1e-8);d=desired[mask,22:];st=student[mask,22:];energy=float((d*d).sum())
     v['phase_feedback'][name]=dict(arm_rms_deg=float(np.rad2deg(np.sqrt(np.mean(err[mask,22:]**2)))),nonarm_rms_deg=float(np.rad2deg(np.sqrt(np.mean(err[mask,:22]**2)))),expert_local_increment_rms_deg=float(np.rad2deg(np.sqrt(np.mean(d*d)))),feedback_projection_gain=float((st*d).sum()/energy) if energy>1e-12 else None)
   out[e['label']]=v
 (RUN/'analysis/feedback-audit.json').write_text(json.dumps(out,indent=2,allow_nan=False))
 print('FINAL_SHADOW_LABEL_AUDIT',len(out),'records; no optimization')
if __name__=='__main__':main()
