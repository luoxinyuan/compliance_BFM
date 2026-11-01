"""Final stage comparison and genuine extra body response, from saved physics only."""
import json
import numpy as np
from m8_analysis import RUN,read,analyze
from pathlib import Path

def main():
 ledger=read(RUN/'compute-ledger.json');attempt=[a for a in ledger['attempts'] if a['stage']=='final' and a.get('result_status')=='completed'];assert len(attempt)==1;out=Path(attempt[0]['path'])/'outputs';summary={};episodes={}
 for p in sorted(out.glob('*__*.json.gz')):
  e=read(p);episodes[e['label']]=e;summary[e['label']]=analyze(e)
 comparisons={};free={}
 for label,e in episodes.items():
  name,ctl=label.split('__');s=e['condition'];ref=episodes.get(name+'__standing',episodes.get(name+'__m7'))
  if ref is None:continue
  r=e['physics'];b=ref['physics'];assert len(r)==len(b);active=np.array([x['active'] for x in r]);q=np.array([x['qpos'] for x in r]);qb=np.array([x['qpos'] for x in b]);c=e['control']
  if not active.any():
   free[label]={k:float(abs(np.array([x[k] for x in r])-np.array([x[k] for x in b])).max()) for k in ['qpos','qvel','wrist','force']};continue
  extra={};
  for field in ['wrist','shoulder_xyz','root']:
   d=np.array([x[field] for x in r])-np.array([x[field] for x in b]);extra[field+'_mean_extra_cm']=(100*d[active].mean(0)).tolist();extra[field+'_end_extra_cm']=(100*d[np.flatnonzero(active)[-1]]).tolist()
  extra['knee_mean_extra_deg']=np.degrees((q[active][:,[10,16]]-qb[active][:,[10,16]]).mean(0)).tolist();extra['hip_mean_extra_deg']=np.degrees((q[active][:,[7,13]]-qb[active][:,[7,13]]).mean(0)).tolist();extra['waist_mean_extra_deg']=np.degrees((q[active,19:22]-qb[active,19:22]).mean(0)).tolist()
  a=summary[label]['phases']['all_contact'];br=summary[ref['label']]['phases']['all_contact'];extra['error_reduction_percent']=100*(1-a['tracking_3d_rms_cm']/br['tracking_3d_rms_cm']);extra['force_reduction_percent']=100*(1-a['force_rms_n']/br['force_rms_n'])
  cc=[x for x in c if s['onset']<=x['time']<s['release']];err=np.array([np.array(x['q_PD'])-x['oracle_q_PD'] for x in cc]);extra['student_expert_qpd_rms_deg']=float(np.degrees(np.sqrt(np.mean(err**2))));extra['leg_waist_qpd_rms_deg']=float(np.degrees(np.sqrt(np.mean(err[:,:15]**2))));extra['arm_qpd_rms_deg']=float(np.degrees(np.sqrt(np.mean(err[:,22:]**2))))
  comparisons[label]=extra
 result=dict(summary=summary,paired_extra_response=comparisons,no_contact_native_parity=free,scope='One seed, fixed standing start, one maintained attachment per episode; unseen amount/speed/local-offset combinations, not statistical/general-body or unilateral-contact qualification.')
 (RUN/'analysis/final-comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:dict(error=v['phases'].get('all_contact',{}).get('tracking_3d_rms_cm'),force=v['phases'].get('all_contact',{}).get('force_rms_n'),qpd=v['qpd_step_max_deg']) for k,v in summary.items()},indent=2))
if __name__=='__main__':main()
