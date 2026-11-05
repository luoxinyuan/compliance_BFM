import json,pathlib,numpy as np
r=pathlib.Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/feedback-coexistence-20261105');allrows=json.loads((r/'physical-metrics.json').read_text());rows=[x for x in allrows if '/final-' in x['path']];out={}
for name in ['I','C0','seed4601_last','seed4602_last']:
 print('\n',name)
 for task in ['holdout_local6','holdout_long72','holdout_stop','early','no_contact']:
  x=next(x for x in rows if x['label']==name+'__'+task);c=x['contact'];print(task,'qpd',round(x['qpd_max_step_deg'],2))
  if c:print('contact',round(c['error_cm'],3),'forceXYZ',np.round(c['force_xyz_rms'],2).tolist(),'rootX',round(c['root_delta_cm'][0],2),'relwristX',round(c['relative_wrist_shoulder_delta_cm'][0],2),'footpath',np.round(c['foot_xy_path_cm'],2).tolist(),'slip',round(c['slip_peak'],3))
  for k in ['move','decelerate','hold','recover']:
   v=x['phase'].get(k)
   if v:print(k,'err',round(v['error_cm'],2) if v['error_cm'] is not None else None,'force',round(v['force_rms'],2),'rootX',round(v['root_delta_cm'][0],2),'speedfinal',round(v['root_final_speed'],3))
print('maxquality',dict(episodes=len(allrows),boundary=sum(bool(x['boundary']) for x in allrows),support_loss=sum(v['support_loss_ticks'] for x in allrows for v in x['phase'].values() if v),qpd_max=max(x['qpd_max_step_deg'] for x in allrows),qmax=max(v['actual_q_max_step_deg'] for x in allrows for v in x['phase'].values() if v),tilt=max(v['max_tilt'] for x in allrows for v in x['phase'].values() if v)))
# Compact auditable final aggregation, not trajectories.
(r/'final-physical-summary.json').write_text(json.dumps(rows,indent=2))
