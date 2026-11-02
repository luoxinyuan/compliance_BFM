from pathlib import Path
import json,gzip,numpy as np
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/direct-transfer-20261103')
def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
a=read(R/'A-final-04/job/ppo15__student_final_single.json.gz');b=read(R/'A-final-07/job/student_ppo15.json.gz');ac=a['control'];bc=b['control'];assert len(ac)==len(bc);t=np.array([x['time'] for x in ac]);za=np.array([x['executed_z'][0] for x in ac]);zb=np.array([x['executed_z'][0] for x in bc]);diff=np.linalg.norm(za-zb,axis=1);angle=np.rad2deg(np.arccos(np.clip(np.sum(za*zb,axis=1)/(np.linalg.norm(za,axis=1)*np.linalg.norm(zb,axis=1)),-1,1)))
r=dict(pre8_z_max_L2=float(diff[t<8].max()),first_command_change_time=float(t[np.flatnonzero(diff>1e-4)[0]]),post8_command_angle_mean_deg=float(angle[t>=8].mean()),post8_command_angle_peak_deg=float(angle[t>=8].max()),normal_final_root=a['physics'][-1]['root'],stop_final_root=b['physics'][-1]['root'],normal_final_anchor=a['physics'][-1]['anchor'],stop_final_anchor=b['physics'][-1]['anchor'])
assert r['pre8_z_max_L2']==0.;(R/'analysis/stop-command-causality.json').write_text(json.dumps(r,indent=2));print(r)
