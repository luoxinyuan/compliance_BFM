"""Saved trajectory comparisons only; never constructs MuJoCo or steps a policy."""
import os,time,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,numpy as np
T=time.monotonic();torch.set_num_threads(2)
P=Path(os.environ['COMPLIANCE_PROJECT_ROOT']);S=P/'experiments/adapter-return-pilot-20261113';O=Path(__file__).parent/'data'
out=[];firstfive=[];pairs=[]
for case in (2,3):
 traces={n:torch.load(S/f'runs/evaluate-case-{case:02d}-01/{n}-mean/trace.pt',map_location='cpu',weights_only=False) for n in ['V1','control','return']}
 v=traces['V1'];vphys=v['physics']
 ac=traces['control']['actions'][0]['audit'];ar=traces['return']['actions'][0]['audit']
 pairs.append(dict(case=case,exact_equal_action_components=int((ac['executed']==ar['executed']).sum()),qPD_rms=float((ac['qexec']-ar['qexec']).square().mean().sqrt())))
 for name,t in traces.items():
  for i in [0,4,14,49,149]:
   a=t['actions'][i];va=v['actions'][i];p=t['physics'][i*4+3];vp=vphys[i*4+3]
   ba=a['audit'];bva=va['audit'];q=np.array(p['qpos'][7:]);vq=np.array(vp['qpos'][7:]);pd=np.array(t['controls'][i]['q_PD'])
   out.append(dict(case=case,model=name,elapsed_end_s=(i+1)*.02,actual_q_rms_vsV1=float(np.mean((q-vq)**2)**.5),targetq_rms_vsV1=float(np.mean((pd-v['controls'][i]['q_PD'])**2)**.5),base_qPD_rms_vsV1=float((ba['qbase']-bva['qbase']).square().mean().sqrt()),z_angle_deg=(0. if torch.equal(a['z'],va['z']) else float(torch.rad2deg(torch.acos(((a['z'].double()*va['z'].double()).sum()/(a['z'].double().norm()*va['z'].double().norm())).clamp(-1,1))))),root_speed=float(np.linalg.norm(p['qvel'][:3])),slip=float(p['stage_loaded_slip_mps']),yaw_deg=float(np.rad2deg(np.arctan2(np.array(p['pelvis_matrix'])[1,0],np.array(p['pelvis_matrix'])[0,0])))))
  firstfive.append(dict(case=case,model=name,contract='Actual divergent closed loops after common initial state, not teacher-forced states',qPD=[c['q_PD'] for c in t['controls'][:5]],actual_q=[p['qpos'][7:] for p in t['physics'][:20]],root_velocity=[p['qvel'][:3] for p in t['physics'][:20]]))
(O/'saved-first-action-pair.json').write_text(json.dumps(pairs,indent=2))
(O/'temporal-summary.json').write_text(json.dumps(out,indent=2));(O/'first-five-controls.json').write_text(json.dumps(firstfive,indent=2))
(O/'temporal-receipt.json').write_text(json.dumps(dict(wall_s=time.monotonic()-T,new_physics=0,new_updates=0,cuda_initialized=torch.cuda.is_initialized()),indent=2))
