"""Saved physical M1 records only, no model loading or rollout."""
import json,sys
from pathlib import Path
import numpy as np
from m0_analysis import summarize
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/m1-state-response-20261031A'
def analyze():
 ledger=json.loads((R/'compute-ledger.json').read_text());results={}
 for run in ledger['attempts']:
  out=Path(run['path'])/'outputs'
  for file in out.glob('*__*.json'):
   e=json.loads(file.read_text())
   if 'physics' not in e or not e['physics']:continue
   if e['boundary'] is not None:
    results[e['label']]=dict(boundary=e['boundary'],seconds=len(e['physics'])/200,source=str(file));continue
   x=summarize(e,{});x['source']=str(file);x['geometry']=e['geometry'];x['goal_candidate']=e['goal_candidate'];x['contact_state_sha256']=e['contact_state_sha256'];
   ct=e['condition'];rows=e['physics'];tt=np.asarray([r['force_time'] for r in rows]);active=np.asarray([r['active'] for r in rows]);wrist=np.asarray([r['wrist'] for r in rows]);shoulder=np.asarray([r['shoulder_xyz'] for r in rows]);L=e['geometry']['arm_chain_upper_bound_m'];origin=np.asarray(e['initial']['wrist']);center=np.asarray(e['initial']['shoulder_xyz'])
   rel=wrist-shoulder;room=np.sqrt(np.maximum(0,L*L-(rel[:,1:]**2).sum(-1)))-rel[:,0]
   x['minimum_fixed_yz_forward_sphere_room_cm']=float(room[active].min()*100) if active.any() else None
   x['actual_exceeded_initial_shoulder_sphere_cm']=float((np.linalg.norm(wrist[active]-center,axis=-1)-L).max()*100) if active.any() else None
   if active.any():
    final_ix=np.flatnonzero(active)[-1];x['hand_end_relative_to_contactstart_cm']=((wrist[final_ix]-origin)*100).tolist();x['shoulder_end_relative_to_contactstart_cm']=((shoulder[final_ix]-center)*100).tolist()
   x['initial_root_speed_mps']=float(np.linalg.norm(np.asarray(e['initial']['qvel'])[:3]));x['initial_wrist_shoulder_xyz_cm']=((origin-center)*100).tolist()
   results[e['label']]=x
 comparisons={}
 for state in ['standing','raised']:
  for base in ['standing','pose_hold']:
   k=state+'__'+base
   if k not in results or 'phases' not in results[k]:continue
   b=results[k]['phases'].get('all_contact');
   if not b:continue
   for key,e in results.items():
    if not key.startswith(state+'__') or 'phases' not in e or 'all_contact' not in e['phases']:continue
    a=e['phases']['all_contact'];comparisons[key+'_vs_'+base]=dict(error_improvement=1-a['tracking_rms_cm']/b['tracking_rms_cm'],force_improvement=1-a['force_rms_n']/b['force_rms_n'])
 dest=R/'analysis';dest.mkdir(exist_ok=True);(dest/'physical-summary.json').write_text(json.dumps(dict(episodes=results,comparisons=comparisons),indent=2,allow_nan=False))
 for k,e in results.items():
  if 'phases' not in e:print(k,'BOUNDARY',e['boundary']);continue
  a=e['phases'].get('all_contact',{});h=e['phases'].get('hold',{});w=e['phases']['whole'];print(k,'all/holdCM',a.get('tracking_rms_cm'),h.get('tracking_rms_cm'),'force',a.get('force_rms_n'),'cap',a.get('force_cap_fraction'),'qstep',w['qPD_step_max_deg'],'recover',e.get('root_speed_release_plus_2s_mps',e['final_root_speed_mps']))
 return results
if __name__=='__main__':analyze()
