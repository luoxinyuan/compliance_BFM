"""Saved-observation actor counterfactuals; no new rollout or optimization."""
import json,gzip
from pathlib import Path
import numpy as np,torch
from m9b_stage import OLD
from m9_cyclic import CyclicFeedback
from m9_analysis import summarize

def run(env,model,contract,out):
 scale=torch.tensor(contract['q_PD_scale'],device=env.standing.device);offset=torch.tensor(contract['q_PD_offset'],device=env.standing.device)
 def q(c,z):
  obs={k:torch.tensor(c['bfm_obs'][k],device=env.standing.device) for k in env.keys};a=model.act(obs,torch.tensor(z,device=env.standing.device),mean=True);return (offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5)[0].cpu().numpy()
 results={};refs=json.loads((OLD/'native-cycles.json').read_text())['cycles']
 for name in ['new_straight','new_right','unsupported_left','new_heading']:
  for ctl in ['expert','student']:
   source=OLD/'final-attempt-01/outputs'/f'{name}__{ctl}.json.gz'
   with gzip.open(source,'rt') as f:e=json.load(f)
   cs=e['control'];rs=e['physics'];dt=.02;indices=np.argsort([c['q_PD_step_max_rad'] for c in cs])[::-1];indices=[int(i) for i in indices if i>0][:12];counter=[]
   for i in indices:
    c,p=cs[i],cs[i-1];a=q(c,c['executed_z']);b=q(c,p['executed_z']);d=q(p,p['executed_z']);a0=q(p,c['executed_z']);command=.5*((a-b)+(a0-d));state=.5*((a-a0)+(b-d));j=int(np.argmax(abs(a-d)));phase=c['oracle_controller_state']['phase'];prior=p['oracle_controller_state']['phase']
    counter.append(dict(time=c['time'],joint=contract['dof_names'][j],jump_deg=float(np.degrees((a-d)[j])),command_shapley_deg=float(np.degrees(command[j])),state_shapley_deg=float(np.degrees(state[j])),reconstruction_max_rad=float(abs(command+state-(a-d)).max()),record_parity_rad=float(abs(a-c['q_PD']).max()),phase_wrap=int(phase)!=int(prior),alpha=c['oracle_controller_state']['alpha'],turn=c['oracle_controller_state']['turn']))
   active=[r for r in rs if r['active']];F=np.array([r['force'] for r in active]);root=np.array([r['root'] for r in active]);w=np.array([r['wrist'] for r in active]);velocity=np.array([r['qvel'][:3] for r in active]);av=np.array([r['anchor_velocity'] for r in active]);R=np.array([r['pelvis_matrix'] for r in active]);heading=np.unwrap(np.arctan2(R[:,1,0],R[:,0,0]));moving=np.linalg.norm(velocity[:,:2],axis=1)>.05
   # Teacher memory replay, reconstruct current guide from saved postphysics row at same time.
   fb=CyclicFeedback({k:r['period'] for k,r in refs.items()},True);replay=[]
   for c in cs:
    rr=min(rs,key=lambda r:abs(r['time']-c['time']));guide=np.array(rr['anchor_velocity'])+.5*(np.array(rr['anchor'])-rr['wrist']);rot=np.array(c['sensor']['world_R_body']);s=fb.step(c['measured_force_xyz'],rot,guide=guide);ref=c['oracle_controller_state'];replay.append([abs(s[k]-ref[k]) for k in ['alpha','phase','turn']])
   results[e['label']]=dict(source=str(source),physical=summarize(e),largest_jumps=counter,top12_wraps=sum(r['phase_wrap'] for r in counter),state_contribution_median_abs_deg=float(np.median([abs(r['state_shapley_deg']) for r in counter])),command_contribution_median_abs_deg=float(np.median([abs(r['command_shapley_deg']) for r in counter])),force_energy_xyz_fraction=(np.mean(F**2,0)/np.mean(np.sum(F**2,1))).tolist(),wrist_relative_root_xyz_range_m=np.ptp(w-root,axis=0).tolist(),wrist_relative_root_speed_rms_xyz=np.sqrt(np.mean(np.diff(w-root,axis=0)**2/.005**2,axis=0)).tolist(),actual_moving_fraction=float(moving.mean()),moving_world_velocity_mean=velocity[moving].mean(0).tolist(),moving_partner_velocity_mean=av[moving].mean(0).tolist(),moving_heading_range_deg=np.degrees([heading[moving].min(),heading[moving].max()]).tolist(),query_memory_replay_max=np.max(replay,axis=0).tolist())
   print('DIAGNOSED',e['label'],flush=True)
 (out/'diagnosis.json').write_text(json.dumps(results,indent=2));return dict(status='completed',student_optimizer_updates=0,source='existing final traces now marked development; no new physics',results=str(out/'diagnosis.json'))
