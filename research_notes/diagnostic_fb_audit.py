"""Read-only archived boundary attribution. No learning or counterfactual reconstruction."""
import csv,gzip,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import mujoco
from dual_track_high import write,sha

P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
OLD=P/'experiments/dual-track-formal-20261109'

def audit(out,model=None):
 begun=time.monotonic();root=OLD/'repo'
 xml=root/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'
 if not xml.exists():xml=root/'data/robots/g1/scene_29dof_freebase_mujoco.xml'
 m=model or mujoco.MjModel.from_xml_path(str(xml))
 ids=np.flatnonzero(m.jnt_limited);names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,int(j)) for j in ids]
 qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];bounds=m.jnt_range[ids]
 contract=json.loads((OLD/'runs/fb-1m/action-contract.json').read_text());dofs=contract['dof_names']
 episodes=json.loads((OLD/'runs/fb-1m/episodes.json').read_text())
 paths={};total=0
 for f in (OLD/'runs/fb-1m/failures').glob('*.json.gz'):
  if 'source-prefix' in f.name:continue
  data=json.load(gzip.open(f,'rt'));key=data['condition'].get('parent_id');paths.setdefault(key,[]).append((f,data))
 report=[];missing=[]
 for i,ep in enumerate(episodes):
  start=total;total+=ep['controls']
  if ep['boundary'] is None:continue
  s=ep['spec'];matches=paths.get(s['parent_id'],[])
  match=next(((p,d) for p,d in matches if abs(d['boundary']['time']-ep['boundary']['time'])<1e-8),None)
  if match is None:missing.append(i);continue
  p,d=match;prefix_path=p.with_name(p.name.replace('.json.gz','-source-prefix.json.gz'))
  prefix=json.load(gzip.open(prefix_path,'rt')) if prefix_path.exists() else None
  rows=d['physics'];controls=d['control'];first=None
  for j,r in enumerate(rows):
   q=np.asarray(r['qpos'])[qa];excess=np.maximum(bounds[:,0]-q,0)+np.maximum(q-bounds[:,1],0)
   triggers=dict(joint_limit=bool(excess.max()>.02),height=r['height']<.45,tilt=r['tilt_deg']>30,
      ground=bool(r['undesired_ground_contacts']),native_failed=bool(r['failed']))
   if any(triggers.values()):first=(j,r,excess,triggers);break
  if first is None:missing.append(i);continue
  j,r,excess,triggers=first;k=int(np.argmax(excess));name=names[k];ci=max(0,sum(c['time']<=r['force_time']+1e-8 for c in controls)-1);control=controls[ci]
  idx=dofs.index(name) if name in dofs else None
  source_last=prefix['physics'][-1] if prefix and prefix['physics'] else None
  src_excess=None
  if source_last:
   sq=np.asarray(source_last['qpos'])[qa];src_excess=float((np.maximum(bounds[:,0]-sq,0)+np.maximum(sq-bounds[:,1],0)).max())
  # Count reflects actual native controls, not inferred episode probabilities.
  global_control=start+ci;updates_before=max(0,global_control//1024-1)*16
  row=dict(episode_index=i,parent_id=s['parent_id'],source=s['source'],family=s['demand_family'],
   archive=str(p),archive_sha256=sha(p),trigger_joint=name,triggers=triggers,
   time=r['time'],handoff_time=s['handoff_time'],since_handoff_s=r['time']-s['handoff_time'],
   onset=s['onset'],contact_started=r['time']>=s['onset'],active=r['active'],force_xyz=r['force'],
   q_rad=float(r['qpos'][qa[k]]),lower_rad=float(bounds[k,0]),upper_rad=float(bounds[k,1]),excess_rad=float(excess[k]),
   qdot_rad_s=float(r['qvel'][va[k]]),q_PD_rad=control['q_PD'][idx] if idx is not None else None,
   action=control['action'][idx] if idx is not None else None,torque_Nm=r['torque'][idx] if idx is not None else None,
   source_last_excess_rad=src_excess,source_last_q_rad=source_last['qpos'][qa[k]] if source_last else None,
   actual_control_index=global_control,updates_before_control=updates_before,
   actor_execution='sampled mutable actor with clipped noise; source uses original frozen actor',
   trace_first_boundary_index=j,stored_boundary_time=ep['boundary']['time'],height=r['height'],tilt_deg=r['tilt_deg'],failed=r['failed'])
  report.append(row)
 write(out/'old-fb-boundaries.json',report)
 with (out/'old-fb-boundaries.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(report[0]));w.writeheader();w.writerows(report)
 stats=dict(recorded_episodes=len(episodes),physics_review=sum(e['boundary'] is not None for e in episodes),
  complete_boundary_traces=len(report),missing_episodes=missing,
  joints=dict(Counter(r['trigger_joint'] for r in report)),
  no_contact_at_trigger=sum(not r['active'] for r in report),precontact=sum(not r['contact_started'] for r in report),
  first100ms=sum(r['since_handoff_s']<=.100001 for r in report),
  before_any_update=[r['episode_index'] for r in report if r['updates_before_control']==0],
  all_joint_boundary=all(r['triggers']['joint_limit'] for r in report),
  first_trigger=report[0],last_trigger=report[-1],
  causal_limit='Post512 model/Adam/replay lost. No old-policy deterministic/noise/update ablation can be recovered.',
  boundary_policy_unchanged_rad=.02,xml=str(xml),xml_sha256=sha(xml),wall_s=time.monotonic()-begun)
 write(out/'old-fb-audit.json',stats);return stats

def run(e,m,c,out,args):
 result=audit(out,e.sim.model);return dict(status='completed',audit=result,training_updates=0)

if __name__=='__main__':
 import sys
 out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False);audit(out)
