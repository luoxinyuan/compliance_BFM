"""P0-B paired no-contact safety qualification and conditional engineering updates."""
import argparse,copy,json,os,subprocess,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ.setdefault('TMPDIR',str(STAGE/'tmp'));Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(STAGE/'torch-cache'))
import m11_ipc


def study(e,m,contract,out):
 import torch,numpy as np,mujoco
 from native_env import install as native_install
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 native_install();speed(e);torch.set_num_threads(1);torch.set_grad_enabled(False)
 (out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix')
 from dual_track_fb_online import factory,Collector,run_updates,cpu
 from dual_track_task import sample_spec,global_rng,restore_global_rng
 from dual_track_high import write
 from diagnostic_fb_resume import tree_digest,publish,NormalStop
 from fb_ordered_grad import install as ordered
 from fb_safe_action import SafeAction,install,bind_task,KEY,VERSION
 model=e.sim.model;ranges=[]
 for name in contract['dof_names']:
  joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
  assert joint>=0;ranges.append(model.jnt_range[joint].tolist())
 contract['safe_execution_v1']=dict(version=VERSION,ankle_target_inset_rad=.05,handoff_s=.35,joint_ranges=ranges,metadata_key=KEY)
 old,_=factory(compile=False);agent,_=factory(compile=False);ordered(agent)
 assert tree_digest(old._model.state_dict())==tree_digest(agent._model.state_dict())
 transform=SafeAction(contract,ranges,'cuda');install(agent._model,m,transform)
 write(out/'safe-action-contract.json',dict(version=VERSION,q_PD_ankle_inset_rad=.05,
  ranges=ranges,ankle_indices=transform.indices,transition_s=.35,metadata_key=KEY,
  metadata_not_neural_input=True,actual_q_excess_boundary_rad=.02,
  retain_no_contact_qPD_coefficient=.1,retain_scale_rad2=.01,formal_FB_training=False))
 r=np.random.default_rng(58021);groups=[]
 for source in ('standing','goal_arm5','motion_walk_task'):
  for i in range(10):
   s=sample_spec(r,source,'none');s['release']=s['source_until']+6.;s['end']=s['source_until']+8.
   # Parent IDs version the complete task, never reuse a different path's label.
   from dual_track_task import digest
   s['parent_id']=digest(dict(protocol=VERSION,spec=s));s['common_prefix_parent_id']=s['parent_id']
   groups.append(dict(spec=s,seed=58601+len(groups)))
 write(out/'paired-parent-manifest.json',groups)
 results=[]
 from fb_safe_data import SafeNativeSequences
 col=Collector(e,m,contract,old,8192);bind_task(col.task);col.sequences=SafeNativeSequences()
 # One live Task/callback object throughout; never stack reset hooks.
 for g in groups:
  for kind in ('native_mean','original_step0_random','safe_handoff_random'):
   candidate=old if kind=='original_step0_random' else agent
   col.proxy.policy_model=m if kind=='native_mean' else candidate._model
   col.proxy.stochastic=kind!='native_mean'
   try:col.reset(g['spec'])
   except RuntimeError as exc:
    results.append(dict(parent_id=g['spec']['parent_id'],source=g['spec']['source'],kind=kind,source_failure=str(exc)))
    write(out/'paired-summaries.json',results);continue
   col.task.label=g['spec']['parent_id'][:12]+'-'+kind
   prefix=col.task.prefix_data['audit']['sha256']
   torch.manual_seed(g['seed']);np.random.seed(g['seed'])
   proposals=[]
   while not col.task.done:
    raw=col.task.native(col.task.spec['source'],float(e.tether.time))/16
    col.proxy.take_step(raw)
    if kind=='safe_handoff_random' and transform.last is not None:
     proposals.append({k:v.cpu().tolist() for k,v in transform.last.items()})
   q=np.array([x['qpos'][7:] for x in col.task.physics]);dq=np.array([x['qvel'][6:] for x in col.task.physics])
   qpd=np.array([x['q_PD'] for x in col.task.controls]);tau=np.array([x['torque'] for x in col.task.physics])
   idx=transform.indices
   rows=col.task.physics;first,last=rows[0],rows[-1]
   result=dict(parent_id=g['spec']['parent_id'],source=g['spec']['source'],kind=kind,prefix_sha256=prefix,
    boundary=col.task.boundary,controls=len(col.task.controls),force_max=max(np.linalg.norm(x['force']) for x in rows),
    actual_ankle_min=q[:,idx].min(0).tolist(),actual_ankle_max=q[:,idx].max(0).tolist(),
    ankle_qdot_max=np.abs(dq[:,idx]).max(0).tolist(),ankle_qPD_min=qpd[:,idx].min(0).tolist(),ankle_qPD_max=qpd[:,idx].max(0).tolist(),
    ankle_torque_max=np.abs(tau[:,idx]).max(0).tolist(),
    root_delta=(np.array(last['root'])-first['root']).tolist(),
    loaded_slip_peak=max(x['stage_loaded_slip_mps'] for x in rows),
    yaw_end_deg=float(np.degrees(np.arctan2(last['pelvis_matrix'][1][0],last['pelvis_matrix'][0][0]))))
   col.task.save(out)
   import gzip
   with gzip.open(out/(col.task.label+'-'+kind+'-safe-proposals.json.gz'),'wt') as f:json.dump(proposals,f)
   results.append(result);write(out/'paired-summaries.json',results)
   print('SAFE_PAIR',result['source'],kind,result['boundary'],flush=True)
 # Qualification never lowers original physical boundaries or native quality guards.
 qualified=True;reasons=[];bad_old=0;bad_safe=0
 for g in groups:
  rr=[x for x in results if x['parent_id']==g['spec']['parent_id']]
  if len(rr)!=3 or any('source_failure' in x for x in rr):qualified=False;reasons.append(dict(parent=g['spec']['parent_id'],reason='source_qualification_missing'));continue
  a,b,c=rr
  assert a['prefix_sha256']==b['prefix_sha256']==c['prefix_sha256'],'Paired source prefix differs'
  assert all(x['force_max']==0. for x in rr),'No-contact positive control has actual external force'
  bad_old+=b['boundary'] is not None;bad_safe+=c['boundary'] is not None
  if c['boundary'] is not None:qualified=False;reasons.append(dict(parent=c['parent_id'],reason='safe_physics_boundary'))
  if c['loaded_slip_peak']>a['loaded_slip_peak']+.2:qualified=False;reasons.append(dict(parent=c['parent_id'],reason='slip_native_degradation'))
  if abs(c['yaw_end_deg'])>abs(a['yaw_end_deg'])+10:qualified=False;reasons.append(dict(parent=c['parent_id'],reason='yaw_native_degradation'))
  if a['source']=='motion_walk_task' and c['root_delta'][0]<.75*a['root_delta'][0]:qualified=False;reasons.append(dict(parent=c['parent_id'],reason='walking_progress_lost'))
  if a['source']!='motion_walk_task' and np.linalg.norm(c['root_delta'][:2])>np.linalg.norm(a['root_delta'][:2])+.05:qualified=False;reasons.append(dict(parent=c['parent_id'],reason='standing_arm_drift'))
 if bad_old>=4 and bad_safe>.25*bad_old:qualified=False;reasons.append(dict(reason='less_than_75pct_boundary_reduction'))
 write(out/'paired-qualification.json',dict(qualified=qualified,old_boundaries=bad_old,safe_boundaries=bad_safe,reasons=reasons,
  thresholds_locked_before_execution=True,engineering_updates_authorized_only_if_qualified=True))
 if not qualified:return dict(status='safety_recipe_stopped',qualification=False,engineering_transitions=0,engineering_FB_updates=0,formal_FB_training=0)
 # The original CPR minibatch is the bounded native no-contact preservation domain.
 from fb_safe_update import install_retention
 agent._safe_qpd_scale=transform.scale
 agent._safe_action_transform=transform
 install_retention(agent,m)
 col.proxy.policy_model=agent._model;col.proxy.stochastic=True
 original_take=col.proxy.take_step
 def take(raw,native_limit=5):
  rows,done,stats=original_take(raw,native_limit)
  # Records actual low-level execution; raw proposals are not replay actions.
  # Per-control raw diagnostics are attached by the proxy hook below.
  for row,record in zip(rows,col.proxy.records):
   row['raw_actor_proposal']=record['raw_actor_proposal']
   row['actual_q_PD']=record['actual_q_PD']
   row['safety_transform_active']=record['safety_transform_active']
  return rows,done,stats
 original_act=col.proxy.act
 def act(obs,z,mean=True):
  action=original_act(obs,z,mean)
  if not col.proxy.source_mode:
   record=col.proxy.pending;record['raw_actor_proposal']=transform.last['raw_proposal'].cpu()
   record['actual_q_PD']=transform.last['q_PD'].cpu();record['safety_transform_active']=transform.last['active'].cpu()
  return action
 col.proxy.act=act;col.proxy.take_step=take
 col.output=out;col.bootstrap_sources();updates=0;stop=NormalStop(out)
 progress=dict(validation=True,formal_training=False,collection_block_start=0,collection_block_target=1024,collected_within_block=0,completed_agent_updates=0)
 publish(out/'safe-step0',agent,col,0,ARGS.runtime,progress,'initial')
 for target in range(1024,8193,1024):
  if stop.pending():break
  col.collect(target-col.transitions)
  if target>1024:
   run_updates(agent,col,updates,16,compile=False);updates+=16
  progress.update(collection_block_start=target,collection_block_target=target+1024,collected_within_block=0,completed_agent_updates=updates)
  publish(out/f'safe-step{target}',agent,col,updates,ARGS.runtime,progress,'rolling')
  write(out/'engineering-progress.json',dict(transitions=col.transitions,updates=updates,episodes=col.episode_rows,formal_training=0))
  if col.episode_rows and sum(x['boundary'] is not None for x in col.episode_rows[-10:])>=5:
   (out/'STOP_REQUEST.json').write_text(json.dumps(dict(reason='post-update early termination collapse')));break
 publish(out/f'user-stop-safe-{col.transitions}',agent,col,updates,ARGS.runtime,progress,'user_stop')
 return dict(status='engineering_completed',qualification=True,engineering_transitions=col.transitions,engineering_FB_updates=updates,formal_FB_training=0)


def main():
 global ARGS
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);ARGS=p.parse_args()
 ARGS.runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 ARGS.output.mkdir(parents=True,exist_ok=False)
 os.environ.update(CLEAN_KIND='direct',P4000_CANONICAL_ROW_ACTOR='1',BFM_RECORD_NOMINAL_DIAGNOSTIC='0',BFM_RECOVERY_OUTPUT=str(ARGS.output))
 os.chdir(ARGS.output)
 import m10_stage,m10_physics
 m10_physics.OUT=ARGS.output;m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
