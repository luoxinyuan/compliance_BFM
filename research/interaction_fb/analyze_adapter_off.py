import json, math, argparse, hashlib
from pathlib import Path
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);a=p.parse_args();S=a.stage;D=S.parent/'interaction-pilot-b-diagnosis-20261112'
out={};loaded={}
def exact(x,y):
 if torch.is_tensor(x):return torch.equal(x,y)
 if isinstance(x,np.ndarray):return x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
 if isinstance(x,dict):return x.keys()==y.keys() and all(exact(x[k],y[k]) for k in x)
 if isinstance(x,(tuple,list)):return len(x)==len(y) and all(exact(i,j) for i,j in zip(x,y))
 if isinstance(x,float):return x.hex()==y.hex()
 return x==y
for mode in ['sample','mean','offsample','offmean']:
 d=(S if mode.startswith('off') else D)/'runs'/f'{mode}-B19'
 j=json.loads((d/'probe.json').read_text());t=torch.load(d/'probe-trace.pt',map_location='cpu',weights_only=False);o=json.loads((d/'same-state-actions.json').read_text());loaded[mode]=(j,t,o)
 rows=t['physics']; cs=o['controls'];con=o['contract'];idx=next(i for i,c in enumerate(con) if c['name']=='left_ankle_pitch_joint');qa=con[idx]['qposadr'];lo,hi=con[idx]['range']
 def metrics(rs):
  qw,qx,qy,qz=cs[0]['pre_qpos'][3:7];initial_yaw=math.degrees(math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz)))
  def yaw(row):
   mat=np.asarray(row['pelvis_matrix']).reshape(3,3);return math.degrees(math.atan2(mat[1,0],mat[0,0]))
  return dict(t_start=rs[0]['time'],t_end=rs[-1]['time'],ticks=len(rs),max_joint_excess_rad=max(r['joint_limit_excess_max_rad'] for r in rs),left_ankle_pitch_min_rad=min(r['qpos'][qa] for r in rs),left_ankle_pitch_final_rad=rs[-1]['qpos'][qa],root_xy_displacement_m=float(np.linalg.norm(np.asarray(rs[-1]['root'])[:2]-np.asarray(cs[0]['pre_qpos'])[:2])),final_root_xyz=rs[-1]['root'],final_root_speed_mps=float(np.linalg.norm(rs[-1]['qvel'][:3])),height_min_m=min(r['height'] for r in rs),tilt_max_deg=max(r['tilt_deg'] for r in rs),pelvis_yaw_final_deg=yaw(rs[-1]),pelvis_yaw_change_from_restored_deg=(yaw(rs[-1])-initial_yaw+180)%360-180,loaded_slip_peak_mps=max(float(np.max(r['stage_loaded_slip_mps'])) for r in rs),loaded_slip_p95_mps=float(np.percentile([np.max(r['stage_loaded_slip_mps']) for r in rs],95)),no_support_ticks=sum(not any(r['support']) for r in rs),force_max_n=max(float(np.linalg.norm(r['force'])) for r in rs))
 out[mode]=dict(status=j['status'],error=j['error'],agent_digest=j['frozen_agent_digest'],rng_digest=j['restored_rng_digest'],trace_sha=j['trace_sha256'],updates=j['updates'],full=metrics(rows),matched_510ms=metrics([r for r in rows if r['time']<=12.31000001]),mean_q_PD_min_rad=min(c['mean_q_PD'][0][idx] for c in cs),mean_q_PD_max_rad=max(c['mean_q_PD'][0][idx] for c in cs),mean_q_PD_outside_count=sum(not lo<=c['mean_q_PD'][0][idx]<=hi for c in cs),controls=len(cs),first_mean_q_PD=cs[0]['mean_q_PD'][0][idx],first_actual_q_PD=cs[0]['actual_q_PD'][0][idx],first_base_q_PD=cs[0]['base_q_PD'][0][idx],restored_cursor=j['original_cursor'],end_cursor=j['end_cursor'],diagnostic_transitions=j['end_cursor']-j['original_cursor'])
 if mode.startswith('off'):
  assert all(not c['adapter_applied'] for c in cs)
  assert all(np.array_equal(np.array(c['mean']),np.clip(np.array(c['base_mean']),-1,1)) for c in cs)
  assert j['updates']==38
for off,on in [('offsample','sample'),('offmean','mean')]:
 j,t,o=loaded[off];jj,tt,oo=loaded[on]
 # Entire archived prefix controls and commands stay identical up to the first intervention.
 prefix=[r for r in t['controls'] if r['time']<11.79999999];oldprefix=[r for r in tt['controls'] if r['time']<11.79999999]
 checks=dict(agent=j['frozen_agent_digest']==jj['frozen_agent_digest'],rng=j['restored_rng_digest']==jj['restored_rng_digest'],qpos=exact(o['controls'][0]['pre_qpos'],oo['controls'][0]['pre_qpos']),qvel=exact(o['controls'][0]['pre_qvel'],oo['controls'][0]['pre_qvel']),slot=exact(o['controls'][0]['slot'],oo['controls'][0]['slot']),task=exact(t['task_spec'],tt['task_spec']),prefix_controls=exact(prefix,oldprefix),first_command_and_prefix=exact(t['commands'][:len(tt['commands'])-5],tt['commands'][:-5]))
 # Commands count varies with hard stop; find exact shared prefix length directly.
 same=0
 for x,y in zip(t['commands'],tt['commands']):
  if not exact(x,y):break
  same+=1
 checks.pop('first_command_and_prefix');checks['first_command_shared']=same>=len(prefix)//5+1
 first_new=next(r for r in t['controls'] if r['time']>=11.79999999);first_old=next(r for r in tt['controls'] if r['time']>=11.79999999)
 checks['first_executed_z']=exact(first_new['executed_z'],first_old['executed_z']);checks['first_bfm_obs']=exact(first_new['bfm_obs'],first_old['bfm_obs'])
 assert all(checks.values()),checks
 out[off]['initial_identity_checks']=checks;out[off]['shared_command_prefix_count']=same
assert all(v['agent_digest']==out['sample']['agent_digest'] for v in out.values())
(S/'adapter-off-analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
