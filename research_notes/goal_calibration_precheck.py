"""Read-only saved-state checks of two goal commands; no weights or physics updates."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
import mujoco
from wholebody_physics import goal_observation,save,read,RUNS
from ccl02a_teacher import arm_ik
SOURCE=RUNS/'issue21-wholebody-20261030A/attempt-01/outputs'
PINS={'goal-path.json':'2465366666a6ebe4868884fa84a88610507170d5cf5c01a7fdc57d3703f053e7','geometry.json':'8af3657086e250ab9476d5ed2fe186f15ff816f105756517f0ca1a8df5c4d6bb','goal_bfm.json':'b6d1ba7733185f06944455e38db58fbd0f24dcdcc2891229444cf254b0ac33b3'}
def angle(x,y):
 cosine=(x*y).sum(-1)/(x.norm(dim=-1)*y.norm(dim=-1));return float(torch.rad2deg(torch.acos(cosine.clamp(-1,1))).max())
def build_and_check(env,model,contract,common,out,protocol):
 for name,sha in PINS.items():assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==sha
 path=read(SOURCE/'goal-path.json')['waypoints'];geometry=read(SOURCE/'geometry.json');old=read(SOURCE/'goal_bfm.json')
 m=env.sim.model;dev=env.standing.device;standing=env.standing;d=mujoco.MjData(m);d.qpos[:]=geometry['qpos'];d.qvel[:]=old['initial']['qvel'];mujoco.mj_forward(m,d)
 initial=goal_observation(env,d.qpos,d.qvel);obs={k:torch.tensor(v,device=dev,dtype=torch.float32) for k,v in common['bfm_obs'].items()}
 parity={k:float((v-obs[k]).abs().max()) for k,v in initial.items()};assert max(parity.values())<1e-4
 z0=model.goal_inference(goal_observation(env,path[0]['qpos'],np.zeros(m.nv)))
 ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in contract['dof_names']];vadr=m.jnt_dofadr[ids];limits=m.jnt_range[ids];arm=contract['arm_indices']
 jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,np.array(geometry['wrist']),env.tether.body)
 scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev)
 def qtarget(a):return offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
 nominal=model.act(obs,standing,mean=True);qnom=qtarget(nominal);prev_action=torch.tensor(common['previous_action'],device=dev,dtype=torch.float32)
 rows=[];goals={'anchored':[],'native':[]};series={k:[] for k in goals};norm_errors=[];bparity=[];arm_clipping=[]
 for i,r in enumerate(path):
  goal=model.goal_inference(goal_observation(env,r['qpos'],np.zeros(m.nv)))
  old_b=torch.tensor(r['z'],device=dev,dtype=torch.float32);b=model.project_z(standing+goal-z0)
  bparity.append(float((b-old_b).abs().max()));assert bparity[-1]<2e-6
  # Execute B's exact saved command after reproducing its formula; avoid numerically altered control.
  goals['anchored'].append(old_b);goals['native'].append(goal)
  error=np.array(r['target'])-geometry['wrist'];common_correction=arm_ik(jac[:,vadr[arm]],error)
  row=dict(index=i,target_pose=r['qpos'],anchor=r['target'],target_fk_error_m=r['ik_error_m'],per_mode={})
  for label,z in [('native',goal),('anchored',old_b)]:
   norm_errors.append(abs(float(z.norm())-16));a=model.act(obs,z,mean=True);q0=qtarget(a)[0].cpu().numpy();correction=np.clip(q0[arm]+common_correction,limits[arm,0],limits[arm,1])-q0[arm]
   arm_clipping.append(float(abs(correction-common_correction).max()));da=torch.zeros_like(a);da[:,arm]=torch.tensor(correction,device=dev,dtype=a.dtype)/scale[:,arm]
   a=a+da;q=qtarget(a);qt=q[0].cpu().numpy();dq=(q-qnom)[0].cpu().numpy();proxy=mujoco.MjData(m);proxy.qpos[:]=geometry['qpos'];proxy.qpos[7:]=qt;mujoco.mj_forward(m,proxy)
   excess=np.maximum(limits[:,0]-qt,0)+np.maximum(qt-limits[:,1],0)
   row['per_mode'][label]=dict(z=z.tolist(),z_norm=float(z.norm()),angle_from_standing_deg=angle(z,standing),action=a[0].tolist(),q_PD=qt.tolist(),waist_q_PD_delta_deg=np.degrees(dq[12:15]).tolist(),leg_delta_rms_deg=float(np.degrees(np.sqrt(np.mean(dq[:12]**2)))),wrist_q_PD_FK=proxy.xpos[env.tether.body].tolist(),q_PD_joint_excess_rad=float(excess.max()),action_clip_ratio=float((a*5).abs().max()/contract['clip']),teacher_correction=correction.tolist())
   series[label].append(q);assert torch.isfinite(a).all()
  row['A_B_angle_deg']=angle(goal,old_b);row['A_B_q_PD_rms_deg']=float(torch.rad2deg((series['native'][-1]-series['anchored'][-1]).square().mean().sqrt()))
  rows.append(row)
 summary=dict(goal_inference_already_projects=True,backward_observation_parity=parity,z_goal_initial_vs_standing_deg=angle(z0,standing),anchored_formula_parity_max=max(bparity),norm_error_max=max(norm_errors),common_arm_correction_clip_change_max_rad=max(arm_clipping),per_mode={})
 for label,zs in goals.items():
  # Fixed observation diagnostic: target path changes at10Hz; release at original50Hz/.2s.
  qseq=[qtarget(prev_action)]+series[label];release=[]
  for j in range(11):
   f=max(0,1-j/10);z=model.project_z(standing+f*(zs[-1]-standing)) if f else standing
   a=model.act(obs,z,mean=True);da=torch.zeros_like(a);da[:,arm]=torch.tensor(rows[-1]['per_mode'][label]['teacher_correction'],device=dev,dtype=a.dtype)*f/scale[:,arm];release.append(qtarget(a+da))
  maxstep=max(float(torch.rad2deg((b-a).abs().max())) for a,b in zip(qseq,qseq[1:]));releasestep=max(float(torch.rad2deg((b-a).abs().max())) for a,b in zip(release,release[1:]))
  v=[r['per_mode'][label] for r in rows]
  summary['per_mode'][label]=dict(onset_jump_deg=float(torch.rad2deg((qseq[1]-qseq[0]).abs().max())),path_step_max_deg=maxstep,release_step_max_deg=releasestep,angle_max_deg=max(r['angle_from_standing_deg'] for r in v),q_PD_excess_max_rad=max(r['q_PD_joint_excess_rad'] for r in v),action_clip_ratio_max=max(r['action_clip_ratio'] for r in v))
 assert max(norm_errors)<1e-4
 permitted=all(v['q_PD_excess_max_rad']<1e-6 and v['action_clip_ratio_max']<1 and max(v['path_step_max_deg'],v['release_step_max_deg'])<=protocol['precheck_max_command_step_deg'] for v in summary['per_mode'].values())
 summary.update(physical_allowed=permitted,screen_note='Fixed-state forward screen, not a dynamical stability guarantee. Step threshold10deg is a simulation preflight guard; original physical stop rules remain. Release forward diagnostic blends same bounded correction over.2s; actual controller retains original causal fade timing.',student_cone_upper_bound_deg=math.degrees(math.atan(.3)))
 save(out,'same-state-forward.json',dict(summary=summary,states=rows));save(out,'precheck-decision.json',summary);save(out,'goal-path.json',read(SOURCE/'goal-path.json'));save(out,'geometry.json',geometry);save(out,'reused-inputs.json',{str(SOURCE/n):s for n,s in PINS.items()})
 return goals,summary
