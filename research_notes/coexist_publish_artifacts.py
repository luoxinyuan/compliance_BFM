import json,pathlib,shutil,time,hashlib
P=pathlib.Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/feedback-coexistence-20261105';W=P/'bridge-v2-owned/trees/feedback-coexistence/repo/research_notes';O=W/'coexist_artifacts';O.mkdir(exist_ok=True)
for name in ['joint-extension-decision.json','final-frozen-models.json','training-summary.json','sampled-noise-audit.json','learning-curves.png','local-video-validation.json']:
 shutil.copy2(R/name,O/name)
keys=['error_cm','force_rms','force_xyz_rms','force_xyz_mean','signed_error_xyz_cm','cap_fraction','root_delta_cm','root_xy_path_cm','foot_xy_path_cm','relative_wrist_shoulder_delta_cm','wrist_shoulder_in_torso_axes_delta_cm','shoulder_elbow_q_delta_deg','absolute_mechanical_work_proxy_J','max_tilt','slip_peak','support_loss_ticks','torque_saturation_peak','actual_q_max_step_deg','root_final_speed','torso_yaw_delta_deg','pelvis_yaw_delta_deg','waist_final_deg']
rows=json.loads((R/'physical-metrics.json').read_text());short=[]
for x in rows:
 y={k:x[k] for k in ['path','sha256','label','condition','boundary','qpd_max_step_deg','qpd_p99_step_deg','return_value']};y['contact']={k:x['contact'][k] for k in keys} if x['contact'] else None;y['phase']={n:({k:v[k] for k in keys} if v else None) for n,v in x['phase'].items()};short.append(y)
(O/'physical-summary.json').write_text(json.dumps(short,separators=(',',':')))
curves=json.loads((R/'learning-curves.json').read_text());c={}
for seed,d in curves.items():
 c[seed]=dict(development=[dict(iteration=v['iteration'],score=v['score'],rows=[{k:r[k] for k in ['label','error_cm','force_rms','cap_fraction','boundary']} for r in v['rows']]) for v in d['development']],training=[{k:v[k] for k in ['iteration','rl_decisions','coverage','reward_terms','execution','task_advantage','retention']}|dict(kl=v['optimization']['epochs'][-1]['joint_gaussian_KL'],updated_EV=v['optimization']['updated_value_EV'],old_GAE_EV=v['optimization']['old_GAE_value_EV'],logprob_error=v['optimization']['pre_logprob_max_error'],gradient_components=v['optimization']['gradient_components']) for v in d['training']])
(O/'curves-summary.json').write_text(json.dumps(c,separators=(',',':')))
ledgers=[json.loads(p.read_text())|dict(file=p.name) for p in R.glob('*-ledger.json')];assert all(x['status']=='finished' and x.get('exit_code')==0 for x in ledgers)
output=sum(p.stat().st_size for p in R.rglob('*') if p.is_file());cost=dict(ledgers=ledgers,supervised_job_wall=sum(x['job_wall'] for x in ledgers),analysis_seconds=json.loads((R/'analysis-status.json').read_text())['seconds'],other_cpu_analysis_conservative_allowance_seconds=180,output_bytes=output,RL_decisions=49152,actor_optimizer_steps=194,new_Expert_supervision_updates=0,new_Student_updates=0,native_reward_prompting_runs=0,limits=dict(job_wall=18000,output_bytes=12*1024**3,RL_decisions=98304),assessment='No budget extension. GPU jobs completed. Additional CPU analysis allowance includes reward audit/plots/identity checks/local decode; no BFM inference on CPU.')
assert cost['supervised_job_wall']+cost['analysis_seconds']+180<18000 and output<12*1024**3
(R/'resource-closeout.json').write_text(json.dumps(cost,indent=2));(O/'resource-closeout.json').write_text(json.dumps(cost,indent=2));shutil.copy2(R/'videos/INDEX.md',O/'VIDEO_INDEX.md')
for name in ['final-learned','final-reference','final-FR']:shutil.copy2(R/name/'prefix-parity.json',O/(name+'-prefix-parity.json'))
print('jobwall',cost['supervised_job_wall'],'analysis',cost['analysis_seconds'],'GiB',output/1024**3,'artifact bytes',sum(p.stat().st_size for p in O.iterdir() if p.is_file()))
