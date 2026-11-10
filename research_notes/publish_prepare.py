"""Prepare small reviewed summaries from completed owned diagnostic jobs."""
import copy,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
S=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/fb-native-audit-20261110')
R=S/'runtime';A=R/'research_notes/fb_native_bias_artifacts';A.mkdir(exist_ok=False)
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inv=read(S/'runs/native-inventory-01/inventory.json');inv['dof_names']=read(S/'runs/native-inventory-01/action-contract.json')['dof_names']
b=read(S/'runs/b-mean-01/b-mean-results.json')
for row in b['rows']:
 row.pop('z');row.pop('centered_z_DIAGNOSTIC_ONLY')
write(A/'b-mean-summary.json',b);write(A/'inventory-summary.json',inv)
tr=read(S/'runs/trace-analysis/result.json');write(A/'update-trace-summary.json',tr)
jobs=[read(p) for p in sorted((S/'jobs').glob('*.json'))];assert len(jobs)==5 and all(j['status']=='completed' for j in jobs)
for j in jobs:j['output_scope']='owned private stage, complete logs/trajectories retained'
measured=sum(j['elapsed_s'] for j in jobs)
files=[p for p in S.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(S).parts]
size=sum(p.stat().st_size for p in files)
cost=dict(measured_supervised_job_s=measured,preparation_and_unseparately_timed_CPU_analysis_reserve_s=900,
 conservative_charged_job_s=measured+900,ceiling_s=7200,private_output_bytes=size,output_ceiling_bytes=12*(1<<30),
 costs=[{k:j[k] for k in ['name','runtime','elapsed_s','pid','returncode','status']} for j in jobs],
 actual_engineering_complete_agent_updates=4,new_engineering_native_transitions=1118,
 replay_restore_steps_not_new_training_samples=True,native_inventory_physics_ticks=31200,
 native_inventory_reset_prefill_physics_ticks=12*600,native_inventory_actual_source_suffix_ticks=sum(len(json.load(gzip.open(S/'runs/native-inventory-01'/(r['name']+'-actual.json.gz'),'rt'))['source_prefix']['physics']) for r in inv['rows']),
 offline_sampled_state_rows=30000,offline_unique_replay_index_rows=int(len(np.unique(np.load(S/'runs/b-mean-01/sample-indices.npy'),axis=0))),offline_backward_state_evaluations=240005,
 caveat='900s is conservative reserve, not a measured CPU job duration; source/backend reconstruction costs included in measured job wall; no formal training',
 original_bfm_sha256=inv['model_sha256'],formal_PPO_Student_FB_long_runs=0)
assert cost['conservative_charged_job_s']<=7200 and size<12*(1<<30)
write(A/'cost-runtime.json',cost);write(S/'ledger.json',cost);write(A/'protocol.json',read(S/'protocol.json'))
identities=dict(base_commit='55cfe85928365ed4933c40a4b62ebc18e6324e12',original_bfm_sha256=inv['model_sha256'],
 original_replay_sha256=b['replay_sha256'],source_references={})
P=S.parents[1]
paths=[P/'bridge-v2-owned/runs/native-contact-20261106/native-assets.json',
 P/'bridge-v2-owned/runs/m10-task-appropriate-20261101/source_control-attempt-01/outputs/references.json',
 P/'experiments/dual-track-diagnostics-20261110/runs/d2-control-02/user-stop-1037/manifest.json']
for p in paths:identities['source_references'][str(p.relative_to(P))]=sha(p)
for name in ['trace-control-01','trace-resume-01','trace-resume-02']:
 x=read(S/'runs'/name/'trace-result.json');identities[name]={k:x[k] for k in ['post_model','post_adam','operators']}
identities['hash_scope']='post_model/post_adam are canonical tensor-tree digests, not model file SHA'
write(A/'identities.json',identities)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
for kind in ['standing','forward_motion','backward_reward','arm_lift_goal','lower_body_goal','left_waist_goal','right_waist_goal']:
 with gzip.open(S/'runs/native-inventory-01'/(kind+'-actual.json.gz'),'rt') as f:x=json.load(f)
 p=x['physics'];t=np.array([r['time'] for r in p]);root=np.array([r['root'] for r in p]);rel=np.array([np.asarray(r['wrist'])-r['shoulder_xyz'] for r in p])
 yaw=lambda key:np.unwrap([np.arctan2(np.asarray(r[key])[1,0],np.asarray(r[key])[0,0]) for r in p])
 if kind in ['standing','forward_motion','backward_reward']:ax[0,0].plot(t,root[:,0]-root[0,0],label=kind)
 if kind in ['standing','lower_body_goal']:ax[0,1].plot(t,root[:,2],label=kind)
 if kind in ['standing','arm_lift_goal']:ax[1,0].plot(t,rel[:,2],label=kind)
 if kind in ['standing','left_waist_goal','right_waist_goal']:ax[1,1].plot(t,np.degrees(yaw('torso_matrix')-yaw('pelvis_matrix')),label=kind)
for a,title,ylabel in zip(ax.flat,['Real root translation','Real root height','Real wrist relative to shoulder','Real waist yaw'],['Root X displacement (m)','Root Z (m)','Wrist - shoulder Z (m)','Torso - pelvis yaw (deg)']):
 a.set(title=title,xlabel='Physics time (s)',ylabel=ylabel);a.grid(alpha=.2);a.legend(fontsize=8)
 for v in [5.5,11.5,13.]:a.axvline(v,color='gray',ls='--',lw=.7)
fig.suptitle('Frozen original BFM + original PD; actual no-force MuJoCo trajectories')
fig.savefig(A/'native-actual-timeseries.png',dpi=140);plt.close(fig)
print(json.dumps(cost,indent=2))
