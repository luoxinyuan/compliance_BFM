"""Saved-artifact-only closeout following explicit user stop of the second seed."""
import json,gzip,hashlib,time,collections,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];S=ROOT.parent;P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');OLD=P/'experiments/interactive-contact-pilot-20261108'
def rd(p):
 with gzip.open(p,'rt')if str(p).endswith('.gz')else p.open()as f:return json.load(f)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def quality(folder,row):
 ep=rd(folder/(row['label']+'.json.gz'));rs=ep['physics'];cs=ep['control'];feet=np.array([x['feet_xyz']for x in rs]);root=np.array([x['root']for x in rs]);q=np.array([x['qpos']for x in rs]);forces=np.array([x['force']for x in rs]);active=np.array([x['active']for x in rs]);qpd=np.array([x['q_PD']for x in cs]);jumps=np.rad2deg(np.abs(np.diff(qpd,axis=0)));actual=np.rad2deg(np.abs(np.diff(q[:,7:],axis=0)));imax=np.unravel_index(np.argmax(jumps),jumps.shape)
 return dict(q_PD20ms_peak_deg=float(jumps.max()),q_PD20ms_p99_deg=float(np.quantile(jumps,.99)),q_PD_peak_joint_index=int(imax[1]),q_PD_peak_time=cs[imax[0]+1]['time'],q_actual5ms_peak_deg=float(actual.max()),feet_path_m=np.linalg.norm(np.diff(feet,axis=0),axis=-1).sum(axis=0).tolist(),feet_max_excursion_m=np.linalg.norm(feet-feet[0],axis=-1).max(axis=0).tolist(),root_start_to_end_cm=((root[-1]-root[0])*100).tolist(),root_max_excursion_m=float(np.linalg.norm(root-root[0],axis=-1).max()),actual_active_force_XYZ_RMS=np.sqrt(np.mean(forces[active]**2,axis=0)).tolist()if active.any()else[0,0,0],physics_ticks=len(rs),BFM_control_steps=len(cs),q_PD_is_target_not_actual_motion=True)
def prefix(folder,a,b,until):
 x=rd(folder/(a+'.json.gz'));y=rd(folder/(b+'.json.gz'));out={}
 for key in ['physics','control']:
  xx=[r for r in x[key]if r['time']<until-1e-7];yy=[r for r in y[key]if r['time']<until-1e-7];out[key]=dict(n=len(xx),same_count=len(xx)==len(yy),all_fields_equal=xx==yy)
 return out

def main():
 start=time.monotonic();out=S/'runs/closeout-01';out.mkdir(exist_ok=False);e=rd(S/'runs/eval-02/summaries.json');gate=rd(S/'runs/eval-02/route-gate.json');sel=rd(S/'runs/eval-02/selection-before-final.json');hist=rd(S/'runs/history-cost-02/result.json');fit=rd(S/'runs/fit-01/result.json');ledger=rd(S/'ledger.json');seeds={};trainfamilies={};newphysics=0;quality_rows=[]
 import torch
 bc=OLD/'runs/bc-fit-01/old_data_only-500.pt';bcstate=torch.load(bc,map_location='cpu',weights_only=True)['state_dict'];identity={}
 for seed in [5401,5402]:
  folder=S/f'runs/ppo-clean-{seed}';learn=rd(folder/'learning.json');summ=rd(folder/'summaries.json');batches=[];counts=collections.Counter();walls=collections.Counter();phases=collections.Counter();iterations=[r['iteration']for r in learn]
  for it in iterations:
   stats=rd(folder/f'rollout-{it}-stats.json');assert len(stats)==768
   for st in stats:counts[st['demand_family']]+=1;walls[st['demand_family']]+=st['decision_wall_s'];phases[st['phase']]+=1;newphysics+=st['physics_steps']
   batches.append(dict(iteration=it,task_decisions=dict(collections.Counter(r['demand_family']for r in stats)),phase_decisions=dict(collections.Counter(r['phase']for r in stats)),sample_mean_z_angle_deg=dict(mean=float(np.mean([r['z_sample_mean_angle_deg']for r in stats])),p90=float(np.quantile([r['z_sample_mean_angle_deg']for r in stats],.9))),raw_mean_norm=float(np.mean([r['raw_mean_norm']for r in stats])),raw_sample_norm=float(np.mean([r['raw_sample_norm']for r in stats])),executed_std=float(np.mean([r['executed_std']for r in stats]))))
  state=torch.load(folder/'candidate-0.pt',map_location='cpu',weights_only=True)['state_dict'];keys=[k for k in bcstate if k.startswith('actor.')or k in ['center','scale','log_std']];assert all(torch.equal(state[k],bcstate[k])for k in keys);assert any(not torch.equal(state[k],bcstate[k])for k in bcstate if k.startswith('critic.'))
  identity[seed]=dict(actor_normalizer_std_bitwise_equal_to_BC500=True,critic_fresh=True)
  seeds[seed]=dict(status='completed_fixed7'if seed==5401 else'user_stopped_during_dev4_no_final_node',learning=learn,development=summ,actual_saved_RL_decisions=len(iterations)*768,batches=batches,family_decisions=dict(counts),family_decision_wall_s=dict(walls),phase_decisions=dict(phases),checkpoints={str(i):dict(path=str(folder/f'candidate-{i}.pt'),sha256=sha(folder/f'candidate-{i}.pt'))for i in [0,4,7]if(folder/f'candidate-{i}.pt').exists()})
  trainfamilies[seed]=dict(decisions=dict(counts),rollout_decision_wall_s=dict(walls),note='Reset, optimization, initialization and development are separate shared costs; not exclusive per-family optimizer wall.')
  for row in summ:quality_rows.append(dict(label=row['label'],seed=seed,iteration=row['iteration'],**quality(folder,row)))
 for row in e:quality_rows.append(dict(label=row['label'],role=row['role'],case=row['case'],**quality(S/'runs/eval-02',row)))
 prefixes={role:prefix(S/'runs/eval-02',role+'_prefix59_continue',role+'_prefix59_stop',8.4)for role in ['BC500','Sinteractive']};assert all(v[k]['all_fields_equal']for v in prefixes.values()for k in ['physics','control'])
 failedstats=rd(S/'runs/ppo-5401/rollout-1-stats.json');newphysics+=sum(r['physics_steps']for r in failedstats);saved_decisions=768+sum(v['actual_saved_RL_decisions']for v in seeds.values());assert saved_decisions==9216
 # dev4_none is printed/saved before entering batch5; absence proves stop was still inside evaluation.
 secondlog=(S/'logs/ppo-clean-5402.log').read_text();assert 'STUDENT_PPO 5402 4 3072'in secondlog and 'STUDENT_PPO_DEV 4 none'not in secondlog and not(S/'runs/ppo-clean-5402/rollout-5.npz').exists()
 jobs=ledger['jobs'];a=next(x for x in jobs if x['name']=='ppo-clean-5401');b=next(x for x in jobs if x['name']=='ppo-clean-5402');overlap=max(0,min(a['started']+a['elapsed_s'],b['started']+b['elapsed_s'])-max(a['started'],b['started']))
 evidence=dict(scope='Issue53 clean interactive-label continuation, then user-stopped PPO closeout',interactive_result=gate,selection=sel,BC_training=fit,BC_sample_allocation=rd(S/'runs/fit-01/sample-cost-counts.json'),BC_evaluation=e,PPO=seeds,PPO_family_costs=trainfamilies,quality=quality_rows,common_prefix_audit=prefixes,initial_identity=identity,Issue53_requested_cost_supplement=hist,operator_stop=rd(S/'user-early-stop.json'),PPO_final_sealed_evaluation_performed=False,paired_full_endpoint_comparison=False,resources=dict(ledger=ledger,supervised_updates=1000,new_full_queries=0,Student_RL_decisions_actual=saved_decisions,failed_RL_decisions=768,successful_seed_decisions={k:v['actual_saved_RL_decisions']for k,v in seeds.items()},RL_physics_steps_including_failed=newphysics,job_seconds_with_CPU_reserve=sum(j.get('elapsed_s',0)for j in ledger['jobs'])+ledger['cpu_analysis_reserve_s'],output_bytes=sum(p.stat().st_size for p in S.rglob('*')if p.is_file())),parallel=dict(baseline=rd(S/'parallel-baseline.json'),actual_overlap_s=overlap,note='Same immutable b175 runtime; per-process job-wall added, not replaced by makespan. Task mix and accepted SGD counts differ, so not a controlled speedup benchmark.'),limits='No further physical evaluation, search or training after user stop; saved-data analysis/render only; no next stage',source_runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip())
 (out/'evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,3,figsize=(12,6));cases=['short08','normal72','stop66']
 for col,c in enumerate(cases):
  for seed,color in [(5401,'C0'),(5402,'C1')]:
   rows=sorted([r for r in seeds[seed]['development']if r['label'].endswith('_'+c)],key=lambda r:r['iteration'])
   for ix,key in enumerate(['error_cm','force_rms']):axs[ix,col].plot([r['iteration']for r in rows],[r['all'][key]for r in rows],marker='o',label=f'seed {seed}'+(' (stopped4)'if seed==5402 else''),color=color)
  axs[0,col].set_title(c)
  for ix in range(2):axs[ix,col].grid(alpha=.2);axs[ix,col].set_ylabel('Active3D error (cm)'if ix==0 else'Actual force RMS (N)');axs[ix,col].set_xlabel('Student PPO batch')
 axs[0,0].legend();fig.suptitle('Same cleanBC500 start; seed1 fixed7, seed2 user-stopped4; development only');fig.tight_layout();fig.savefig(out/'learning.png',dpi=140);plt.close(fig)
 (out/'result.json').write_text(json.dumps(dict(status='completed_saved_data_only',new_physics=0,new_training=0,new_queries=0,wall_s=time.monotonic()-start),indent=2));print(json.dumps(dict(interactive_useful=gate['useful'],actual_RL_decisions=saved_decisions,parallel_overlap_s=overlap,prefixes=prefixes),indent=2))
if __name__=='__main__':main()
