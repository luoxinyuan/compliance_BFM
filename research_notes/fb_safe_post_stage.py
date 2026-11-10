"""Nine new source prefixes, old native mean vs updated safe no-contact actor."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ.setdefault('TMPDIR',str(STAGE/'tmp'));Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
import m11_ipc

def study(e,m,contract,out):
 import torch,numpy as np,mujoco
 from native_env import install
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 install();speed(e);torch.set_num_threads(1);torch.set_grad_enabled(False)
 (out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix')
 from fb_safe_resume import restore
 from dual_track_task import sample_spec,digest
 from dual_track_high import write
 agent,col,progress,meta=restore(ARGS.package,e,m,contract)
 q_addresses=[int(e.sim.model.jnt_qposadr[mujoco.mj_name2id(e.sim.model,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in contract['dof_names']]
 rng=np.random.default_rng(58022);rows=[]
 for source in ('standing','goal_arm5','motion_walk_task'):
  for i in range(3):
   s=sample_spec(rng,source,'none');s['release']=s['source_until']+6.;s['end']=s['source_until']+8.
   paired_prefix=None
   for kind in ('native_mean','updated_safe_random'):
    if kind=='updated_safe_random' and paired_prefix is None:
     rows.append(dict(source=source,index=i,kind=kind,source_failure='native positive-control source unavailable'));write(out/'post-update-summaries.json',rows);continue
    col.proxy.policy_model=m if kind=='native_mean' else agent._model;col.proxy.stochastic=kind!='native_mean'
    try:col.reset(s)
    except RuntimeError as exc:
     rows.append(dict(source=source,index=i,kind=kind,source_failure=str(exc)));write(out/'post-update-summaries.json',rows);continue
    col.task.label='post-'+source+'-'+str(i)+'-'+kind
    prefix=col.task.prefix_data['audit']['sha256']
    if kind=='native_mean':paired_prefix=prefix
    else:assert prefix==paired_prefix,'Post-update paired native prefix differs'
    torch.manual_seed(58701+i);np.random.seed(58701+i)
    while not col.task.done:
     raw=col.task.native(source,float(e.tether.time))/16;col.proxy.take_step(raw)
    pp=col.task.physics;col.task.save(out)
    q=np.array([[x['qpos'][i] for i in q_addresses] for x in pp]);idx=agent._safe_action_transform.indices
    row=dict(source=source,index=i,kind=kind,prefix_sha256=prefix,boundary=col.task.boundary,
      native_controls=len(col.task.controls),physics_ticks=len(pp),
      root_delta=(np.array(pp[-1]['root'])-col.task.initial['root']).tolist(),
      loaded_slip_peak=max(x['stage_loaded_slip_mps'] for x in pp),
      ankle_min=q[:,idx].min(0).tolist(),ankle_max=q[:,idx].max(0).tolist(),
      force_max=max(np.linalg.norm(x['force']) for x in pp),
      q_PD_step_max_rad=max(x['q_PD_step_max_rad'] for x in col.task.controls))
    rows.append(row);write(out/'post-update-summaries.json',rows)
    print('SAFE_POST',source,i,kind,row['boundary'],flush=True)
 return dict(status='completed',actual_no_contact_validation_rows=len(rows),new_online_train_transitions=0,new_FB_updates=0,formal_FB_training=0)

def main():
 global ARGS
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--package',type=Path,required=True);ARGS=p.parse_args()
 import torch;torch.use_deterministic_algorithms(True)
 ARGS.runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 ARGS.output.mkdir(parents=True,exist_ok=False)
 os.environ.update(CLEAN_KIND='direct',P4000_CANONICAL_ROW_ACTOR='1',BFM_RECORD_NOMINAL_DIAGNOSTIC='0',BFM_RECOVERY_OUTPUT=str(ARGS.output))
 os.chdir(ARGS.output)
 import m10_stage,m10_physics
 m10_physics.OUT=ARGS.output;m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
