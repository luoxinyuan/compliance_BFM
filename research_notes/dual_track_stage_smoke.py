"""Actual source/history/replay/terminal smoke; no policy updates or RL rollout."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc


def study(e,m,contract,out):
 import torch,numpy as np
 from native_env import install
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 from dual_track_task import Task,parent_manifests,CFG
 from dual_track_native_data import NativeActor,NativeSequences
 from dual_track_runtime import TrainingBranch
 install();speed(e);torch.set_num_threads(1);torch.set_grad_enabled(False)
 (out/'prefix').mkdir();backend(e,m,out/'prefix')
 manifest=parent_manifests();(out/'parent-manifests.json').write_text(json.dumps(manifest,indent=2))
 assert len(manifest['dev'])==30 and len(manifest['sealed'])==60
 task=Task(e,m,contract,5607);proxy=NativeActor(task,m);task.m=proxy;sequences=NativeSequences();results=[]
 for source in CFG['source_probs']:
  s=next(r['spec'] for r in manifest['dev'] if r['source']==source)
  cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
  proxy.begin_source();task.reset(s,record=True,label=source);proxy.finish_pending()
  assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
  sequences.add(source,s['source_parent_id'],proxy.source_pairs);proxy.begin_online()
  records,done,stats=proxy.take_step(task.native(source,float(e.tether.time))/16)
  assert len(records)==5 and not done
  assert all(not r['next']['terminated'].item() for r in records)
  before=TrainingBranch(task);q=e.sim.data.qpos.copy();f=task.features().clone()
  task.reset(s,record=True,label='disturb');before.restore()
  assert np.array_equal(e.sim.data.qpos,q) and torch.equal(task.features(),f)
  results.append(dict(source=source,actual_source_history_sha=task.prefix_data['audit']['sha256'],
     actual_native_records=len(records),source_records=len(proxy.source_pairs),native_reward_sum=sum(float(r['aux_rewards']['contact_physical']) for r in records),
     complete_branch_restore=True))
 # A physical early-stop injection tests the same path as a genuine unsafe tick:
 # preserve that first state, finish observation bookkeeping, defer autoreset.
 task.inject_terminal_tick=float(e.tether.time)+.005
 records,done,stats=proxy.take_step(task.native(source,float(e.tether.time))/16)
 assert done and stats['terminated'] and not stats['truncated']
 assert records[-1]['next']['terminated'].item() and len(records)==1
 assert np.array_equal(e.sim.data.qpos,np.asarray(task.physics[-1]['qpos']))
 assert task.native_reset_deferred and task.native_info['final_observation'] is not None
 assert abs(float(records[-1]['duration'])-.005)<1e-8
 assert all(torch.equal(v.cpu(),records[-1]['next']['observation'][k]) for k,v in task.native_info['final_observation'].items())
 task.inject_terminal_tick=None
 # Fresh source clears the old callback/terminal state, then an administrative
 # time limit bootstraps from the true successor without crossing reset.
 proxy.begin_source();task.reset(s,record=True,label='time-limit-test');proxy.finish_pending();proxy.begin_online()
 task.spec['end']=float(e.tether.time)+.1
 records,done,stats=proxy.take_step(task.native(source,float(e.tether.time))/16)
 assert done and stats['truncated'] and not stats['terminated'] and not task.stage_boundary
 assert records[-1]['next']['truncated'].item()
 sample=sequences.sample(1024);assert sample['observation']['contact_privileged_v1'].shape==(1024,97)
 result=dict(status='passed',source_checks=results,real_pre_reset_terminal_successor=True,
    first_unsafe_tick_preserved_without_further_physics=True,administrative_truncation_distinct=True,
    source_balanced_CPR_sequences=True,reset_preserves_external_training_RNG=True,manifests=dict(dev=30,sealed=60,actual_scientific_evaluations=0),
    optimizer_steps=0,formal_RL_decisions=0,formal_FB_transitions=0)
 (out/'verification.json').write_text(json.dumps(result,indent=2));return dict(status='completed',verification=result,student_optimizer_updates=0)


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert a.output.is_dir() and not (a.output/'verification.json').exists()
 os.environ.update(CLEAN_KIND='direct',P4000_CANONICAL_ROW_ACTOR='1',BFM_RECORD_NOMINAL_DIAGNOSTIC='0',BFM_RECOVERY_OUTPUT=str(a.output));os.chdir(a.output)
 import m10_stage,m10_physics
 m10_physics.OUT=a.output;m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
