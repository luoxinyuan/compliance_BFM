"""Issue58 bounded new-process FB repair regression, not formal FB training."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(STAGE/'torch-cache'))
os.environ.setdefault('TMPDIR',str(STAGE/'tmp'));Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc


def study(e,m,c,out):
 import torch
 from native_env import install as native_install
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 native_install();speed(e);torch.set_num_threads(1);torch.set_grad_enabled(False)
 (out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix')
 import diagnostic_fb_resume as regression
 import dual_track_fb_online as online
 from fb_ordered_grad import install, VERSION
 from diagnostic_fb_resume import tree_digest
 live={}
 old_publish=regression.publish
 def publish(*args,**kwargs):
  live['agent'],live['collector']=args[1],args[2]
  return old_publish(*args,**kwargs)
 regression.publish=publish
 factory=online.factory
 def new_factory(*a,**kw):
  agent,manifest=factory(*a,**kw)
  return install(agent),manifest
 regression.factory=new_factory;online.factory=new_factory
 agents=[]
 def observed_factory(*a,**kw):
  agent,manifest=new_factory(*a,**kw);agents.append(agent);return agent,manifest
 regression.factory=observed_factory;online.factory=observed_factory
 original_restore=regression.restore
 def restored(*a,**kw):
  agent,col,progress,meta=original_restore(*a,**kw)
  agent=install(agent);agents.append(agent)
  return agent,col,progress,meta
 regression.restore=restored
 # Preserve exact physical, replay, RNG and all optimizer checks of Issue57.
 started=time.monotonic()
 try:result=regression.run(e,m,c,out,ARGS)
 finally:
  (out/'optimizer-gradients.json').write_text(json.dumps([a._repair_events for a in agents],indent=2))
 agent,c=live['agent'],live['collector']
 from dual_track_task import global_rng,restore_global_rng
 from dual_track_fb_online import cpu
 from trace_update import Fixed
 cache=c.sequences.sample(1024);batch=c.replay.sample(1024)
 identities=dict(expert=tree_digest(cache),train=tree_digest(batch),rng=tree_digest(global_rng()))
 (out/'fixed16-inputs.json').write_text(json.dumps(identities,indent=2))
 torch.save(dict(expert=cpu(cache),train=cpu(batch),rng=global_rng()),out/'fixed16-inputs.pt')
 if ARGS.compiled:
  agent.cfg=agent.cfg.model_copy(update={'compile':True});agent.setup_compile()
 curves=[]
 for i in range(16):
  begun=time.monotonic()
  with torch.enable_grad():metrics=agent.update(dict(expert_slicer=Fixed(cache),train=Fixed(batch)),i+2)
  torch.cuda.synchronize()
  curves.append(dict(index=i,model=tree_digest(agent._model.state_dict()),adam=tree_digest(agent.optimizer_dict),
   rng=tree_digest(global_rng()),gradients=agent._repair_events[-6:],update_s=time.monotonic()-begun))
  (out/'fixed16-results.json').write_text(json.dumps(curves,indent=2))
  print('REPAIR_FIXED16',i,ARGS.compiled,curves[-1]['update_s'],flush=True)
 torch.save(cpu(agent._model.state_dict()),out/'fixed16-model.pt')
 torch.save(cpu(agent.optimizer_dict),out/'fixed16-optimizers.pt')
 result['fixed_input_continuous_updates']=16
 result['compiled_validation_requested']=ARGS.compiled
 result['engineering_FB_updates']+=16
 result.update(repair=VERSION,mode_validation='compiled' if ARGS.compiled else 'eager',original_strict_gate_unchanged=True,
               elapsed_s=time.monotonic()-started)
 (out/'repair-result.json').write_text(json.dumps(result,indent=2))
 return result


def main():
 global ARGS
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['d2control','d2resume'],required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--package',type=Path);p.add_argument('--compiled',action='store_true')
 ARGS=p.parse_args();ARGS.runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 import torch
 torch.use_deterministic_algorithms(True)
 ARGS.output.mkdir(parents=True,exist_ok=False)
 (ARGS.output/'launch.json').write_text(json.dumps(dict(runtime=ARGS.runtime,mode=ARGS.mode,
  source='8dd48a0',pid=os.getpid(),start=time.time(),formal_training=False),indent=2))
 os.environ.update(CLEAN_KIND='direct',P4000_CANONICAL_ROW_ACTOR='1',BFM_RECORD_NOMINAL_DIAGNOSTIC='0',BFM_RECOVERY_OUTPUT=str(ARGS.output))
 os.chdir(ARGS.output)
 import m10_stage,m10_physics
 m10_physics.OUT=ARGS.output;m10_stage.run_study=study
 try:m10_physics.main()
 except BaseException as exc:
  import traceback
  (ARGS.output/'failure.json').write_text(json.dumps(dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()),indent=2))
  raise
if __name__=='__main__':main()
