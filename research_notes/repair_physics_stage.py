"""P0-A same-state action and full real continuation after exact16 updates."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ.setdefault('TMPDIR',str(STAGE/'tmp'));Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
import m11_ipc


def study(e,m,contract,out):
 import torch,numpy as np
 from native_env import install as native_install
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 native_install();speed(e);torch.set_num_threads(1);torch.set_grad_enabled(False)
 (out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix')
 from diagnostic_fb_resume import restore,tree_digest
 from diagnostic_frozen import portable
 from dual_track_task import global_rng
 from dual_track_fb_online import cpu
 agent,col,progress,meta=restore(ARGS.package,e,m,contract)
 if col.task.done:
  from dual_track_task import sample_spec
  col.reset(sample_spec(np.random.default_rng(58091),'standing','mid'))
 obs=col.task.extended_observation();z=col.task.last_command
 a=torch.load(ARGS.reference_model,map_location='cpu',weights_only=True)
 b=torch.load(ARGS.resumed_model,map_location='cpu',weights_only=True)
 assert tree_digest(a)==tree_digest(b)
 agent._model.load_state_dict(a);first=agent._model.act(obs,z,mean=True)
 agent._model.load_state_dict(b);second=agent._model.act(obs,z,mean=True)
 assert torch.equal(first,second),'Equal recovered state produced different eager action'
 agent._model.load_state_dict(a if ARGS.branch=='reference' else b)
 # The16 updates contain no physical steps. The saved1064 physical state is
 # therefore still the real current state; choose the same new declared stream.
 torch.manual_seed(58091);np.random.seed(58091)
 commands=0
 while not col.task.done:
  raw=col.task.native(col.task.spec['source'],float(e.tether.time))/16
  col.proxy.take_step(raw);commands+=1
 col.task.label='repair-physical';col.task.save(out)
 torch.save(cpu(portable(col.task)),out/'physical-final.pt')
 result=dict(status='completed',branch=ARGS.branch,formal_training=0,new_FB_updates=0,
  same_state_actions_exact=True,model_tree_digest=tree_digest(agent._model.state_dict()),
  continuation_high_commands=commands,continuation_controls=len(col.task.controls),
  physical_tree_digest=tree_digest(col.task.physics),control_tree_digest=tree_digest(col.task.controls),
  final_state_tree_digest=tree_digest(portable(col.task)),rng_tree_digest=tree_digest(global_rng()),
  boundary=col.task.boundary,source=col.task.spec['source'],original_q_excess_boundary_rad=.02)
 (out/'physical-result.json').write_text(json.dumps(result,indent=2));return result


def main():
 global ARGS
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--package',type=Path,required=True)
 p.add_argument('--reference-model',type=Path,required=True);p.add_argument('--resumed-model',type=Path,required=True)
 p.add_argument('--branch',choices=['reference','resumed'],required=True);ARGS=p.parse_args()
 ARGS.runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 import torch;torch.use_deterministic_algorithms(True)
 ARGS.output.mkdir(parents=True,exist_ok=False)
 os.environ.update(CLEAN_KIND='direct',P4000_CANONICAL_ROW_ACTOR='1',BFM_RECORD_NOMINAL_DIAGNOSTIC='0',BFM_RECOVERY_OUTPUT=str(ARGS.output))
 os.chdir(ARGS.output)
 import m10_stage,m10_physics
 m10_physics.OUT=ARGS.output;m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
