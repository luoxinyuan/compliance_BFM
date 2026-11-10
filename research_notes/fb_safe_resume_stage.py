"""P0-B new-process interruption regression within8179 actual online samples."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ.setdefault('TMPDIR',str(STAGE/'tmp'));Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
import m11_ipc

def study(e,m,contract,out):
 import torch
 from native_env import install
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 install();speed(e);torch.set_num_threads(1);torch.set_grad_enabled(False)
 (out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix')
 from fb_safe_resume import restore
 from diagnostic_fb_resume import fingerprints,publish
 from dual_track_fb_online import run_updates
 from dual_track_high import write
 agent,col,progress,meta=restore(ARGS.package,e,m,contract)
 updates=progress['completed_agent_updates'];before=fingerprints(agent,col,updates)
 expected=json.loads((ARGS.package.parent/'safe-stopped-fingerprints.json').read_text());assert before==expected
 col.collect(1011);stat=run_updates(agent,col,updates,16,compile=False);updates+=16
 after=fingerprints(agent,col,updates,stat['last_metrics']);write(out/'safe-resumed-fingerprints.json',after)
 expected=json.loads((ARGS.package.parent/'safe-after-fingerprints.json').read_text())
 assert after==expected,('Safe resumed collection/update differs',{k:(after[k],expected[k]) for k in expected if after[k]!=expected[k]})
 progress.update(collection_block_start=7168,collection_block_target=8192,collected_within_block=0,completed_agent_updates=updates)
 publish(out/'resumed-safe-7168',agent,col,updates,ARGS.runtime,progress,'user_stop',previous=ARGS.package)
 return dict(status='completed',new_process_safe_recovery_exact=True,new_actual_online_transitions=1011,
  engineering_updates=16,formal_FB_training=0,combined_online_transition_cap=8179)

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
