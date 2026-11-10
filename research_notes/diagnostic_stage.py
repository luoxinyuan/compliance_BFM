"""Issue57 isolated diagnostic entry; never starts formal PPO/1M training."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
STAGE=P/'experiments/dual-track-diagnostics-20261110'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(STAGE/'torch-cache'))
os.environ.setdefault('TMPDIR',str(STAGE/'tmp'));Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
import m11_ipc

def study(e,m,c,out):
 import torch
 from native_env import install
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 install();speed(e);torch.set_num_threads(1);torch.set_grad_enabled(False)
 (out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix')
 if ARGS.mode=='d1':
  from diagnostic_frozen import run
 elif ARGS.mode=='d2audit':
  from diagnostic_fb_audit import run
 else:
  from diagnostic_fb_resume import run
 return run(e,m,c,out,ARGS)

def main():
 global ARGS
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['d1','d2audit','d2control','d2resume'],required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--package',type=Path)
 ARGS=p.parse_args();ARGS.runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 if ARGS.mode in ('d2control','d2resume'):
  # Exact new-process training regression needs deterministic backward kernels,
  # not just identical RNG. This affects only the new engineering entry.
  os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
  import torch
  torch.use_deterministic_algorithms(True)
 ARGS.output.mkdir(parents=True,exist_ok=False)
 (ARGS.output/'launch.json').write_text(json.dumps(dict(mode=ARGS.mode,runtime=ARGS.runtime,pid=os.getpid(),start=time.time()),indent=2))
 os.environ.update(CLEAN_KIND='direct',P4000_CANONICAL_ROW_ACTOR='1',BFM_RECORD_NOMINAL_DIAGNOSTIC='0',BFM_RECOVERY_OUTPUT=str(ARGS.output))
 os.chdir(ARGS.output)
 import m10_stage,m10_physics
 m10_physics.OUT=ARGS.output;m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
