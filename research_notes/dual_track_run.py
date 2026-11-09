"""Pinned stage entry; single actual MuJoCo environment, no shared asset writes."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc


def study(e,m,contract,out):
 import torch
 from native_env import install
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 install();speed(e);torch.set_num_threads(1);torch.set_grad_enabled(False)
 (out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix')
 if ARGS.mode.startswith('high_'):
  from dual_track_high import train,evaluate
  return (train if ARGS.mode=='high_train' else evaluate)(e,m,contract,out,ARGS)
 from dual_track_fb_online import run
 return run(e,m,contract,out,ARGS)


def main():
 global ARGS
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('high_train','high_eval','fb_smoke','fb_train'),required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--kl',type=float,choices=(.02,.08))
 p.add_argument('--checkpoint',type=Path);p.add_argument('--sha256');p.add_argument('--iteration',type=int,default=0)
 p.add_argument('--split',choices=('dev','sealed'),default='dev');p.add_argument('--record',action='store_true')
 ARGS=p.parse_args();assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 ARGS.runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 if not ARGS.output.exists():ARGS.output.mkdir()
 assert not (ARGS.output/'launch.json').exists()
 (ARGS.output/'launch.json').write_text(json.dumps(dict(runtime=ARGS.runtime,mode=ARGS.mode,kl=ARGS.kl,protocol='dual_track_formal_run_v1.json'),indent=2))
 if ARGS.mode in ('high_train','fb_train'):
  from dual_track_admission import require_launchable
  require_launchable(json.loads((ROOT/'research_notes/dual_track_formal_contract_v1.json').read_text()),'frozen' if ARGS.mode=='high_train' else 'fb')
 os.environ.update(CLEAN_KIND='direct',P4000_CANONICAL_ROW_ACTOR='1',BFM_RECORD_NOMINAL_DIAGNOSTIC='0',BFM_RECOVERY_OUTPUT=str(ARGS.output))
 os.chdir(ARGS.output)
 import m10_stage,m10_physics
 m10_physics.OUT=ARGS.output;m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
