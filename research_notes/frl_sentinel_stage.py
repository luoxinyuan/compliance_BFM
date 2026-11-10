"""Exposed62cm/.22m/s sentinel only. Never a development selector or new holdout."""
import argparse,gzip,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ.setdefault('TMPDIR',str(STAGE/'tmp'));Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
import m11_ipc

def study(e,m,contract,out):
 import torch,numpy as np
 from native_env import install
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 install();speed(e);torch.set_num_threads(1);torch.set_grad_enabled(False)
 (out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix')
 from frl_v2_task import configure,Task
 configure()
 from frl_v2_high import build,sha,write
 from native_transfer import phase_report
 from dual_track_quality import quality
 assert sha(ARGS.checkpoint)==ARGS.sha256
 ck=torch.load(ARGS.checkpoint,map_location='cpu',weights_only=False);net=build();net.load_state_dict(ck['model']);net.eval().requires_grad_(False)
 old=STAGE.parents[0]/'dual-track-diagnostics-20261110/runs/d1-01/3ffd2fc10497-i0.json.gz'
 s=json.load(gzip.open(old,'rt'))['condition']
 assert s['source']=='standing' and .62<s['distance']<.63
 assert abs(s['distance']/s['ramp']-.22)<1e-8
 task=Task(e,m,contract,58017);task.reset(s,record=True,label='exposed62cm-sentinel')
 while not task.done:
  x=torch.cat([task.student_features(),task.features()],-1);task.step(net.distribution(x).mean)
 task.save(out)
 report=dict(status='completed',scope='exposed regression sentinel only; not selection or holdout success',
  checkpoint_sha256=ARGS.sha256,original_train_runtime=ck['counters']['runtime'],original_iteration=ck['counters']['iteration'],
  condition=s,quality=quality(task),boundary=task.boundary,**phase_report(task))
 write(out/'sentinel-summary.json',report);return report

def main():
 global ARGS
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--sha256',required=True);ARGS=p.parse_args()
 ARGS.runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 ARGS.output.mkdir(parents=True,exist_ok=False)
 os.environ.update(CLEAN_KIND='direct',P4000_CANONICAL_ROW_ACTOR='1',BFM_RECORD_NOMINAL_DIAGNOSTIC='0',BFM_RECOVERY_OUTPUT=str(ARGS.output))
 os.chdir(ARGS.output)
 import m10_stage,m10_physics
 m10_physics.OUT=ARGS.output;m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
