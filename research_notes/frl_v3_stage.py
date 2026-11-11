"""Issue59 isolated scientific entry; archived snapshots/controllers never updated here."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ.setdefault('TMPDIR',str(STAGE/'tmp'));Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(STAGE/'torch-cache'))
import m11_ipc

def study(e,m,contract,out):
    import torch
    from native_env import install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    install();speed(e);torch.set_num_threads(1);torch.set_grad_enabled(False)
    (out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix')
    from frl_v3_counterfactual import run
    return run(e,m,contract,out,ARGS)

def main():
    global ARGS
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);ARGS=p.parse_args()
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
    ARGS.runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    ARGS.output.mkdir(parents=True,exist_ok=False)
    (ARGS.output/'launch.json').write_text(json.dumps(dict(runtime=ARGS.runtime,mode='frozen_counterfactual',started=time.time(),pid=os.getpid(),training=False),indent=2))
    os.environ.update(CLEAN_KIND='direct',P4000_CANONICAL_ROW_ACTOR='1',BFM_RECORD_NOMINAL_DIAGNOSTIC='0',BFM_RECOVERY_OUTPUT=str(ARGS.output))
    os.chdir(ARGS.output)
    import m10_stage,m10_physics
    m10_physics.OUT=ARGS.output;m10_stage.run_study=study
    try:m10_physics.main()
    except BaseException as exc:
        import traceback
        (ARGS.output/'failure.json').write_text(json.dumps(dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
