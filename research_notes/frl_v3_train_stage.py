"""Issue59 conditional B-only training/evaluation or bounded lambda0 qualification."""
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
    import frl_v3_task
    groups=frl_v3_task.configure(ARGS.risk_lambda,ARGS.regression)
    (out/'parent-manifests.json').write_text(json.dumps(groups,indent=2))
    if ARGS.mode=='unit_checks':
        from frl_v3_checks import run
        return run(e,m,contract,out,ARGS)
    from frl_v3_high import train,evaluate
    return (train if ARGS.mode=='high_train' else evaluate)(e,m,contract,out,ARGS)

def main():
    global ARGS
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['unit_checks','high_train','high_eval'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=['B'],default='B');p.add_argument('--kl',type=float,choices=[.02],default=.02);p.add_argument('--checkpoint',type=Path);p.add_argument('--sha256');p.add_argument('--iteration',type=int,default=0);p.add_argument('--split',choices=['dev'],default='dev');p.add_argument('--record',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--reward-gate',type=Path);p.add_argument('--risk-lambda',type=float,choices=[0.,.33],default=.33);p.add_argument('--regression',action='store_true');p.add_argument('--p1-output',type=Path,default=STAGE/'runs/counterfactual-02');ARGS=p.parse_args()
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
    ARGS.runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    ARGS.p1_gate=json.loads((ARGS.p1_output/'gate.json').read_text());verification=json.loads((ARGS.p1_output/'counterfactual-verification.json').read_text());assert ARGS.p1_gate['P2_allowed'] and verification['all_archived_continuations_bitwise']
    if ARGS.mode=='high_train':
        assert ARGS.reward_gate is not None
        gate=json.loads(ARGS.reward_gate.read_text());assert gate['formal_RL_launch_allowed'] and gate['status']=='qualified_matched_physical_reward'
        assert gate['nonreward_correctness_passed']
    ARGS.output.mkdir(parents=True,exist_ok=False)
    (ARGS.output/'launch.json').write_text(json.dumps(dict(runtime=ARGS.runtime,mode=ARGS.mode,risk_lambda=ARGS.risk_lambda,started=time.time(),pid=os.getpid(),smoke=ARGS.smoke,p1_gate_qualified=True),indent=2))
    os.environ.update(CLEAN_KIND='direct',P4000_CANONICAL_ROW_ACTOR='1',BFM_RECORD_NOMINAL_DIAGNOSTIC='0',BFM_RECOVERY_OUTPUT=str(ARGS.output))
    os.chdir(ARGS.output)
    import m10_stage,m10_physics
    m10_physics.OUT=ARGS.output;m10_stage.run_study=study
    try:m10_physics.main()
    except BaseException as exc:
        import traceback
        (ARGS.output/'failure.json').write_text(json.dumps(dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
