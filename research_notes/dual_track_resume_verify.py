"""Fresh-process restore of the new checkpoint contract; no training/query."""
import os,sys,json,argparse,time,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc

def study(e,m,contract,out):
 import torch,numpy as np
 from p4000_env import Task
 from native_env import install
 from dual_track_runtime import PrivilegedBranch,restore_training_checkpoint
 from retention_eval import MODELS
 from rolling_backend import install as backend
 from m11_runtime_speed import install as speed
 torch.set_num_threads(1);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;install();speed(e)
 (out/'prefix').mkdir();b=backend(e,m,out/'prefix')
 task=Task(e,m,contract,5501);net=PrivilegedBranch(torch.load(MODELS['BC500'][0],map_location='cpu',weights_only=True)).cuda();g=torch.Generator(device='cuda')
 t=time.monotonic();counters=restore_training_checkpoint(ARGS.checkpoint,task,net,None,[g]);elapsed=time.monotonic()-t
 ck=torch.load(ARGS.checkpoint,map_location='cpu',weights_only=False);r=ck['rng']
 assert random.getstate()==r['python']
 ns=np.random.get_state();assert ns[0]==r['numpy'][0] and np.array_equal(ns[1],r['numpy'][1]) and ns[2:]==r['numpy'][2:]
 assert torch.equal(torch.get_rng_state(),r['cpu']) and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),r['cuda']))
 assert task.rng.bit_generator.state==r['task'] and torch.equal(g.get_state(),r['generators'][0])
 assert all(torch.equal(v.detach().cpu(),ck['model'][k]) for k,v in net.state_dict().items())
 result=dict(new_process=True,physics_history_digest_exact=True,all_rng_exact=True,model_state_exact=True,counters=counters,restore_s=elapsed,backend=b,formal_training=False)
 (out/'verification.json').write_text(json.dumps(result,indent=2));return dict(status='completed',verification=result,student_optimizer_updates=0)
def main():
 global ARGS
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);ARGS=p.parse_args();out=ARGS.output;out.mkdir(exist_ok=False)
 os.environ.update(CLEAN_KIND='direct',P4000_CANONICAL_ROW_ACTOR='1',BFM_RECOVERY_OUTPUT=str(out));os.chdir(out)
 import m10_stage,m10_physics
 m10_physics.OUT=out;m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
