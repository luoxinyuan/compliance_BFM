"""New-process cold-start timing; does not flush shared OS caches or train."""
import time
ENTRY=time.monotonic()
import f1_l0 as base
import torch,numpy as np
from diagnostic_frozen import portable
from dual_track_runtime import TrainingBranch
from f1_mjdata_contract import install_restore_guard,canonical,metadata,exact

def study(e,m,c,out):
 model_env_ready=time.monotonic()-ENTRY
 from native_env import install
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 from frl_v2_high import build
 from frl_v3_risk import score
 import frl_v3_task
 install_restore_guard();install();speed(e);torch.set_num_threads(1);torch.set_grad_enabled(False)
 (out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix');frl_v3_task.configure(0.)
 t=frl_v3_task.Task(e,m,c,5801);n=build().eval().requires_grad_(False)
 p=base.OLD/'runs/frl-formal-B-01/candidate-0.pt';n.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model']);ready=time.monotonic()-ENTRY
 a=torch.load(base.V/'runs/counterfactual-02/state-01.pt',map_location='cpu',weights_only=False);torch.manual_seed(56017);np.random.seed(56017);t.reset(a['spec'],record=True,label='cold-P1-1')
 for raw in a['actual_raw_prefix']:t.step(torch.tensor(raw,device='cuda',dtype=torch.float32).reshape(1,-1))
 assert torch.equal(torch.cat([t.student_features(),t.features()],-1).cpu(),a['input']);source_ready=time.monotonic()-ENTRY
 ss=TrainingBranch(t);traces=[];walls=[]
 for repeat in range(2):
  ss.restore();start=time.monotonic();p=len(t.physics);cc=len(t.controls);at=e.tether.time
  for j in range(30):
   assert not t.done
   x=torch.cat([t.student_features(),t.features()],-1);t.step(n.distribution(x).mean)
  import copy
  ph=copy.deepcopy(t.physics[p:]);ctrl=copy.deepcopy(t.controls[cc:]);r=dict(physics=ph,controls=ctrl,score=score(ph,ctrl,at),state=canonical(portable(t),metadata(e.sim.model,e.sim.data)))
  torch.cuda.synchronize();walls.append(time.monotonic()-start);traces.append(r);torch.save(r,out/f'branch-{repeat}.pt')
 assert exact(traces[0],traces[1])
 result=dict(status='passed',process_entry_to_model_environment_s=model_env_ready,process_entry_to_usable_task_and_actor_s=ready,process_entry_to_restored_P1_1_s=source_ready,hot_3s_candidate_s=walls,cold_definition='fresh Python/CUDA/model/environment process; shared filesystem and driver caches NOT flushed',repeated_valid_state_exact=True,runtime=base.ARGS.runtime,bfm_sha=c['model_sha256'],actor_sha=base.sha(p) if isinstance(p,base.Path) else base.sha(base.OLD/'runs/frl-formal-B-01/candidate-0.pt'),training=0,permission='O-F frozen benchmark only')
 base.write(out/'checks.json',result);return result
if __name__=='__main__':base.study=study;base.main()
