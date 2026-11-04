"""Finite update test on a saved calibration batch; no new physics or candidate model."""
import argparse,json,os,time,copy,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',required=True);p.add_argument('--resume');p.add_argument('--kind');p.add_argument('--stage');p.add_argument('--iterations');p.add_argument('--seed');p.add_argument('--max-seconds');a=p.parse_args()
 a.output.mkdir();os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,CLEAN_KIND='direct');started=time.monotonic()
 import torch
 from m39b_replay import Replay,sha
 from m39b_update import Expert,update
 from frozen_bfm_placement import tensor_digest,INFERENCE_MODULES
 torch.set_num_threads(1);torch.set_grad_enabled(False)
 plan=json.loads(Path(__file__).with_name('m39b_protocol.json').read_text());calp=Path(os.environ['M39B_CALIBRATION']);cal=json.loads(calp.read_text());root=calp.parent
 ready=json.loads((root/'workers.json').read_text())[0]
 replay=Replay(plan,ready['contract'],ready['standing'],a.output,'A')
 ck=torch.load(plan['initial']['path'],map_location='cuda',weights_only=True);batch={k:v.cuda() for k,v in torch.load(root/'calibration-batch.pt',weights_only=True).items()}
 frozen={n:tensor_digest(getattr(replay.model,n)) for n in INFERENCE_MODULES};rows=[]
 for kind,weight in [('raw',.002),('action',cal['action_weight'])]:
  torch.manual_seed(3911);replay.gen.manual_seed(391101);replay.kind=kind
  net=Expert(1088,'direct').cuda();net.load_state_dict(ck['state_dict']);initial=replay.state_sha(net);assert initial==cal['initial_state_sha']
  opt=torch.optim.Adam([dict(params=net.actor.parameters(),lr=plan['training']['actor_lr']),dict(params=net.critic.parameters(),lr=plan['training']['critic_lr'])])
  stats=update(net,opt,batch,epochs=1,replay=replay,bc_weight=weight)
  assert stats['optimizer_updates']>0 and stats['gradient_components'][0]['weighted_retention']>0
  assert stats['pre_logprob_max_error']<3e-4 and stats['epochs'][-1]['joint_gaussian_KL']<=.02001
  assert replay.state_sha(net)!=initial
  assert all(p.grad is None and not p.requires_grad for p in replay.model.parameters())
  rows.append(dict(kind=kind,weight=weight,initial_state_sha=initial,update=stats,diagnostics=replay.diagnostics(net)))
 assert {n:tensor_digest(getattr(replay.model,n)) for n in INFERENCE_MODULES}==frozen
 assert sha(plan['initial']['path'])==plan['initial']['sha256']
 result=dict(status='completed',passed=True,decisions=0,new_physics=0,diagnostic_optimizer_steps=sum(r['update']['optimizer_updates'] for r in rows),rows=rows,elapsed_s=time.monotonic()-started,frozen_bfm_unchanged=True,source_checkpoint_unchanged=True)
 (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
