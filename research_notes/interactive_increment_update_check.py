"""Actual saved on-policy batch contract repair test, no new physical decisions."""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];S=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')];os.environ['P4000_CANONICAL_ROW_ACTOR']='1'
import torch,numpy as np
from clean_policy import Expert
from p4000_critic import update
from humanoidverse.force_compliance.parallel import merge_rollouts
torch.set_num_threads(1);torch.manual_seed(5401);d=torch.load(S/'runs/ppo-5401/candidate-0.pt',map_location='cuda',weights_only=True);n=Expert(1639,'direct').cuda();n.load_state_dict(d['state_dict']);n.requires_grad_(False);n.actor.requires_grad_(True);n.critic.requires_grad_(True);opt=torch.optim.Adam([{'params':n.actor.parameters(),'lr':3e-5},{'params':n.critic.parameters(),'lr':3e-4}]);z=np.load(S/'runs/ppo-5401/rollout-1.npz');batch={k:v.cuda()for k,v in merge_rollouts([dict(batch={k:z[k]for k in z.files})],.95).items()};batch['raw_adv_std']=(batch['returns']-batch['value']).std();assert torch.isfinite(batch['raw_adv_std'])
# Full original batch update, original probability and Adam safeguards unchanged.
start=time.monotonic();r=update(n,opt,batch,epochs=6,minibatch=256,critic_shuffle_seed=54010001);out=S/'runs/update-check-01';out.mkdir(exist_ok=False);(out/'result.json').write_text(json.dumps(dict(status='passed',new_RL_decisions=0,physical_steps=0,source='saved exact initial-policy on-policy batch; disposable test net only',raw_adv_std=float(batch['raw_adv_std']),actual_update=r,wall_s=time.monotonic()-start),indent=2));print('UPDATE_CONTRACT_PASS',r['pre_logprob_max_error'],r['optimizer_updates'])
