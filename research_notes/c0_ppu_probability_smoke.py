"""One cached on-policy batch numerical update; no new physics or formal seed result."""
import json,time
from pathlib import Path
import numpy as np,torch
from c0_ppu_probability import Expert
from humanoidverse.force_compliance.parallel import merge_rollouts
from m39b_update import update
B=Path('/root/compliance-ppu-20261106/issue47')
torch.set_num_threads(1);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
started=time.monotonic();p=B/'ppo4701';c=torch.load(p/'initial.pt',map_location='cuda',weights_only=True)
n=Expert(c['input_dim'],'direct').cuda();n.load_state_dict(c['state_dict'])
r=[dict(batch=dict(np.load(p/f'worker-{j}/rollout-0000.npz'))) for j in range(3)]
batch={k:v.cuda() for k,v in merge_rollouts(r,.95).items()};batch['raw_adv_std']=(batch['returns']-batch['value']).std()
before=n.distribution(batch['obs']).log_prob(batch['action']).sum(-1)
error=float((before-batch['logp']).abs().max());assert error<3e-4,error
opt=torch.optim.Adam([dict(params=n.actor.parameters(),lr=1e-5),dict(params=n.critic.parameters(),lr=3e-4)])
out=update(n,opt,batch,replay=None,bc_weight=0.)
out.update(initial_logprob_error=error,elapsed_s=time.monotonic()-started,new_physics_decisions=0,cached_samples=768,
           not_a_formal_training_model=True,network_and_std_unchanged=True)
(B/'probability-smoke.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)
