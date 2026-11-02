"""Replay the saved real PPO batch; no BFM inference or new rollout."""
import time,json,sys,os
from pathlib import Path
import torch
from narrow_policy import Expert,update
start=time.monotonic();torch.set_num_threads(1)
r=Path(sys.argv[1]);initial=torch.load(r/'initial.pt',weights_only=True,map_location='cpu');batch=torch.load(r/'audit-batch.pt',weights_only=True,map_location='cpu');rows=[]
for lr in [3e-5,1e-5]:
 n=Expert(initial['input_dim']);n.load_state_dict(initial['state_dict']);n.log_std.requires_grad_(False)
 opt=torch.optim.Adam([dict(params=n.actor.parameters(),lr=lr),dict(params=n.critic.parameters(),lr=3e-4)])
 torch.manual_seed(17);u=update(n,opt,batch,epochs=1,minibatch=len(batch['reward']))
 rows.append(dict(actor_lr=lr,**u))
out=dict(scope='same real batch, one full-batch controlled update, CPU policy only, no BFM or new rollout',seconds=time.monotonic()-start,rows=rows)
(r/'same-batch-update-check.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
