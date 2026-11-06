"""Bounded same-input function/probability and discarded-clone update test."""
import argparse,json,gzip,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();start=time.monotonic()
import torch,numpy as np
from native_policy import Expert as Raw
from native_policy_coordinate import Expert as Coord
from clean_policy import update
from humanoidverse.force_compliance.parallel import merge_rollouts
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(4809)
ck=torch.load(a.root/'ppo-guided/initial.pt',map_location='cuda',weights_only=True);R=Raw(1088).cuda();R.load_state_dict(ck['state_dict']);C=Coord(1088).cuda();C.initialize_from_raw(ck['state_dict']);D=torch.from_numpy(np.load(a.root/'native-directions.npy')).cuda();results=[]
for i in range(3):
 z=np.load(a.root/f'ppo-guided/worker-{i}/rollout-0000.npz');results.append({'batch':{k:z[k] for k in z.files}})
b={k:v.cuda() for k,v in merge_rollouts(results,.95).items()};b['raw_adv_std']=(b['returns']-b['value']).std();x=b['obs'];checks=[]
with torch.no_grad():
 m0=R.raw_mean(x);assert torch.equal(m0,C.raw_mean(x));assert torch.equal(R.distribution(x).mean,C.distribution(x).mean)
 assert sum(v.numel() for v in R.actor.parameters())==sum(v.numel() for v in C.actor.parameters())
 for delta in [0.,.0001]:
  C.initialize_from_raw(ck['state_dict'])
  if delta:C.actor[-1].weight.normal_(0,delta);C.actor[-1].bias.normal_(0,delta)
  ds=[C.distribution(v[None]) for v in x];u=torch.cat([d.sample() for d in ds]);lp=torch.cat([d.log_prob(v[None]).sum(-1) for d,v in zip(ds,u)]);err=float((C.distribution(x).log_prob(u).sum(-1)-lp).abs().max());assert err<3e-4,err;checks.append({'delta_weight_std':delta,'joint_logprob_max':err})
C.initialize_from_raw(ck['state_dict']);changes=[]
for name,n in [('raw',R),('coordinate',C)]:
 torch.manual_seed(4811);opt=torch.optim.Adam([{'params':n.actor.parameters(),'lr':1e-5},{'params':n.critic.parameters(),'lr':3e-4}]);u=update(n,opt,b,epochs=2)
 with torch.no_grad():
  m=n.raw_mean(x);d=m-m0;nd=d@D.T;orth=d-nd@D;angle=torch.rad2deg(torch.acos(torch.nn.functional.cosine_similarity(m,m0).clamp(-1,1)))
 changes.append({'variant':name,'optimizer_updates_on_discarded_clone':u['optimizer_updates'],'joint_KL':u['epochs'][-1]['joint_gaussian_KL'],'mean_native_span_change':float(nd.norm(dim=-1).mean()),'mean_orthogonal_change':float(orth.norm(dim=-1).mean()),'mean_z_direction_deg':float(angle.mean()),'actor_clip_threshold':.5,'KL_threshold':.02,'original_logprob_threshold':3e-4})
result={'status':'passed','initial_raw_mean_bitwise_equal':True,'initial_Gaussian_mean_bitwise_equal':True,'trainable_actor_parameters':sum(v.numel() for v in C.actor.parameters()),'probability_checks':checks,'same_saved_batch_discarded_update':changes,'RL_decisions_added':0,'retained_training_updates_added':0,'new_method':'mean-head coordinate change, not a probability bug fix; all hidden layers train; no second actor or radius','seconds':time.monotonic()-start,'ended':time.time()}
(a.root/'native-coordinate-check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
