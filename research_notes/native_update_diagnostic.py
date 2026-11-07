"""Read-only gradients/parameter changes on saved development rollout states.
No optimizer.step, BFMinference, physics or new PPO samples.
"""
import time,argparse,json,hashlib
from pathlib import Path
START=time.monotonic()
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
import torch,numpy as np
from native_policy import Expert
from humanoidverse.force_compliance.parallel import merge_rollouts
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
D=torch.from_numpy(np.load(a.root/'native-directions.npy')).cuda();result={'scope':'saved on-policy rollout0 gradients plus same-input checkpoint function change; not causal physical intervention','models':[],'sampling_decisions_added':0,'optimizer_updates_added':0}
for arm in ['ordinary','guided']:
 root=a.root/f'ppo-{arm}';ck=torch.load(root/'initial.pt',map_location='cuda',weights_only=True);n=Expert(ck['input_dim']).cuda();n.load_state_dict(ck['state_dict']);results=[]
 for wid in range(3):
  z=np.load(root/f'worker-{wid}/rollout-0000.npz');results.append({'batch':{k:z[k] for k in z.files}})
 batch={k:v.cuda() for k,v in merge_rollouts(results,.95).items()};x=batch['obs'];u=batch['action'];old=batch['logp'];adv=batch['advantages'];dist=n.distribution(x);err=float((dist.log_prob(u).sum(-1)-old).abs().max());assert err<3e-4,err
 loss=-(adv.detach()*dist.log_prob(u).sum(-1)).mean();n.zero_grad();loss.backward();G=torch.cat([n.actor[-1].weight.grad,n.actor[-1].bias.grad[:,None]],dim=1);GD=D@G;orth=G-D.T@GD
 trace_diag=n.raw_L.square().sum(-1);theory_inside=4/(.008**2+.15**2) if arm=='guided' else 4/.008**2;theory_outside=252/.008**2
 with torch.no_grad():m0=n.raw_mean(x);mu0=n.distribution(x).mean
 row={'arm':arm,'saved_rollout0_samples':len(x),'joint_logprob_max_error':err,'PPO_score_gradient_actor_norm':float(torch.sqrt(sum((p.grad**2).sum() for p in n.actor.parameters()))),'last_head_gradient_native_span_norm':float(GD.norm()),'last_head_gradient_orthogonal_norm':float(orth.norm()),'last_head_native_gradient_energy_fraction':float(GD.square().sum()/G.square().sum()),'Gaussian_score_covariance_trace_native_subspace':theory_inside,'Gaussian_score_covariance_trace_orthogonal':theory_outside,'changes':[]}
 for it in [8,16,24,32]:
  f=root/f'candidate-{it:04d}.pt'
  if not f.exists():continue
  c=torch.load(f,map_location='cuda',weights_only=True);other=Expert(ck['input_dim']).cuda();other.load_state_dict(c['state_dict'])
  with torch.no_grad():
   m=other.raw_mean(x);delta=m-m0;native=delta@D.T;perp=delta-native@D;angle=torch.rad2deg(torch.acos(torch.nn.functional.cosine_similarity(m,m0,dim=-1).clamp(-1,1)))
   row['changes'].append({'iteration':it,'actual_raw_mean_change_rms':float(delta.square().mean().sqrt()),'whitened_mean_change_rms':float((other.distribution(x).mean-mu0).square().mean().sqrt()),'mean_native_span_change_norm':float(native.norm(dim=-1).mean()),'mean_orthogonal_change_norm':float(perp.norm(dim=-1).mean()),'legal_z_direction_change_mean_deg':float(angle.mean()),'legal_z_direction_change_peak_deg':float(angle.max()),'states':'same actual rollout0 observations, not different trajectories'})
  del other
 result['models'].append(row);del n
result.update(seconds=time.monotonic()-START,ended=time.time(),hypothesis_boundary='score covariance/gradient dominance is not proof of unique optimization cause; physical control outside candidate span can be useful. Whitened and raw units must not be conflated.')
(a.root/'update-geometry-diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
