"""Matched raw or five-substep q_PD retention on identical measured histories.
Frozen observation replay is not a physical rollout. No BFM parameter is trained.
"""
import gzip,json,hashlib,math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from clean_policy import direct_command

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def interp(p,z):
 a=torch.arange(1,6,device=z.device,dtype=z.dtype)[None,:,None]/5
 dot=(p*z).sum(-1)/256
 if bool((dot < -.9999).any()):raise ValueError('Antipodal command interpolation is not qualified')
 theta=torch.acos(dot.clamp(-1+1e-7,1-1e-7))[:,None,None]
 zz=(torch.sin((1-a)*theta)*p[:,None]+torch.sin(a*theta)*z[:,None])/torch.sin(theta)
 zz=16*zz/zz.norm(dim=-1,keepdim=True)
 return torch.cat([zz[:,:4],z[:,None]],1)

class Replay:
 def __init__(self,plan,contract,standing,out,arm):
  from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
  from frozen_bfm_placement import place_inference_modules
  from m10_physics import MODEL
  self.kind='raw' if arm=='A' else 'action';self.device='cuda';self.out=out
  assert sha(MODEL/'checkpoint/model/model.safetensors')==contract['model_sha256']=='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361'
  # Loaded in both arms, so model residency and diagnostics are matched.
  self.model=load_model_from_checkpoint_dir(MODEL/'checkpoint',device='cpu')
  self.placement=place_inference_modules(self.model)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  self.model.eval().requires_grad_(False)
  self.contract=contract;self.scale=torch.tensor(contract['q_PD_scale'],device='cuda');self.offset=torch.tensor(contract['q_PD_offset'],device='cuda');self.standing=torch.tensor(standing,device='cuda')[None]
  path=Path(plan['replay']['path']);self.data_sha=sha(path);assert self.data_sha==plan['replay']['sha256']
  with gzip.open(path,'rt') as f:rows=json.load(f)
  rows=[r for r in rows if not r['episode'].startswith('D2_') or (all(c['post_active'] for c in r['sequence']) and sum(v*v for v in r['x'][1041:1050])**.5>.02)]
  self.n=len(rows);assert self.n>0
  self.x=torch.tensor([r['x'] for r in rows],device='cuda');self.raw=torch.tensor([r['raw'] for r in rows],device='cuda');self.z=torch.tensor([r['z'] for r in rows],device='cuda');self.prev=torch.tensor([r['previous_z'] for r in rows],device='cuda')
  assert (self.x[:,565:821]*16-self.prev).abs().max()<1e-5,'Wrong actual executed previous z in input'
  self.qt=torch.tensor([[c['q_PD'] for c in r['sequence']] for r in rows],device='cuda')
  self.at=torch.tensor([[c['action'] for c in r['sequence']] for r in rows],device='cuda')
  self.mask=torch.arange(5,device='cuda')[None,:]<torch.tensor([r['valid_steps'] for r in rows],device='cuda')[:,None]
  obs={k:torch.tensor([[c['bfm_obs'][k][0] for c in r['sequence']] for r in rows],device='cuda') for k in contract['bfm_inputs']}
  with torch.no_grad():
   norm=self.model._normalize({k:v.reshape(-1,*v.shape[2:]) for k,v in obs.items()})
   self.obs={k:v.detach().reshape(self.n,5,*v.shape[1:]) for k,v in norm.items()}
  self.gen=torch.Generator(device='cuda').manual_seed(391101)
  self.fixed=torch.linspace(0,self.n-1,min(128,self.n),device='cuda').round().long()
  (out/'replay.json').write_text(json.dumps(dict(samples=self.n,data_sha256=self.data_sha,episodes=dict(Counter(r['episode'] for r in rows)),kind=self.kind,placement=self.placement,scope='real source trajectories only; no failed combination labels; fixed measured observations not physical unroll'),indent=2))
 def sample(self,n):return torch.randint(self.n,(n,),device='cuda',generator=self.gen)
 def actions(self,z,ix):
  zz=interp(self.prev[ix],z)
  ob={k:v[ix].reshape(-1,*v.shape[2:]) for k,v in self.obs.items()}
  a=self.model._actor(ob,zz.reshape(-1,256),self.model.cfg.actor_std).mean.float()
  q=self.offset+(a*5).clamp(-self.contract['clip'],self.contract['clip'])*self.scale/5
  return a.reshape(-1,5,29),q.reshape(-1,5,29)
 def loss(self,net,ix,kind=None):
  d=net.distribution(self.x[ix]);kind=kind or self.kind
  if kind=='raw':return .5*((d.mean-self.raw[ix])/d.stddev).square().sum(-1).mean()
  z=direct_command(d.mean,self.standing);_,q=self.actions(z,ix)
  return (q-self.qt[ix])[self.mask[ix]].square().mean()
 @staticmethod
 def state_sha(net):
  h=hashlib.sha256()
  for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 def diagnostics(self,net):
  with torch.no_grad():
   ix=self.fixed;r=net.distribution(self.x[ix]).mean;z=direct_command(r,self.standing);a,q=self.actions(z,ix)
   return dict(raw_norm_mean=float(r.norm(dim=-1).mean()),raw_target_norm_mean=float(self.raw[ix].norm(dim=-1).mean()),raw_rms=float((r-self.raw[ix]).square().mean().sqrt()),z_angle_deg=float(torch.rad2deg(torch.acos((z*self.z[ix]).sum(-1).div(256).clamp(-1,1))).mean()),qpd_rms_deg=float(torch.rad2deg((q-self.qt[ix])[self.mask[ix]]).square().mean().sqrt()),action_rms=float((a-self.at[ix])[self.mask[ix]].square().mean().sqrt()))
 @staticmethod
 def count_phases(results):
  c=Counter();f=Counter()
  for r in results:
   for s in r['stats']:c[s['phase']]+=1;f[s['demand_family']]+=1
  return dict(decisions_by_phase=dict(c),decisions_by_family=dict(f),completed_episodes_by_family=dict(Counter(e['condition'].get('demand_family','unspecified') for r in results for e in r['episodes'])))

def grad(loss,net):
 values=torch.autograd.grad(loss,tuple(net.actor.parameters()),allow_unused=False)
 return torch.cat([v.reshape(-1) for v in values])
def calibrate(net,replay,batch,out,plan):
 from locomotion_rollout import spherical
 torch.set_grad_enabled(False);g=torch.Generator(device='cuda').manual_seed(3939)
 before=replay.state_sha(net);rows=[];errors=[]
 # Replay all reliable records including no-contact and release. No parameter update.
 for ix in torch.arange(replay.n,device='cuda').split(64):
  a,q=replay.actions(replay.z[ix],ix)
  errors.append(dict(qpd=float((q-replay.qt[ix])[replay.mask[ix]].abs().max()),action=float((a-replay.at[ix])[replay.mask[ix]].abs().max())))
 assert max(x['qpd'] for x in errors)<1e-4,errors
 for _ in range(8):
  ix=torch.randint(replay.n,(128,),generator=g,device='cuda');pi=torch.randperm(len(batch['obs']),generator=g,device='cuda')[:256]
  with torch.enable_grad():
   d=net.distribution(batch['obs'][pi]);lr=d.log_prob(batch['action'][pi]).sum(-1)-batch['logp'][pi]
   ratio=lr.exp();adv=batch['advantages'][pi]
   pg=-torch.minimum(ratio*adv,ratio.clamp(.85,1.15)*adv).mean()
   gp=grad(pg,net);lr_loss=replay.loss(net,ix,'raw');gr=grad(lr_loss,net);la_loss=replay.loss(net,ix,'action');ga=grad(la_loss,net)
  assert torch.isfinite(ga).all() and ga.norm()>1e-9 and torch.isfinite(gr).all() and gp.norm()>0
  rows.append(dict(raw_loss=float(lr_loss),action_qpd_rad2=float(la_loss),ppo_grad=float(gp.norm()),raw_grad=float(gr.norm()),action_grad=float(ga.norm()),raw_ppo_cos=float(torch.nn.functional.cosine_similarity(gr,gp,dim=0)),action_ppo_cos=float(torch.nn.functional.cosine_similarity(ga,gp,dim=0)),old_logprob_error=float(lr.abs().max())))
 weight=plan['replay']['raw_weight']*np.median([r['raw_grad'] for r in rows])/np.median([r['action_grad'] for r in rows]);assert np.isfinite(weight) and weight>0
 for r in rows:r.update(weighted_raw_ppo_ratio=.002*r['raw_grad']/r['ppo_grad'],weighted_action_ppo_ratio=float(weight)*r['action_grad']/r['ppo_grad'])
 ix=torch.arange(5,device='cuda');bat=interp(replay.prev[ix],replay.z[ix]);intererr=max(float((bat[j,k]-spherical(replay.prev[ix[j]][None],replay.z[ix[j]][None],(k+1)/5)[0]).abs().max()) for j in range(5) for k in range(5));assert intererr<1e-4
 r=net.distribution(replay.x[ix]).mean;z=direct_command(r,replay.standing)
 _,q1=replay.actions(z,ix);_,q2=replay.actions(direct_command(2*r,replay.standing),ix)
 assert (q1-q2).abs().max()<1e-5
 # Full action -> z -> actor derivative verified above; ensure no frozen grads/mutation.
 assert all(not p.requires_grad and p.grad is None for p in replay.model.parameters())
 assert replay.state_sha(net)==before
 report=dict(passed=True,action_weight=float(weight),raw_weight=.002,initial_sha256=plan['initial']['sha256'],initial_state_sha=before,data_sha256=replay.data_sha,gradient_batches=rows,max_replay_qpd_rad=max(x['qpd'] for x in errors),max_replay_action=max(x['action'] for x in errors),interpolation_error=intererr,positive_raw_scale_mean_qpd_error=float((q1-q2).abs().max()),note='Positive raw scaling preserves mean direction, not the fixed-noise stochastic policy; frozen observation replay is not physical rollout',physical_calibration_decisions=len(batch['reward']))
 (out/'calibration.json').write_text(json.dumps(report,indent=2,allow_nan=False));print('CALIBRATION',json.dumps(report),flush=True)
