"""Read-only same-state policy/action analysis and archived PPO gradients."""
import argparse,os,sys,json,gzip,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
from retention_eval import MODELS,sha,write,P,S,OLD

def read(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else Path(p).open() as f:return json.load(f)
def study(e,m,contract,out):
 import torch,numpy as np
 from clean_policy import Expert,direct_command
 from locomotion_rollout import spherical
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 from native_env import install
 install();torch.set_num_threads(1);torch.set_grad_enabled(False);write(out/'speed.json',speed(e));write(out/'backend.json',backend(e,m,P/'experiments/native-rolling-h100-20261107/compatibility-prefix'))
 nets={}
 for role in ['s5401b4','s5401b7']:
  p,d=MODELS[role];assert sha(p)==d;n=Expert(1639,'direct').cuda();n.load_state_dict(torch.load(p,map_location='cuda',weights_only=True)['state_dict']);n.requires_grad_(False);nets[role]=n
 rows=[]
 for role,folder in [('s5401b4','A0-eval-01'),('s5401b7','A0-eval-02')]:
  ep=read(S/'runs'/folder/(role+'_normal72.json.gz'));ds=read(S/'runs'/folder/(role+'-restricted-decisions.json.gz'))['normal72'];cs=ep['control'];assert len(cs)==5*len(ds)
  for j,d in enumerate(ds):
   x=torch.tensor(d['restricted'],dtype=torch.float32,device='cuda')[None];last=torch.tensor(d['last_z'],dtype=torch.float32,device='cuda')[None];raw={k:n.distribution(x).mean for k,n in nets.items()};z={k:direct_command(v,e.standing) for k,v in raw.items()};qps={k:[] for k in nets};replay=0.
   for sub,c in enumerate(cs[j*5:j*5+5]):
    obs={k:torch.tensor(v,device='cuda',dtype=torch.float32) for k,v in c['bfm_obs'].items()}
    for k in nets:
     zz=spherical(last,z[k],(sub+1)/5);a=m.act(obs,zz,mean=True);q=torch.tensor(contract['q_PD_offset'],device='cuda')+(a*5).clamp(-contract['clip'],contract['clip'])*torch.tensor(contract['q_PD_scale'],device='cuda')/5;qps[k].append(q[0].cpu().numpy())
     if k==role:replay=max(replay,float(np.max(abs(q[0].cpu().numpy()-np.array(c['q_PD'])))))
   qa,qb=np.array(qps['s5401b4']),np.array(qps['s5401b7']);rows.append(dict(source=role,time=d['time'],raw_delta_rms=float((raw['s5401b4']-raw['s5401b7']).square().mean().sqrt()),z_angle_deg=float(torch.rad2deg(torch.acos(((z['s5401b4']*z['s5401b7']).sum()/256).clamp(-1,1)))),five_substeps_qpd_rms_deg=float(np.degrees(np.sqrt(np.mean((qa-qb)**2)))),five_substeps_qpd_max_deg=float(np.degrees(abs(qa-qb).max())),per_substep_qpd_rms_deg=np.degrees(np.sqrt(np.mean((qa-qb)**2,axis=1))).tolist(),same_state_replay_max_rad=replay,scope='same actual 5 BFMap observations and previous z; not counterfactual5step physical rollout'))
 write(out/'same-state.json',rows)
 # Actual on-policy data, gradients at the exact pre-update actor, no optimizer mutation.
 from humanoidverse.force_compliance.parallel import merge_rollouts
 gradients=[]
 for iteration in [4,7]:
  folder=OLD/'runs/ppo-clean-5401';n=Expert(1639,'direct').cuda();n.load_state_dict(torch.load(folder/f'candidate-{iteration-1}.pt',map_location='cuda',weights_only=True)['state_dict']);n.requires_grad_(False);n.actor.requires_grad_(True)
  with np.load(folder/f'rollout-{iteration}.npz') as a:b0={k:a[k] for k in a.files}
  batch={k:v.cuda() for k,v in merge_rollouts([dict(batch=b0)],.95).items()};stats=read(folder/f'rollout-{iteration}-stats.json');families={'long':lambda s:s['demand_family']=='long','near':lambda s:s['demand_family'] in ['near','zero'],'stop':lambda s:s['demand_family']=='stop'};gs={};info={}
  for family,select in families.items():
   ids=[i for i,s in enumerate(stats) if select(s)][:64]
   if not ids:continue
   ix=torch.tensor(ids,device='cuda');n.zero_grad(set_to_none=True)
   with torch.enable_grad():
    lp=n.distribution(batch['obs'][ix]).log_prob(batch['action'][ix]).sum(-1);loss=-(lp*batch['advantages'][ix].detach()).mean();g=torch.autograd.grad(loss,tuple(n.actor.parameters()));v=torch.cat([x.flatten() for x in g]).detach()
   gs[family]=v;info[family]=dict(samples=len(ids),indices=ids,norm=float(v.norm()),mean_advantage=float(batch['advantages'][ix].mean()),raw_logprob_reproduction_max=float(abs(lp.detach()-batch['logp'][ix]).max()))
  cos={a+'__'+b:float(torch.dot(gs[a],gs[b])/(gs[a].norm()*gs[b].norm()).clamp_min(1e-12)) for a in gs for b in gs if a<b}
  gradients.append(dict(iteration=iteration,families=info,cosines=cos,meaning='diagnostic instantaneous score-function gradients, no gradient surgery or causal proof'))
 write(out/'gradient-conflict.json',gradients)
 return dict(status='completed',new_physics=0,new_RL=0,queries=0,same_state_rows=len(rows),maximum_replay_qpd_rad=max(r['same_state_replay_max_rad'] for r in rows))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);write(a.output/'launch.json',dict(runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__)))
 os.environ['CLEAN_KIND']='direct';os.environ['P4000_CANONICAL_ROW_ACTOR']='1';os.environ['BFM_RECOVERY_OUTPUT']=str(a.output);os.chdir(a.output)
 import m10_stage,m10_physics
 m10_physics.OUT=a.output;m10_physics.MODEL=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view';m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
