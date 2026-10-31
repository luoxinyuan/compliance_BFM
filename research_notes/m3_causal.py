"""M2 saved-state three-factor BFM functional decomposition; CUDA, no rollout/fit."""
import itertools,json,math
import numpy as np,torch
from m3_paired import read,SRC
from contact_projector import constrained,check_deadline
from m1_interface import endpoints,base_latent
def run(env,model,contract,out):
 dev=env.standing.device;ends=endpoints(env);scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev);qt=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
 results=[]
 for ctl in ['distill','alignment']:
  positive=read(SRC/f'midpoint__{ctl}.json.gz');negative=read(SRC/f'midpoint__{ctl}__free.json.gz');cp,cn=positive['control'],negative['control'];assert len(cp)==len(cn)
  for t in [12.4,13.0,14.4,16.8,18.4,21.4]:
   check_deadline();i=int(np.argmin(abs(np.array([c['time'] for c in cp])-t)));pair=[cn[i:i+5],cp[i:i+5]];assert all(abs(a['time']-b['time'])<1e-8 for a,b in zip(*pair))
   obs=[{k:torch.tensor([c['bfm_obs'][k][0] for c in rows],device=dev,dtype=torch.float32) for k in env.keys} for rows in pair];raw=[torch.tensor([c['student_raw'] for c in rows],device=dev) for rows in pair];gate=[torch.tensor([c['local_gate'] for c in rows],device=dev) for rows in pair];bases=[base_latent(ends,*r[:,:3].sigmoid().unbind(-1)) for r in raw];f={};parity=[]
   for bits in itertools.product([0,1],repeat=3):
    ob,native,local=bits;z=constrained(model,bases[native],raw[local][:,3:],gate[local])[0];action=model.act(obs[ob],z,mean=True);f[bits]=qt(action).cpu().numpy()
    if bits in [(0,0,0),(1,1,1)]:
     saved=np.array([c['q_PD'] for c in pair[ob]]);parity.append(float(abs(saved-f[bits]).max()))
   assert max(parity)<2e-6,parity
   factors=[]
   for j in range(3):
    val=np.zeros_like(f[(0,0,0)])
    for bits in itertools.product([0,1],repeat=3):
     if bits[j]:continue
     n=sum(bits);w=math.factorial(n)*math.factorial(2-n)/math.factorial(3);yes=list(bits);yes[j]=1;val+=w*(f[tuple(yes)]-f[bits])
    factors.append(val)
   delta=f[(1,1,1)]-f[(0,0,0)];assert abs(sum(factors)-delta).max()<2e-6
   metrics=lambda q:dict(arm_rms_deg=float(np.rad2deg(np.sqrt(np.mean(q[:,22:]**2)))),nonarm_rms_deg=float(np.rad2deg(np.sqrt(np.mean(q[:,:22]**2)))),mean_signed_qPD_deg=np.rad2deg(q.mean(0)).tolist())
   r=dict(controller=ctl,time=t,positive_native=[c['native_coords'] for c in pair[1]],negative_native=[c['native_coords'] for c in pair[0]],gate_positive=gate[1].tolist(),parity_max=max(parity),total=metrics(delta),factors={k:metrics(v) for k,v in zip(['BFM_observation_feedback','native_coordinate_command','gated_local_residual'],factors)});results.append(r);print('M3_FACTOR',ctl,t,r['total']['arm_rms_deg'],{k:v['arm_rms_deg'] for k,v in r['factors'].items()},flush=True)
 result=dict(status='completed',student_optimizer_updates=0,physics_steps=0,blocks=results,scope='Exact symmetric 3-factor Shapley on saved actual 50Hz observations/commands: 8 combinations each. Functional q_PD decomposition, not independent physical intervention. BFM feedback includes proprio/action history, native and residual both may depend on force AND proprio; no unique force-channel causality claim. RMS components do not sum, signed vectors do.')
 (out/'causal-decomposition.json').write_text(json.dumps(result,indent=2,allow_nan=False));return result
