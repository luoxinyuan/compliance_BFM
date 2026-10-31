"""Finite saved-state CUDA diagnostics; never steps physics or updates a Student."""
import gzip,json,math
from pathlib import Path
import numpy as np,torch
from contact_projector import constrained,check_deadline
from m1_interface import endpoints,policy_latent
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/m2-unseen-body-20261031A'
def read(p):return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
def run(env,model,contract,out):
 ledger=read(R/'compute-ledger.json');done=[a for a in ledger['attempts'] if a['stage']=='comparison' and a.get('result_status')=='completed'];assert len(done)==1
 src=Path(done[0]['path'])/'outputs';dev=env.standing.device;scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev);qt=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
 ends=endpoints(env);decomposition=[];capacity=[];manifest=[]
 for p in sorted(src.glob('*.json.gz')):
  e=read(p);cs=e['control'];q=np.array([c['q_PD'] for c in cs]);dq=np.abs(np.diff(q,axis=0));tt=np.array([c['time'] for c in cs]);dq[tt[1:]<e['control_ready_s']-1e-8]=-1;i,j=np.unravel_index(dq.argmax(),dq.shape);a,b=cs[i],cs[i+1]
  ob={k:torch.tensor(v,device=dev,dtype=torch.float32) for k,v in b['bfm_obs'].items()};oa={k:torch.tensor(v,device=dev,dtype=torch.float32) for k,v in a['bfm_obs'].items()};za=torch.tensor(a['executed_z'],device=dev);zb=torch.tensor(b['executed_z'],device=dev)
  aa=model.act(oa,za,mean=True);bb=model.act(ob,zb,mean=True);ba=model.act(ob,za,mean=True);ab=model.act(oa,zb,mean=True)
  parity=max(float(abs(aa[0]-torch.tensor(a['base_action'],device=dev)).max()),float(abs(bb[0]-torch.tensor(b['base_action'],device=dev)).max()));assert parity<1e-6
  aq,bq,baq,abq=[qt(x)[0].cpu().numpy() for x in [aa,bb,ba,ab]]
  zterm=bq-baq;obs_term=baq-aq;zfirst=abq-aq;obslast=bq-abq;actual=np.array(b['q_PD'])-np.array(a['q_PD']);extra=actual-(bq-aq)
  decomposition.append(dict(label=e['label'],time=b['time'],joint=contract['dof_names'][j],record_parity_max=parity,actual_target_step_signed_deg=float(np.rad2deg(actual[j])),z_at_new_obs_deg=float(np.rad2deg(zterm[j])),state_at_old_z_deg=float(np.rad2deg(obs_term[j])),z_at_old_obs_deg=float(np.rad2deg(zfirst[j])),state_at_new_z_deg=float(np.rad2deg(obslast[j])),extra_teacher_target_deg=float(np.rad2deg(extra[j])),additive_error_rad=float(abs(zterm+obs_term+extra-actual).max())))
  if e['controller'] not in ['distill','alignment'] or not e['condition']['enabled']:continue
  for t in [13.0,14.4,16.8,17.6,18.4,21.4]:
   i=int(np.argmin(abs(tt-t)));block=cs[i:i+5];assert len(block)==5 and all(c['oracle_query_valid'] for c in block);check_deadline()
   obs={k:torch.tensor([c['bfm_obs'][k][0] for c in block],device=dev,dtype=torch.float32) for k in env.keys};norm={k:v.detach() for k,v in model._normalize(obs).items()};target=torch.tensor([c['oracle_q_PD'] for c in block],device=dev);base=torch.tensor([c['oracle_z'][0] for c in block],device=dev);gate=torch.tensor([c['local_gate'] for c in block],device=dev)
   actor=lambda z:model._actor(norm,z,model.cfg.actor_std).mean.float()
   def errors(z):
    err=qt(actor(z))-target;return dict(arm_rms_rad=float(err[:,22:].square().mean().sqrt()),nonarm_rms_rad=float(err[:,:22].square().mean().sqrt()))
   current=torch.tensor([c['executed_z'][0] for c in block],device=dev);entry=dict(label=e['label'],time=t,gate=gate.tolist(),student=errors(current),native_base=errors(base),optimizations={})
   for kind in ['oracle_native_base','student_native_base']:
    if kind=='oracle_native_base':fixed=base
    else:
     from m1_interface import base_latent
     rawsaved=torch.tensor([c['student_raw'] for c in block],device=dev);fixed=base_latent(ends,*rawsaved[:,:3].sigmoid().unbind(-1)).detach()
    with torch.enable_grad():
     raw=torch.zeros(5,256,device=dev,requires_grad=True);opt=torch.optim.Adam([raw],lr=.05)
     for step in range(160):
      check_deadline();opt.zero_grad();z=constrained(model,fixed,raw,gate)[0];err=qt(actor(z))-target;loss=err[:,22:].square().mean()+.1*err[:,:22].square().mean();loss.backward();opt.step()
     z=constrained(model,fixed,raw,gate)[0]
    entry['optimizations'][kind]=dict(**errors(z.detach()),norm_error=float(abs(z.norm(dim=-1)-16).max()),scope='same .3 localtangent, actualgate, independentactual50Hzlatentframes,160steps,neverdeployed')
   capacity.append(entry);print('M2_CAPACITY',e['controller'],t,entry['student'],entry['optimizations'],flush=True)
  manifest.append(str(p))
 result=dict(status='completed',student_optimizer_updates=0,physics_steps=0,diagnostic='Recorded q_PD jump functional decomposition (not physical counterfactual); bounded action capacity at12fixedblocks',decomposition=decomposition,capacity=capacity,sources=manifest,first_boundary='Finite optimization failures are not proof of global BFM impossibility; compare oracle vs learned nativebase and actual release gates')
 (out/'diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False));return result
