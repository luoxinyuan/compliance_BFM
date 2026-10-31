"""Finite diagnostic on saved real observations. No learned policy or physics rollout.
Oracle native-base results are feasibility evidence only, not deployable Student inputs.
"""
import json,hashlib
from pathlib import Path
import numpy as np,torch
from contact_projector import constrained,ContactGate,check_deadline
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUN=P/'bridge-v2-owned/runs/m1-state-response-20261031A'
def read(p):return json.loads(Path(p).read_text())
def save(out,n,x):(out/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def slerp(a,b,u):
 u=u.reshape(-1,1);theta=torch.acos(((a*b).sum(-1,keepdim=True)/256).clamp(-1+1e-7,1-1e-7));return 16*torch.nn.functional.normalize(torch.sin((1-u)*theta)/torch.sin(theta)*a+torch.sin(u*theta)/torch.sin(theta)*b,dim=-1)
def run(env,model,contract,out):
 spec=read(Path(__file__).with_name('m1_representation_protocol.json'));save(out,'protocol.json',spec)
 src=[x for x in read(RUN/'compute-ledger.json')['attempts'] if x['stage']=='teachers' and x.get('result_status')=='completed'];assert len(src)==1;source=Path(src[0]['path'])/'outputs'
 dev=env.standing.device;scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev);qof=lambda a:offset+a.clamp(-1,1)*scale
 records=[];blocks=[]
 for config in ['standing','raised']:
  ep=read(source/(config+'__teacher_goal.json'));cs=ep['control'];g=ContactGate();gates=[]
  for i,c in enumerate(cs):
   if i%5==0:g.update(c['time'],c['measured_force_xyz'])
   gates.append(g.value(c['time']))
  for when in spec['representative_times']:
   ix=min(range(0,len(cs)-4,5),key=lambda i:abs(cs[i]['time']-when));assert abs(cs[ix]['time']-when)<1e-8
   records.extend([dict(c=c,gate=gates[j]) for j,c in enumerate(cs[ix:ix+5],start=ix)])
   blocks.append(dict(configuration=config,time=when,source=str(source/(config+'__teacher_goal.json')),phase='release' if when>=ep['condition']['release'] else 'hold' if when>=ep['condition']['onset']+4 else 'move'))
 n=len(blocks);obs={k:torch.tensor([r['c']['bfm_obs'][k][0] for r in records],device=dev) for k in env.keys};norm={k:v.detach() for k,v in model._normalize(obs).items()};target=torch.tensor([r['c']['q_PD'] for r in records],device=dev);base=torch.tensor([r['c']['executed_z'][0] for r in records],device=dev);base_q=torch.tensor([r['c']['base_q_PD'] for r in records],device=dev);gate=torch.tensor([r['gate'] for r in records],device=dev)
 public=model.act(obs,base,mean=True);actor=lambda z:model._actor(norm,z,model.cfg.actor_std).mean.float();parity=float(abs(public-actor(base)).max());record_parity=float(abs(qof(public)-base_q).max());assert parity<1e-6 and record_parity<1e-5,(parity,record_parity)
 outputs={};saved={}
 for interface in spec['interfaces']:
  check_deadline()
  with torch.enable_grad():
   raw=torch.zeros(n,256,device=dev,requires_grad=True);opt=torch.optim.Adam([raw],lr=.05);trace=[]
   center=env.standing.expand_as(base) if interface=='old_standing_gate' else base
   use_gate=gate if interface!='native_base_oracle_ungated' else torch.ones_like(gate)
   def evaluate():
    z,_=constrained(model,center,raw.repeat_interleave(5,0),use_gate);a=actor(z);q=qof(a);e=q-target;loss=e[:,22:].square().mean()+.1*e[:,:22].square().mean();return loss,e,z,a
   for step in range(spec['steps_per_interface']):
    check_deadline();opt.zero_grad();loss,e,z,a=evaluate();loss.backward();opt.step()
    if step%40==0:trace.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(raw.grad.norm())))
   loss,e,z,a=evaluate()
  per=[];err=e.detach().reshape(n,5,29);dq=(target-base_q).reshape(n,5,29);normerr=float(abs(z.detach().norm(dim=-1)-16).max());angle=torch.rad2deg(torch.acos(((z.detach()*base).sum(-1)/256).clamp(-1,1))).reshape(n,5)
  for i,b in enumerate(blocks):
   arm=float(err[i,:,22:].square().mean().sqrt());nonarm=float(err[i,:,:22].square().mean().sqrt());reference=float(dq[i,:,22:].square().mean().sqrt());per.append(dict(**b,arm_rms_rad=arm,nonarm_rms_rad=nonarm,teacher_correction_arm_rms_rad=reference,ratio=arm/max(reference,1e-12),gate=gate.reshape(n,5)[i].tolist(),angle_from_native_max_deg=float(angle[i].max()),passes_local_action_criterion=arm<=max(.5*reference,.001) and nonarm<=.005))
  outputs[interface]=dict(loss=float(loss.detach()),norm_error=normerr,blocks=per,trace=trace);saved[interface]=dict(raw=raw.detach().cpu(),z=z.detach().cpu(),action=a.detach().cpu(),error=e.detach().cpu())
  print('M1_REPRESENTABILITY',interface,[(b['configuration'],b['time'],round(b['arm_rms_rad'],5),round(b['nonarm_rms_rad'],5),b['passes_local_action_criterion']) for b in per],flush=True)
 torch.save(saved,out/'temporary-latent-diagnostics.pt');report=dict(public_actor_parity=parity,recorded_base_q_PD_parity=record_parity,results=outputs,student_updates=0,meaning='Nativebase is teacher-oracle supplied on each real50Hz observation. This does not qualify a deployable interface, globalBFM impossibility, or a learnedpolicy. Same10Hzraw across5real50Hzobservations; gate-vs-ungated separates release protocol from action capacity.');save(out,'representability.json',report)
 return dict(status='completed',student_optimizer_updates=0,representation_report=str(out/'representability.json'))
