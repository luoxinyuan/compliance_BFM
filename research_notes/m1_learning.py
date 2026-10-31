"""One paired multi-state LPP update budget and one expert-query aggregation.
Privileged oracle base/actions are labels only. Policy sees original1425D deployable history.
"""
import copy,json,hashlib
from pathlib import Path
import numpy as np,torch
from alpha_student import student_input
from alpha_memory import ForceMemory
from contact_projector import ContactGate,constrained,check_deadline
from m1_interface import endpoints,policy_latent,SharedProjector
from frozen_bfm_placement import tensor_digest
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUN=P/'bridge-v2-owned/runs/m1-state-response-20261031A'
read=lambda p:json.loads(Path(p).read_text())
def save(out,n,x):(out/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def completed(stage):
 a=[a for a in read(RUN/'compute-ledger.json')['attempts'] if a['stage']==stage and a.get('result_status')=='completed'];assert len(a)==1,(stage,len(a));return Path(a[0]['path'])/'outputs'
def load_net(path,dev):
 d=torch.load(path,map_location=dev,weights_only=True);s=d['state_dict'];net=SharedProjector(torch.zeros(2,len(s['center']),device=dev)).to(dev);net.load_state_dict(s);return net.eval().requires_grad_(False)
def build_dataset(env,model,contract,out,aggregate=False):
 src=completed('teachers');qa=completed('qualification');sources=[(src/(c+'__teacher_goal.json'),'expert') for c in ['standing','raised']]+[(qa/(c+'__pose_hold__free.json'),'negative') for c in ['standing','raised']]
 if aggregate:
  sources += [(f,'visited') for f in sorted(completed('visits').glob('*__*.json')) if f.name.endswith('__alignment.json') or f.name.endswith('__distill.json')]
 rows=[];manifest=[];dev=env.standing.device
 for group,(path,kind) in enumerate(sources):
  ep=read(path);memory=ForceMemory(320);gate=ContactGate();cs=ep['control'];hist=[];gs=[]
  for i,c in enumerate(cs):
   hist.append(memory.append(c['measured_force_xyz']).copy())
   if i%5==0:gate.update(c['time'],c['measured_force_xyz'])
   gs.append(gate.value(c['time']))
  ix=[i for i,c in enumerate(cs) if c['time']>=ep['condition']['onset']-1e-8 and (kind!='negative' or c['time']<ep['condition']['release']-1e-8)]
  obs={k:torch.tensor([cs[i]['bfm_obs'][k][0] for i in ix],device=dev) for k in env.keys};h=student_input(model,obs,[hist[i] for i in ix],env.keys).detach();assert h.shape[1]==1425
  for j,i in enumerate(ix):
   c=cs[i];oracle=kind=='visited';base=c['oracle_z'] if oracle else c['executed_z'];action=c['oracle_action'] if oracle else c['action'];q=c['oracle_q_PD'] if oracle else c['q_PD']
   rows.append(dict(h=h[j],obs={k:obs[k][j] for k in env.keys},base=base[0],action=action,q=q,base_q=c['oracle_base_q_PD'] if oracle else c.get('base_q_PD',c['q_PD']),gate=gs[i],group=group,configuration=ep['configuration'],kind=kind,time=c['time']))
  manifest.append(dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),kind=kind,group=group,samples=len(ix),configuration=ep['configuration']))
 data=dict(h=torch.stack([r['h'] for r in rows]),obs={k:torch.stack([r['obs'][k] for r in rows]) for k in env.keys},metadata=[{k:r[k] for k in ['configuration','kind','time','group']} for r in rows])
 for k in ['base','action','q','base_q','gate']:data[k]=torch.tensor([r[k] for r in rows],device=dev,dtype=torch.float32)
 data['group']=torch.tensor([r['group'] for r in rows],device=dev);save(out,'dataset-manifest.json',dict(sources=manifest,samples=len(rows),student_inputs='OriginalA2:normalized465Dproprio/action history +320realforceXYZ/5N. No time,anchor,configurationtag,targetpose,oracleprogress. Negative labels stop before artificialclockreturn; no impossible no-contact timedexit label.'))
 torch.save({k:({kk:vv.cpu() for kk,vv in v.items()} if k=='obs' else v.cpu() if torch.is_tensor(v) else v) for k,v in data.items()},out/'dataset.pt');return data

def train(env,model,contract,out,aggregate=False):
 completed('interface');rp=read(completed('representation')/'representability.json');assert all(b['passes_local_action_criterion'] for b in rp['results']['native_base_oracle_gate']['blocks'])
 dev=env.standing.device;d=build_dataset(env,model,contract,out,aggregate);ends=endpoints(env);norm={k:v.detach() for k,v in model._normalize(d['obs']).items()};scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev);qof=lambda a:offset+a.clamp(-1,1)*scale
 actor=lambda z,ix:model._actor({k:v[ix] for k,v in norm.items()},z,model.cfg.actor_std).mean.float()
 # Same finite local action projection produces global-z labels for BOTH methods' shared dataset.
 target_z=[];errors=[]
 for start in range(0,len(d['h']),32):
  check_deadline();ix=torch.arange(start,min(start+32,len(d['h'])),device=dev)
  with torch.enable_grad():
   raw=torch.zeros(len(ix),256,device=dev,requires_grad=True);opt=torch.optim.Adam([raw],lr=.05)
   for _ in range(160):
    check_deadline();opt.zero_grad();z=constrained(model,d['base'][ix],raw,d['gate'][ix])[0];e=qof(actor(z,ix))-d['q'][ix];loss=e[:,22:].square().mean()+.1*e[:,:22].square().mean();loss.backward();opt.step()
   z=constrained(model,d['base'][ix],raw,d['gate'][ix])[0];e=qof(actor(z,ix))-d['q'][ix]
  target_z.append(z.detach());errors.append(e.detach())
  if start%256==0:print('M1_LABEL_PROJECTION',start,len(d['h']),float(loss.detach()),flush=True)
 targets=torch.cat(target_z);err=torch.cat(errors);torch.save(targets.cpu(),out/'projected-global-z.pt')
 represent=[]
 for group in d['group'].unique().tolist():
  mask=d['group']==group;arm=float(err[mask,22:].square().mean().sqrt());nonarm=float(err[mask,:22].square().mean().sqrt());ref=float((d['q'][mask,22:]-d['base_q'][mask,22:]).square().mean().sqrt());represent.append(dict(group=group,arm_rms_rad=arm,nonarm_rms_rad=nonarm,teacher_delta_arm_rms_rad=ref,passed=arm<=max(.5*ref,.001) and nonarm<=.005))
 save(out,'label-projection.json',dict(groups=represent,steps=160,init='zero local raw',oracle_base_not_student_input=True))
 if not all(x['passed'] for x in represent):return dict(status='capability-stopped',student_optimizer_updates=0,reason='Visited-state local action expression failed fixedM0criterion',representation=represent)
 torch.manual_seed(0)
 if aggregate:nets={k:load_net(completed('fit')/(k+'-final.pt'),dev) for k in ['alignment','distill']}
 else:
  initial=SharedProjector(d['h']).to(dev);nets={k:copy.deepcopy(initial) for k in ['alignment','distill']};assert tensor_digest(nets['alignment'])==tensor_digest(nets['distill'])
 for net in nets.values():net.train().requires_grad_(True)
 initials={k:tensor_digest(n) for k,n in nets.items()};opts={k:torch.optim.Adam(n.parameters(),lr=.0003) for k,n in nets.items()};steps=500 if aggregate else 1000;groups=[(d['group']==g).nonzero().flatten() for g in d['group'].unique()];rng=torch.Generator(device=dev).manual_seed(0);logs=[]
 def metrics(net):
  ee=[];zz=[]
  with torch.no_grad():
   for start in range(0,len(d['h']),64):
    ix=torch.arange(start,min(start+64,len(d['h'])),device=dev);z=policy_latent(model,ends,net(d['h'][ix]),d['gate'][ix]);ee.append(qof(actor(z,ix))-d['q'][ix]);zz.append(z-targets[ix])
  e=torch.cat(ee);dz=torch.cat(zz);return dict(arm_rms_rad=float(e[:,22:].square().mean().sqrt()),nonarm_rms_rad=float(e[:,:22].square().mean().sqrt()),latent_rms=float(dz.square().mean().sqrt()),per_group=[dict(group=i,arm_rms_rad=float(e[g,22:].square().mean().sqrt()),nonarm_rms_rad=float(e[g,:22].square().mean().sqrt())) for i,g in enumerate(groups)])
 before={k:metrics(n) for k,n in nets.items()}
 for step in range(steps):
  check_deadline();chosen=torch.randint(len(groups),(16,),generator=rng,device=dev);ix=torch.stack([groups[int(g)][torch.randint(len(groups[int(g)]),(1,),generator=rng,device=dev)[0]] for g in chosen])
  row=dict(step=step)
  for name,net in nets.items():
   with torch.enable_grad():
    opts[name].zero_grad();raw=net(d['h'][ix]);z=policy_latent(model,ends,raw,d['gate'][ix])
    if name=='alignment':e=qof(actor(z,ix))-d['q'][ix];loss=e[:,22:].square().mean()+.1*e[:,:22].square().mean()
    else:loss=(z-targets[ix]).square().mean()
    assert torch.isfinite(loss);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(net.parameters(),10.));opts[name].step();row[name]=dict(loss=float(loss.detach()),gradient_norm_before_clip=gn)
  if step%100==0:logs.append(row);print('M1_TRAIN',row,flush=True)
 after={k:metrics(n) for k,n in nets.items()}
 for name,net in nets.items():
  torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},updates_this_stage=steps,total_updates=1500 if aggregate else 1000,method=name,student_input='1425Ddeployablehistory',fixed_final=True),out/(name+'-final.pt'))
 save(out,'training.json',dict(updates_per_method=steps,same_initialization=initials['alignment']==initials['distill'],initial_digests=initials,batch=16,seed=0,lr=.0003,before=before,after=after,logs=logs,checkpoints={k:hashlib.sha256((out/(k+'-final.pt')).read_bytes()).hexdigest() for k in nets},no_checkpoint_selection=True))
 return dict(status='completed',student_optimizer_updates=steps*2,models=['alignment','distill'],training_report=str(out/'training.json'))
