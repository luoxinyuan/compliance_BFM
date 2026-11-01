"""Bounded M9 continuation, own previous-command memory with paired no-memory control."""
import os,json
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
from m9b_stage import RUN,OLD,completed
from m9_learning import read,sha,old_context,load_net,ResidualPolicy,sensor_features,policy
from m9_history import ContactHistory
from contact_projector import ContactGate,check_deadline
from m8_learning import policy_latent as old_latent

class MemoryPolicy(ResidualPolicy):
 pass

def load_memory(p,dev):
 x=torch.load(p,map_location=dev,weights_only=True);net=MemoryPolicy(torch.zeros(2,len(x['state_dict']['center']),device=dev)).to(dev);net.load_state_dict(x['state_dict']);return net.eval().requires_grad_(False)

def memory_policy(model,old,ends,net,h,h8,gate,previous_z):
 base=old_latent(model,ends,old(h8),gate);z=model.project_z(base+net(torch.cat([h,previous_z/16],-1)));active=h[:,465:].reshape(-1,320,14)[:,:,:3].abs().amax((1,2))>1e-8
 return torch.where(active[:,None],z,ends[0].expand_as(z))

def build(env,model,out,rnd,old,ends):
 dev=env.standing.device
 if rnd==0:
  prior=torch.load(OLD/'train2-attempt-01/outputs/dataset.pt',map_location=dev,weights_only=True)
  groups=prior['groups'];manifest=prior['manifest']
  for g,m in zip(groups,manifest):
   e=read(m['path']);z=np.array([c['executed_z'][0] for c in e['control']]);prev=np.concatenate([env.standing.cpu().numpy(),z[:-1]])
   assert len(g['h'])==len(prev);g['previous_z']=torch.tensor(prev,device=dev,dtype=torch.float32);m['origin']='m9_archived'
 else:
  prior=torch.load(completed('train0')/'dataset.pt',map_location=dev,weights_only=True);groups=prior['groups'];manifest=prior['manifest']
 sources=sorted(completed('collect'+str(rnd)).glob('*.json.gz'))
 for source in sources:
  e=read(source)
  if not e.get('control'):continue
  history=ContactHistory();gate=ContactGate();rows=[];spec=e['condition'];site=spec.get('body','right_wrist_yaw_link');point=spec.get('local_point',[0,0,0]);prev=env.standing[0].cpu().numpy()
  for tick,c in enumerate(e['control']):
   check_deadline();R=Rotation.from_quat(np.array(c['pre_qpos'])[[4,5,6,3]]).as_matrix();f=c['measured_force_xyz'];history.append(f,site,point,R)
   if tick%5==0:gate.update(c['time'],f)
   x,l=sensor_features(history,R);rows.append(dict(x=x,l=l,obs={k:c['bfm_obs'][k][0] for k in env.keys},z=c['oracle_z'][0],q=c['oracle_q_PD'],gate=gate.value(c['time']),prev=prev));prev=np.array(c['executed_z'][0])
  ob={k:torch.tensor(np.array([r['obs'][k] for r in rows]),device=dev,dtype=torch.float32) for k in env.keys};norm=model._normalize(ob);pro=torch.cat([norm[k].reshape(len(rows),-1).float() for k in env.keys],-1)
  T=lambda x:torch.tensor(np.array(x),device=dev,dtype=torch.float32)
  h=torch.cat([pro,T([r['x'] for r in rows])],-1);h8=torch.cat([pro,T([r['l'] for r in rows])],-1)
  g=dict(h=h,h8=h8,z=T([r['z'] for r in rows]),q=T([r['q'] for r in rows]),gate=T([r['gate'] for r in rows]),previous_z=T([r['prev'] for r in rows]),obs=ob);g['base']=old_latent(model,ends,old(h8),g['gate']);groups.append(g);manifest.append(dict(path=str(source),sha256=sha(source),samples=len(rows),family=spec.get('family','walking'),origin='continuation',partial=e['boundary'] is not None,condition=spec))
  print('M9B_DATA',source.name,len(rows),flush=True)
 torch.save(dict(groups=[{k:{kk:vv.cpu() for kk,vv in v.items()} if k=='obs' else v.cpu() for k,v in g.items()} for g in groups],manifest=manifest),out/'dataset.pt')
 return groups,manifest

def run(env,model,contract,out):
 rnd=int(os.environ['M9_STAGE'][-1]);plan=read(Path(__file__).with_name('m9b_protocol.json'))['training'];dev=env.standing.device;old,ends=old_context(env);groups,manifest=build(env,model,out,rnd,old,ends)
 torch.manual_seed(plan['seed']);legacy=load_net(OLD/'train2-attempt-01/outputs/projected-z-final.pt' if not rnd else completed('train0')/'no-memory-final.pt',dev)
 if not rnd:
  net=MemoryPolicy(torch.zeros(2,5201,device=dev)).to(dev);sd=legacy.state_dict();ns=net.state_dict()
  for k,v in sd.items():
   if k in ['center','scale']:ns[k][:4945]=v;ns[k][4945:]=0 if k=='center' else 1
   elif k=='net.0.weight':ns[k][:,:4945]=v;ns[k][:,4945:]=0
   else:ns[k]=v
  net.load_state_dict(ns)
  test=groups[0];assert torch.allclose(net(torch.cat([test['h'][:8],test['previous_z'][:8]/16],-1)),legacy(test['h'][:8]),atol=2e-6,rtol=2e-6)
 else:net=load_memory(completed('train0')/'memory-final.pt',dev)
 nets={'memory':net,'no-memory':legacy};opts={}
 for name,n in nets.items():n.train().requires_grad_(True);opts[name]=torch.optim.AdamW(n.parameters(),lr=plan['lr'],weight_decay=.0001)
 scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev);qof=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
 identity=[]
 for g in groups:
  mx=0.
  for start in range(0,len(g['h']),128):
   ix=slice(start,start+128);q=qof(model.act({k:v[ix] for k,v in g['obs'].items()},g['z'][ix],mean=True));mx=max(mx,float(abs(q-g['q'][ix]).max()))
  assert mx<1e-4;identity.append(mx)
 oldpool=[i for i,m in enumerate(manifest) if m['family']=='old'];priorpool=[i for i,m in enumerate(manifest) if m['family']!='old' and m['origin']=='m9_archived'];newpool=[i for i,m in enumerate(manifest) if m['family']!='old' and m['origin']=='continuation'];assert all([oldpool,priorpool,newpool]);pools=[oldpool,newpool,oldpool,priorpool];rng=torch.Generator(device=dev).manual_seed(plan['seed']+rnd);logs=[]
 for step in range(plan['steps'][rnd]):
  check_deadline();pool=pools[step%4];g=groups[pool[int(torch.randint(len(pool),(1,),generator=rng,device=dev))]];ix=torch.randint(len(g['h']),(plan['batch'],),generator=rng,device=dev);row=dict(step=step)
  for name,n in nets.items():
   h=torch.cat([g['h'][ix],g['previous_z'][ix]/16],-1) if name=='memory' else g['h'][ix]
   with torch.enable_grad():
    opts[name].zero_grad();z=model.project_z(g['base'][ix]+n(h));loss=(z-g['z'][ix]).square().mean();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(n.parameters(),1));opts[name].step()
   row[name]=dict(loss=float(loss),gradient_norm=gn)
  if step%250==0:logs.append(row);print('M9B_TRAIN',rnd,row,flush=True)
 results={}
 for name,n in nets.items():
  metrics=[]
  for g in groups:
   ee=[]
   for start in range(0,len(g['h']),128):
    ix=slice(start,start+128);h=torch.cat([g['h'][ix],g['previous_z'][ix]/16],-1) if name=='memory' else g['h'][ix];z=model.project_z(g['base'][ix]+n(h));ee.append(qof(model.act({k:v[ix] for k,v in g['obs'].items()},z,mean=True))-g['q'][ix])
   metrics.append(float(torch.rad2deg(torch.cat(ee).square().mean().sqrt())))
  ck=out/(name+'-final.pt');torch.save(dict(state_dict={k:v.detach().cpu() for k,v in n.state_dict().items()},round=rnd,updates=sum(plan['steps'][:rnd+1]),fixed_final=True,own_previous_command=name=='memory'),ck);results[name]=dict(checkpoint=str(ck),sha256=sha(ck),qpd_fit_deg=metrics)
 result=dict(status='completed',student_optimizer_updates=2*plan['steps'][rnd],results=results,manifest=manifest,logs=logs,native_action_identity_max_rad=max(identity));(out/'training.json').write_text(json.dumps(result,indent=2));return result
