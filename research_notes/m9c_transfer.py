"""Deployable PZ transfer from qualified queryable feedback, preserving old M9B data."""
import os,json
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
from m9c_stage import RUN,completed
from m9b_stage import RUN as PREVIOUS
from m9_learning import old_context,load_net,read,sha,sensor_features
from m8_learning import policy_latent
from m9_history import ContactHistory
from contact_projector import ContactGate,check_deadline

def run(env,model,contract,out,refs,stage):
 plan=read(Path(__file__).with_name('m9c_protocol.json'))['transfer'];rnd=0 if stage=='train' else 1;dev=env.standing.device;old,ends=old_context(env)
 prior=torch.load(PREVIOUS/'train1-attempt-01/outputs/dataset.pt' if rnd==0 else completed('train')/'dataset.pt',map_location=dev,weights_only=True);groups=prior['groups'];manifest=prior['manifest'];sources=plan['sources'] if rnd==0 else [str(p) for p in completed('collect').glob('*__transfer.json.gz')]
 for source in sources:
  e=read(source);history=ContactHistory();gate=ContactGate();rows=[]
  for tick,c in enumerate(e['control']):
   R=Rotation.from_quat(np.array(c['pre_qpos'])[[4,5,6,3]]).as_matrix();f=c['measured_force_xyz'];history.append(f,'right_wrist_yaw_link',[0,0,0],R)
   if tick%5==0:gate.update(c['time'],f)
   x,l=sensor_features(history,R);rows.append(dict(x=x,l=l,gate=gate.value(c['time']),obs={k:c['bfm_obs'][k][0] for k in env.keys},z=c['oracle_z'][0] if rnd else c['executed_z'][0],q=c['oracle_q_PD'] if rnd else c['q_PD']))
  T=lambda x:torch.tensor(np.array(x),device=dev,dtype=torch.float32);obs={k:T([r['obs'][k] for r in rows]) for k in env.keys};norm=model._normalize(obs);pro=torch.cat([norm[k].reshape(len(rows),-1) for k in env.keys],-1);h=torch.cat([pro,T([r['x'] for r in rows])],-1);h8=torch.cat([pro,T([r['l'] for r in rows])],-1);g=dict(h=h,h8=h8,gate=T([r['gate'] for r in rows]),z=T([r['z'] for r in rows]),q=T([r['q'] for r in rows]),obs=obs);g['base']=policy_latent(model,ends,old(h8),g['gate']);groups.append(g);manifest.append(dict(path=source,sha256=sha(source),samples=len(rows),family='walking',origin='m9c',partial=e['boundary'] is not None))
 torch.save(dict(groups=[{k:({kk:vv.cpu() for kk,vv in v.items()} if k=='obs' else v.cpu()) for k,v in g.items()} for g in groups],manifest=manifest),out/'dataset.pt')
 net=load_net(PREVIOUS/'train1-attempt-01/outputs/no-memory-final.pt' if rnd==0 else completed('train')/'student-final.pt',dev);net.train().requires_grad_(True);torch.manual_seed(23+rnd);opt=torch.optim.AdamW(net.parameters(),lr=3e-5,weight_decay=.0001);pools=[[i for i,m in enumerate(manifest) if m.get('origin')!='m9c'],[i for i,m in enumerate(manifest) if m.get('origin')=='m9c']];logs=[];scale=torch.tensor(contract['q_PD_scale'],device=dev);off=torch.tensor(contract['q_PD_offset'],device=dev);qof=lambda a:off+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
 identity=[]
 for i in pools[1]:
  g=groups[i];errs=[]
  for start in range(0,len(g['h']),128):
   ix=slice(start,start+128);errs.append(float((qof(model.act({k:v[ix] for k,v in g['obs'].items()},g['z'][ix],mean=True))-g['q'][ix]).abs().max()))
  assert max(errs)<1e-4;identity.append(max(errs))
 for step in range(plan['updates_per_round']):
  check_deadline();pool=pools[step%2];g=groups[pool[int(torch.randint(len(pool),(1,),device=dev))]];ix=torch.randint(len(g['h']),(128,),device=dev)
  with torch.enable_grad():
   z=model.project_z(g['base'][ix]+net(g['h'][ix]));loss=(z-g['z'][ix]).square().mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
  if step%250==0:logs.append(dict(step=step,loss=float(loss)));print('TRANSFER',rnd,step,float(loss),flush=True)
 ck=out/'student-final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},fixed_final=True,updates=(rnd+1)*plan['updates_per_round'],round=rnd),ck);result=dict(status='completed',student_optimizer_updates=plan['updates_per_round'],checkpoint=str(ck),sha256=sha(ck),same_state_action_identity_max_rad=max(identity),manifest=manifest,logs=logs);(out/'training.json').write_text(json.dumps(result,indent=2));return result
