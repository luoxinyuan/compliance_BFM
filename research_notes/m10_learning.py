"""Task-balanced transfer from source-corrected native experts, with visited-state labels."""
import json,os
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
from m10_stage import RUN,completed,P
from m9_learning import old_context,load_net,read,sha,sensor_features
from m8_learning import policy_latent
from m9_history import ContactHistory
from contact_projector import ContactGate,check_deadline
PRIOR=P/'bridge-v2-owned/runs/m9-feedback-continuation-20261101B/train1-attempt-01/outputs'
def run(env,model,contract,out,stage):
 plan=read(Path(__file__).with_name('m10_protocol.json'))['training'];rnd=0 if stage=='train' else 1;dev=env.standing.device;old,ends=old_context(env)
 if rnd==0:
  prior=torch.load(PRIOR/'dataset.pt',map_location=dev,weights_only=True)
  keep=[i for i,m in enumerate(prior['manifest']) if 'wrist_' in Path(m['path']).name]
  groups=[prior['groups'][i] for i in keep];manifest=[dict(prior['manifest'][i],pool='wrist') for i in keep];del prior
  sources=plan['sources']
 else:
  prior=torch.load(completed('train')/'dataset.pt',map_location=dev,weights_only=True);groups=prior['groups'];manifest=prior['manifest'];sources=[str(p) for p in completed('collect').glob('*__candidate.json.gz')]
 for source in sources:
  e=read(source);spec=e['condition'];site=spec.get('body','right_wrist_yaw_link');point=spec.get('local_point',[0,0,0]);history=ContactHistory();gate=ContactGate();rows=[]
  for tick,c in enumerate(e['control']):
   check_deadline();R=Rotation.from_quat(np.array(c['pre_qpos'])[[4,5,6,3]]).as_matrix();f=c['measured_force_xyz'];history.append(f,site,point,R)
   if tick%5==0:gate.update(c['time'],f)
   x,l=sensor_features(history,R);rows.append(dict(x=x,l=l,gate=gate.value(c['time']),obs={k:c['bfm_obs'][k][0] for k in env.keys},z=(c['oracle_z'] if rnd else c['executed_z'])[0],q=c['oracle_q_PD'] if rnd else c['q_PD']))
  T=lambda x:torch.tensor(np.array(x),device=dev,dtype=torch.float32);obs={k:T([r['obs'][k] for r in rows]) for k in env.keys};norm=model._normalize(obs);pro=torch.cat([norm[k].reshape(len(rows),-1) for k in env.keys],-1);h=torch.cat([pro,T([r['x'] for r in rows])],-1);h8=torch.cat([pro,T([r['l'] for r in rows])],-1)
  g=dict(h=h,h8=h8,gate=T([r['gate'] for r in rows]),z=T([r['z'] for r in rows]),q=T([r['q'] for r in rows]),obs=obs);g['base']=policy_latent(model,ends,old(h8),g['gate']);groups.append(g)
  manifest.append(dict(path=source,sha256=sha(source),samples=len(rows),pool='body' if site=='torso_link' else ('wrist' if spec.get('family')=='old' else 'walking'),origin='m10',partial=e['boundary'] is not None,condition=spec))
 torch.save(dict(groups=[{k:({kk:vv.cpu() for kk,vv in v.items()} if k=='obs' else v.cpu()) for k,v in g.items()} for g in groups],manifest=manifest),out/'dataset.pt')
 net=load_net(PRIOR/'no-memory-final.pt' if rnd==0 else completed('train')/'student-final.pt',dev);net.train().requires_grad_(True);torch.manual_seed(plan['seed']+rnd);opt=torch.optim.AdamW(net.parameters(),lr=plan['lr'],weight_decay=.0001)
 pools=[[i for i,m in enumerate(manifest) if m['pool']==f] for f in ['wrist','body','walking']];assert all(pools);logs=[];scale=torch.tensor(contract['q_PD_scale'],device=dev);off=torch.tensor(contract['q_PD_offset'],device=dev);qof=lambda a:off+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
 identity=[]
 for g,m in zip(groups,manifest):
  if m.get('origin')!='m10':continue
  errs=[]
  for start in range(0,len(g['h']),128):
   ix=slice(start,start+128);errs.append(float((qof(model.act({k:v[ix] for k,v in g['obs'].items()},g['z'][ix],mean=True))-g['q'][ix]).abs().max()))
  assert max(errs)<1e-4;identity.append(max(errs))
 for step in range(plan['updates'][rnd]):
  check_deadline();pool=pools[step%3];g=groups[pool[int(torch.randint(len(pool),(1,),device=dev))]];ix=torch.randint(len(g['h']),(128,),device=dev)
  with torch.enable_grad():
   z=model.project_z(g['base'][ix]+net(g['h'][ix]));loss=(z-g['z'][ix]).square().mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
  if step%250==0:logs.append(dict(step=step,pool=['wrist','body','walking'][step%3],loss=float(loss)));print('M10_TRANSFER',rnd,step,float(loss),flush=True)
 metrics=[]
 for g,m in zip(groups,manifest):
  errs=[]
  for start in range(0,len(g['h']),128):
   ix=slice(start,start+128);z=model.project_z(g['base'][ix]+net(g['h'][ix]));errs.append(qof(model.act({k:v[ix] for k,v in g['obs'].items()},z,mean=True))-g['q'][ix])
  ee=torch.cat(errs);metrics.append(dict(pool=m['pool'],qpd_rms_deg=float(torch.rad2deg(ee.square().mean().sqrt())),arm_rms_deg=float(torch.rad2deg(ee[:,22:].square().mean().sqrt())),legs_waist_rms_deg=float(torch.rad2deg(ee[:,:15].square().mean().sqrt()))))
 ck=out/'student-final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},fixed_final=True,updates=sum(plan['updates'][:rnd+1]),round=rnd,initialization_sha256=sha(PRIOR/'no-memory-final.pt')),ck)
 result=dict(status='completed',student_optimizer_updates=plan['updates'][rnd],checkpoint=str(ck),sha256=sha(ck),same_state_action_identity_max_rad=max(identity),manifest=manifest,metrics=metrics,logs=logs,pool_sampling='equal updates wrist/body/walking, uniform episodes within pool; no hidden task feature');(out/'training.json').write_text(json.dumps(result,indent=2));return result
