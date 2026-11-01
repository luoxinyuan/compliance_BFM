"""M8-based shared legal-z residual learner. Native experts need no identity fitting.
All new features are real contact provenance, proprioception and action history.
"""
import gzip,json,hashlib
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
from m9_stage import P,RUN,completed
from m9_history import ContactHistory
from contact_projector import ContactGate,check_deadline
from m8_learning import load_net as load_old,policy_latent as old_latent
from m1_interface import endpoints

OLD=P/'bridge-v2-owned/runs/m8-contact-body-20261101A/train2-attempt-01/outputs/distill-final.pt'
OLD_SHA='74f624a96abc3fd4ea63da101dfb0177bf8f426d6fa4da8717d7b219aaee086a'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p)
 if p.suffix=='.gz':
  with gzip.open(p,'rt') as f:return json.load(f)
 return json.loads(p.read_text())
def old_context(env):
 assert sha(OLD)==OLD_SHA
 old=load_old(OLD,env.standing.device);ends=endpoints(env)
 ledger=read(RUN.parent/'m8-contact-body-20261101A/compute-ledger.json');qa=[a for a in ledger['attempts'] if a['stage']=='qualification' and a.get('result_status')=='completed'];assert len(qa)==1
 body=torch.tensor(read(Path(qa[0]['path'])/'outputs/candidate-1.json')['z'],device=env.standing.device)
 return old,[ends[0],ends[2],body]

def sensor_features(history,R):
 x=history.features(R);detected=[r for r in history.rows if r['contact_detected']]
 link=detected[-1]['link'] if detected else None;point=detected[-1]['local_point'] if detected else [0,0,0]
 rows=list(history.rows);valid=np.array([r['link']==link and r['contact_detected'] for r in rows]);f=x[:,:3].copy()
 if len(rows):f[-len(rows):]*=valid[:,None]
 enc=np.r_[link=='right_wrist_yaw_link',link=='torso_link',point]
 return x.reshape(-1),np.r_[f.reshape(-1),enc]

def inputs(model,obs,keys,history,R):
 assert set(keys)=={'state','last_action','history_actor'}
 norm=model._normalize(obs);pro=torch.cat([norm[k].reshape(len(obs[k]),-1).float() for k in keys],-1);x,legacy=sensor_features(history,R)
 h=torch.cat([pro,torch.tensor(x,device=pro.device,dtype=pro.dtype)[None]],-1);h8=torch.cat([pro,torch.tensor(legacy,device=pro.device,dtype=pro.dtype)[None]],-1)
 return h,h8

class ResidualPolicy(torch.nn.Module):
 def __init__(self,h):
  super().__init__();self.register_buffer('center',h.mean(0));self.register_buffer('scale',h.std(0,unbiased=False).clamp_min(.1));self.net=torch.nn.Sequential(torch.nn.Linear(h.shape[1],128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,256));torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
 def forward(self,h):return self.net((h-self.center)/self.scale)
def load_net(p,dev):
 x=torch.load(p,map_location=dev,weights_only=True);net=ResidualPolicy(torch.zeros(2,len(x['state_dict']['center']),device=dev)).to(dev);net.load_state_dict(x['state_dict']);return net.eval().requires_grad_(False)
def policy(model,old,ends,net,h,h8,gate):
 base=old_latent(model,ends,old(h8),gate);z=model.project_z(base+net(h));active=h[:,465:].reshape(-1,320,14)[:,:,:3].abs().amax((1,2))>1e-8
 return torch.where(active[:,None],z,ends[0].expand_as(z))

def build(env,model,contract,out,rnd,old,ends,plan):
 assert set(env.keys)=={'state','last_action','history_actor'}
 dev=env.standing.device;groups=[];manifest=[]
 sources=[Path(p) for p in plan['sources']] if rnd==0 else sorted(completed('collect'+str(rnd)).glob('*__student.json.gz'))
 for source in sources:
  e=read(source);assert e['boundary'] is None or rnd>0;history=ContactHistory();gate=ContactGate();rows=[];spec=e['condition'];site=spec.get('body','right_wrist_yaw_link');point=spec.get('local_point',[0,0,0])
  for tick,c in enumerate(e['control']):
   check_deadline();R=Rotation.from_quat(np.array(c['pre_qpos'])[[4,5,6,3]]).as_matrix();f=c.get('measured_force_xyz',c.get('sensor',{}).get('force_world'));assert f is not None
   history.append(f,site,point,R)
   if tick%5==0:gate.update(c['time'],f)
   x,l=sensor_features(history,R);rows.append(dict(x=x,l=l,obs={k:c['bfm_obs'][k][0] for k in env.keys},z=(c['oracle_z'] if rnd else c['executed_z'])[0],q=c['oracle_q_PD'] if rnd else c['q_PD'],gate=gate.value(c['time'])))
  ob={k:torch.tensor(np.array([r['obs'][k] for r in rows]),device=dev,dtype=torch.float32) for k in env.keys};norm=model._normalize(ob);pro=torch.cat([norm[k].reshape(len(rows),-1).float() for k in env.keys],-1);h=torch.cat([pro,torch.tensor(np.array([r['x'] for r in rows]),device=dev,dtype=torch.float32)],-1);h8=torch.cat([pro,torch.tensor(np.array([r['l'] for r in rows]),device=dev,dtype=torch.float32)],-1)
  group=dict(h=h,h8=h8,z=torch.tensor([r['z'] for r in rows],device=dev),q=torch.tensor([r['q'] for r in rows],device=dev),gate=torch.tensor([r['gate'] for r in rows],device=dev),obs=ob)
  group['base']=old_latent(model,ends,old(h8),group['gate']);groups.append(group);manifest.append(dict(path=str(source),sha256=sha(source),samples=len(rows),condition=spec,partial=e['boundary'] is not None,family='old' if 'm8-contact-body-' in str(source) or spec.get('family')=='old' else 'walking'))
  print('M9_DATA',source.name,len(rows),flush=True)
 if rnd:
  prior=torch.load(completed('train'+str(rnd-1))/'dataset.pt',map_location=dev,weights_only=True);groups=prior['groups']+groups;manifest=prior['manifest']+manifest
 torch.save(dict(groups=[{k:{kk:vv.cpu() for kk,vv in v.items()} if k=='obs' else v.cpu() for k,v in g.items()} for g in groups],manifest=manifest),out/'dataset.pt')
 return groups,manifest

def train(env,model,contract,out,rnd):
 plan=read(Path(__file__).with_name('m9_protocol.json'))['training_fixed'];old,ends=old_context(env);groups,manifest=build(env,model,contract,out,rnd,old,ends,plan);dev=env.standing.device
 torch.manual_seed(0);net=ResidualPolicy(torch.cat([g['h'] for g in groups])).to(dev) if not rnd else load_net(completed('train'+str(rnd-1))/'projected-z-final.pt',dev);net.train().requires_grad_(True);opt=torch.optim.AdamW(net.parameters(),lr=plan['lr'],weight_decay=.0001);rng=torch.Generator(device=dev).manual_seed(rnd)
 scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev);qof=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
 parity=[]
 for g in groups:
  errs=[]
  for start in range(0,len(g['h']),128):
   ix=slice(start,start+128);a=model.act({k:v[ix] for k,v in g['obs'].items()},g['z'][ix],mean=True);errs.append(qof(a)-g['q'][ix])
  ee=torch.cat(errs);parity.append(dict(qpd_native_label_rms_deg=float(torch.rad2deg(ee.square().mean().sqrt())),peak_deg=float(torch.rad2deg(ee.abs().max()))));assert ee.abs().max()<1e-4,'Native legal z/action identity mismatch'
 pools=[[i for i,m in enumerate(manifest) if m['family']==f] for f in ['old','walking']];assert all(pools);logs=[]
 for step in range(plan['steps'][rnd]):
  check_deadline();pool=pools[step%2];gi=pool[int(torch.randint(len(pool),(1,),generator=rng,device=dev))];g=groups[gi];ix=torch.randint(len(g['h']),(plan['batch'],),generator=rng,device=dev)
  with torch.enable_grad():
   opt.zero_grad();z=model.project_z(g['base'][ix]+net(g['h'][ix]));loss=(z-g['z'][ix]).square().mean();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1));opt.step()
  if step%250==0:logs.append(dict(step=step,loss=float(loss),gradient_norm=gn));print('M9_TRAIN',rnd,logs[-1],flush=True)
 metrics=[]
 with torch.no_grad():
  for g in groups:
   errs=[];zs=[]
   for start in range(0,len(g['h']),128):
    ix=slice(start,start+128);z=model.project_z(g['base'][ix]+net(g['h'][ix]));a=model.act({k:v[ix] for k,v in g['obs'].items()},z,mean=True);errs.append(qof(a)-g['q'][ix]);zs.append(z-g['z'][ix])
   ee=torch.cat(errs);metrics.append(dict(qpd_rms_deg=float(torch.rad2deg(ee.square().mean().sqrt())),leg_waist_rms_deg=float(torch.rad2deg(ee[:,:15].square().mean().sqrt())),right_arm_rms_deg=float(torch.rad2deg(ee[:,22:].square().mean().sqrt())),z_rms=float(torch.cat(zs).square().mean().sqrt())))
 ck=out/'projected-z-final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},m8_base_path=str(OLD),m8_base_sha=OLD_SHA,round=rnd,m9_updates=sum(plan['steps'][:rnd+1]),fixed_final=True,input='normalized proprio/action + per-sample force/link/point/frame/validity history; no anchor/time/task IDs',output='native sphere16 global-z residual over frozen M8-PZ base'),ck)
 result=dict(status='completed',student_optimizer_updates=plan['steps'][rnd],checkpoint=str(ck),sha256=sha(ck),native_projection_identity=parity,training_metrics=metrics,logs=logs,manifest=manifest);(out/'training.json').write_text(json.dumps(result,indent=2));return result
