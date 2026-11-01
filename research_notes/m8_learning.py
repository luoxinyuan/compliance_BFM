"""Shared multi-contact legal latent interface and same-state action projection."""
from pathlib import Path
import json,gzip,hashlib
import numpy as np,torch
from contact_projector import constrained,check_deadline
from m1_interface import slerp
from m3_learning import support
from m8_stage import RUN,completed

def blend(a,b,u):
 u=torch.as_tensor(u,device=a.device,dtype=a.dtype).reshape(-1,1);z=slerp(a,b,u)
 exact=torch.where(u==0,a.expand_as(z),torch.where(u==1,b.expand_as(z),z))
 # Preserve endpoint bits AND the original smooth path derivative at initialization.
 return z+(exact-z).detach()
def policy_latent(model,ends,raw,gate):
 base=blend(blend(ends[0],ends[1],raw[:,0].clamp(0,1)),ends[2],raw[:,1].clamp(0,1))
 return constrained(model,base,raw[:,2:],gate)[0]
def project(model,obs,base,gate,target,qof,body=False):
 norm={k:v.detach() for k,v in model._normalize(obs).items()};actor=lambda z:model._actor(norm,z,model.cfg.actor_std).mean.float()
 initial=qof(actor(base))-target
 # Exact representable task expert actions need no numerical drift from needless optimization.
 if float(initial.abs().max())<1e-6:return base.detach(),initial.detach()
 with torch.enable_grad():
  raw=torch.zeros(len(base),256,device=base.device,requires_grad=True);opt=torch.optim.Adam([raw],lr=.05)
  for _ in range(160):
   check_deadline();opt.zero_grad();z=constrained(model,base,raw,gate)[0];e=qof(actor(z))-target
   loss=e.square().mean() if body else e[:,22:].square().mean()+.1*e[:,:22].square().mean();loss.backward();opt.step()
  z=constrained(model,base,raw,gate)[0];e=qof(actor(z))-target
 return z.detach(),e.detach()
class Projector(torch.nn.Module):
 def __init__(self,h):
  super().__init__();self.register_buffer('center',h.mean(0));self.register_buffer('scale',h.std(0,unbiased=False).clamp_min(.1));self.net=torch.nn.Sequential(torch.nn.Linear(h.shape[1],128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,258))
 def forward(self,h):return self.net((h-self.center)/self.scale)
def load_net(p,dev):
 x=torch.load(p,map_location=dev,weights_only=True);net=Projector(torch.zeros(2,len(x['state_dict']['center']),device=dev)).to(dev);net.load_state_dict(x['state_dict']);return net.eval().requires_grad_(False)

def read(p):
 p=Path(p);return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
def write(out,n,x):(out/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def initialize(h,dev):
 from m7_learning import load_net as old_load
 path=RUN.parent/'m7-continuous-20261101A/train2-attempt-02/outputs/distill-final.pt';assert sha(path)=='ba3aa67e3d452579e1cb2e40848aebc4d50fce7bb6d428e0b23b9e6726ac316e'
 old=old_load(path,dev);net=Projector(h).to(dev)
 with torch.no_grad():
  net.center[:1425]=old.center;net.scale[:1425]=old.scale;net.center[1425:]=0;net.scale[1425:]=1
  net.net[0].weight.zero_();net.net[0].weight[:,:1425]=old.net[0].weight;net.net[0].bias.copy_(old.net[0].bias)
  net.net[2].load_state_dict(old.net[2].state_dict());net.net[4].weight.zero_();net.net[4].bias.zero_();net.net[4].weight[0]=old.net[4].weight[0];net.net[4].bias[0]=old.net[4].bias[0];net.net[4].weight[2:]=old.net[4].weight[1:];net.net[4].bias[2:]=old.net[4].bias[1:]
  ix=torch.tensor([0]+list(range(2,258)),device=dev);error=float((net(h[:16])[:,ix]-old(h[:16,:1425])).abs().max());assert error<2e-6
 return net,dict(source=str(path),sha256=sha(path),old_output_initial_max_abs=error,new_input='contact link wrist/torso one-hot + local point XYZ, static sensor geometry; NOT task label',new_output='two continuous native coordinates + original bounded tangent, no selector; same BFM sphere16')
def build(env,model,contract,out,rnd):
 dev,qof=support(env,model,contract);rows=[];manifest=[]
 sources=[completed('contact')/'down_dev__teacher.json.gz',completed('contact')/'down_free__teacher.json.gz',completed('expert')/'down_dev2__teacher.json.gz'] if rnd==0 else sorted(completed('collect'+str(rnd)).glob('*__*.json.gz'))
 for path in sources:
  e=read(path);group=len(manifest);cs=e['control'];ix=list(range(0,len(cs),2));body=e['condition']['body']=='torso_link'
  for i in ix:
   c=cs[i];assert c['oracle_query_valid'];c['student_input'][-5:]=[x*float(max(abs(v) for v in c['student_input'][465:1425])>1e-8) for x in c['student_input'][-5:]];rows.append(dict(h=c['student_input'],obs={k:c['bfm_obs'][k][0] for k in env.keys},base=c['oracle_z'][0],q=c['oracle_q_PD'],base_q=c['oracle_base_q_PD'],gate=c['local_gate'],group=group,body=body))
  manifest.append(dict(path=str(path),sha256=sha(path),samples=len(ix),controller=e['controller'],condition=e['condition'],group=group))
 d={k:torch.tensor([r[k] for r in rows],device=dev,dtype=torch.float32) for k in ['h','base','q','base_q','gate','body']};d['group']=torch.tensor([r['group'] for r in rows],device=dev);d['obs']={k:torch.tensor([r['obs'][k] for r in rows],device=dev) for k in env.keys};d['manifest']=manifest
 zs=torch.zeros_like(d['base']);ee=torch.zeros_like(d['q']);report=[]
 for g in d['group'].unique():
  indices=(d['group']==g).nonzero().flatten();b=bool(d['body'][indices[0]])
  for start in range(0,len(indices),64):
   ix=indices[start:start+64];z,e=project(model,{k:v[ix] for k,v in d['obs'].items()},d['base'][ix],d['gate'][ix],d['q'][ix],qof,b);zs[ix]=z;ee[ix]=e
  report.append(dict(group=int(g),body=b,leg_waist_rms_deg=float(torch.rad2deg(ee[indices,:15].square().mean().sqrt())),right_arm_rms_deg=float(torch.rad2deg(ee[indices,22:].square().mean().sqrt())),all_max_deg=float(torch.rad2deg(ee[indices].abs().max()))));print('M8_PROJECT',rnd,report[-1],flush=True)
 d['z']=zs;write(out,'new-data-projection.json',dict(groups=report,source=manifest,samples=len(rows),steps_at_most=160,exact_native_action_shortcut=True,scope='Body expert queries original BFM at signed-force feedback latent; exact representability checked on actual observations. Wrist expert independently has original bounded right-arm correction. No direct alpha-only supervision.'))
 oldpath=completed('train'+str(rnd-1))/'dataset.pt' if rnd else RUN.parent/'m7-continuous-20261101A/train2-attempt-02/outputs/dataset.pt';old=torch.load(oldpath,map_location=dev,weights_only=True);write(out,'prior-dataset.json',dict(path=str(oldpath),sha256=sha(oldpath)))
 if rnd==0:
  extra=torch.zeros(len(old['h']),5,device=dev);extra[:,0]=(old['h'][:,465:].abs().amax(-1)>1e-8).float();old['h']=torch.cat([old['h'],extra],-1);old['body']=torch.zeros(len(old['h']),device=dev)
 d['group']+=int(old['group'].max())+1
 for k in ['h','base','q','base_q','gate','group','z','body']:d[k]=torch.cat([old[k],d[k]])
 d['obs']={k:torch.cat([old['obs'][k],d['obs'][k]]) for k in env.keys};d['manifest']=old['manifest']+manifest
 torch.save({k:{kk:vv.cpu() for kk,vv in v.items()} if k=='obs' else v.cpu() if torch.is_tensor(v) else v for k,v in d.items()},out/'dataset.pt');return d

def train(env,model,contract,out,rnd):
 from m1_interface import endpoints
 plan=read(Path(__file__).with_name('m8_protocol.json'))['training'];dev,qof=support(env,model,contract);oldends=endpoints(env);bodyz=torch.tensor(read(completed('qualification')/'candidate-1.json')['z'],device=dev);ends=[oldends[0],oldends[2],bodyz];d=build(env,model,contract,out,rnd)
 norm={k:v.detach() for k,v in model._normalize(d['obs']).items()};groups=[(d['group']==g).nonzero().flatten() for g in d['group'].unique()];family=[]
 for m in d['manifest']:
  c=m['condition'];family.append('body' if c.get('body')=='torso_link' else 'transition' if c.get('transition') else 'negative' if c['delta_xyz'][0]<0 else 'positive' if c['enabled'] else 'free')
 pools=[[groups[i] for i,f in enumerate(family) if f==k] for k in ['body','transition','positive','negative','free']];assert all(pools);reports={}
 for method in ['distill','alignment']:
  torch.manual_seed(0)
  if rnd==0:net,init=initialize(d['h'],dev);write(out,method+'-initialization.json',init)
  else:net=load_net(completed('train'+str(rnd-1))/(method+'-final.pt'),dev)
  net.train().requires_grad_(True);opt=torch.optim.AdamW(net.parameters(),lr=plan['lr'],weight_decay=.0001);rng=torch.Generator(device=dev).manual_seed(rnd);logs=[]
  def metrics():
   e=[];dz=[]
   with torch.no_grad():
    for start in range(0,len(d['h']),128):
     ix=slice(start,start+128);z=policy_latent(model,ends,net(d['h'][ix]),d['gate'][ix]);a=model._actor({k:v[ix] for k,v in norm.items()},z,model.cfg.actor_std).mean.float();e.append(qof(a)-d['q'][ix]);dz.append(z-d['z'][ix])
   ee=torch.cat(e);zz=torch.cat(dz);return dict(latent_rms=float(zz.square().mean().sqrt()),qpd_rms_deg=float(torch.rad2deg(ee.square().mean().sqrt())),groups=[dict(group=i,family=family[i],qpd_rms_deg=float(torch.rad2deg(ee[g].square().mean().sqrt())),leg_waist_rms_deg=float(torch.rad2deg(ee[g,:15].square().mean().sqrt())),right_arm_rms_deg=float(torch.rad2deg(ee[g,22:].square().mean().sqrt())),latent_rms=float(zz[g].square().mean().sqrt())) for i,g in enumerate(groups)])
  before=metrics()
  for step in range(plan['steps'][rnd]):
   check_deadline();which=torch.randint(10,(plan['batch'],),generator=rng,device=dev);selected=[pools[0 if int(i)<5 else 1 if int(i)<7 else int(i)-5] for i in which];sample_groups=[p[int(torch.randint(len(p),(1,),generator=rng,device=dev))] for p in selected];ix=torch.stack([g[torch.randint(len(g),(1,),generator=rng,device=dev)[0]] for g in sample_groups])
   with torch.enable_grad():
    opt.zero_grad();z=policy_latent(model,ends,net(d['h'][ix]),d['gate'][ix]);lz=(z-d['z'][ix]).square().mean()
    if method=='alignment':
     a=model._actor({k:v[ix] for k,v in norm.items()},z,model.cfg.actor_std).mean.float();e=qof(a)-d['q'][ix];wrist=e[:,22:].square().mean(-1)+.1*e[:,:22].square().mean(-1);body=e.square().mean(-1);la=torch.where(d['body'][ix]>0,body,wrist).mean();loss=la+.01*lz
    else:la=torch.zeros((),device=dev);loss=lz
    assert torch.isfinite(loss);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1));opt.step()
   if step%250==0:
    row=dict(step=step,loss=float(loss),q_loss=float(la),z_loss=float(lz),gradient_norm=gn);logs.append(row);print('M8_TRAIN',rnd,method,row,flush=True)
  after=metrics();ck=out/(method+'-final.pt');torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},round=rnd,m8_updates=sum(plan['steps'][:rnd+1]),inherited_m7_pz_updates=20500,method=method,input='1430D: native/force history + contact link/local position; no privilege',fixed_final=True),ck);reports[method]=dict(before=before,after=after,updates=plan['steps'][rnd],checkpoint=str(ck),sha256=sha(ck),logs=logs);write(out,'training.json',reports)
 return dict(status='completed',student_optimizer_updates=2*plan['steps'][rnd],samples=len(d['h']),reports=reports,aggregation_round=rnd)
