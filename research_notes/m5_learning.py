"""FIBER-inspired action projection and deployable on-policy LPP, not full FIBER.
Native sphere prior replaces the paper's B(s) term; original 50 Hz actor retained.
The learned progress is NOT supplied by the expert or task clock at deployment.
"""
import gzip,json,hashlib
from pathlib import Path
import numpy as np,torch
from contact_projector import constrained,check_deadline
from m1_interface import endpoints,slerp
from m3_learning import project,support
RUN=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m5-contact-lpp-20261031A')
def read(p):
 p=Path(p);return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
def write(out,n,x):(out/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def done(stage):
 a=[a for a in read(RUN/'compute-ledger.json')['attempts'] if a['stage']==stage and a.get('result_status')=='completed'];assert len(a)==1;return Path(a[0]['path'])/'outputs'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Projector(torch.nn.Module):
 def __init__(self,h):
  super().__init__();self.register_buffer('center',h.mean(0));self.register_buffer('scale',h.std(0,unbiased=False).clamp_min(.1));self.net=torch.nn.Sequential(torch.nn.Linear(h.shape[1],128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,257));torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
 def forward(self,h):return self.net((h-self.center)/self.scale)
def load_net(p,dev):
 x=torch.load(p,map_location=dev,weights_only=True);net=Projector(torch.zeros(2,len(x['state_dict']['center']),device=dev));net.load_state_dict(x['state_dict']);return net.eval().requires_grad_(False)
def policy_latent(model,ends,raw,gate):return constrained(model,slerp(ends[0],ends[2],raw[:,0].clamp(0,1)),raw[:,1:],gate)[0]
def build(env,model,contract,out,rnd):
 dev,qof=support(env,model,contract);src=done('qualification' if rnd==0 else 'collect'+str(rnd));rows=[];manifest=[]
 for path in sorted(src.glob('*.json.gz')):
  e=read(path)
  if e['controller'] not in ['teacher','alignment','distill']:continue
  group=len(manifest);cs=e['control'];ix=list(range(0,len(cs),2))
  valid=[i for i in ix if cs[i].get('oracle_query_valid',False)]
  if len(valid)!=len(ix):raise ValueError('Expert query outside existing actuator bounds; diagnose, not silently drop')
  for i in ix:
   c=cs[i];rows.append(dict(h=c['student_input'],obs={k:c['bfm_obs'][k][0] for k in env.keys},base=c['oracle_z'][0],q=c['oracle_q_PD'],base_q=c['oracle_base_q_PD'],gate=c['local_gate'],group=group))
  manifest.append(dict(path=str(path),sha256=sha(path),samples=len(ix),controller=e['controller'],condition=e['condition'],group=group))
 d={k:torch.tensor([r[k] for r in rows],device=dev,dtype=torch.float32) for k in ['h','base','q','base_q','gate']};d['group']=torch.tensor([r['group'] for r in rows],device=dev);d['obs']={k:torch.tensor([r['obs'][k] for r in rows],device=dev) for k in env.keys};d['manifest']=manifest
 zs=[];ee=[]
 for start in range(0,len(rows),64):
  ix=slice(start,min(start+64,len(rows)));z,e=project(model,{k:v[ix] for k,v in d['obs'].items()},d['base'][ix],d['gate'][ix],d['q'][ix],qof);zs.append(z);ee.append(e)
  if start%512==0:print('M5_PROJECT',rnd,start,len(rows),float(e.square().mean().sqrt()),flush=True)
 d['z']=torch.cat(zs);err=torch.cat(ee);report=[]
 for g in d['group'].unique():
  mask=d['group']==g;report.append(dict(group=int(g),arm_rms_deg=float(torch.rad2deg(err[mask,22:].square().mean().sqrt())),nonarm_rms_deg=float(torch.rad2deg(err[mask,:22].square().mean().sqrt())),arm_max_deg=float(torch.rad2deg(err[mask,22:].abs().max())),base_increment_arm_rms_deg=float(torch.rad2deg((d['q'][mask,22:]-d['base_q'][mask,22:]).square().mean().sqrt()))))
 write(out,'new-data-projection.json',dict(groups=report,source=manifest,steps=160,samples=len(rows),selection='Every second actual 50Hz frame; complete actual force history and actual oracle query; no final test',prior='original native-centered tangent, global executable z labels'))
 if rnd:
  old=torch.load(done('train'+str(rnd-1))/'dataset.pt',map_location=dev,weights_only=True);d['group']+=int(old['group'].max())+1
  for k in ['h','base','q','base_q','gate','group','z']:d[k]=torch.cat([old[k],d[k]])
  d['obs']={k:torch.cat([old['obs'][k],d['obs'][k]]) for k in env.keys};d['manifest']=old['manifest']+manifest
 torch.save({k:{kk:vv.cpu() for kk,vv in v.items()} if k=='obs' else v.cpu() if torch.is_tensor(v) else v for k,v in d.items()},out/'dataset.pt')
 return d

def train(env,model,contract,out,rnd):
 plan=read(Path(__file__).with_name('m5_protocol.json'))['training'];dev,qof=support(env,model,contract);ends=endpoints(env);d=build(env,model,contract,out,rnd);norm={k:v.detach() for k,v in model._normalize(d['obs']).items()};groups=[(d['group']==g).nonzero().flatten() for g in d['group'].unique()];reports={}
 for method in ['alignment','distill']:
  torch.manual_seed(0);net=Projector(d['h']).to(dev) if rnd==0 else load_net(done('train'+str(rnd-1))/(method+'-final.pt'),dev);net.train().requires_grad_(True);opt=torch.optim.AdamW(net.parameters(),lr=plan['lr'],weight_decay=.0001);rng=torch.Generator(device=dev).manual_seed(rnd);logs=[]
  def metrics():
   ee=[];zz=[]
   with torch.no_grad():
    for start in range(0,len(d['h']),64):
     ix=slice(start,start+64);z=policy_latent(model,ends,net(d['h'][ix]),d['gate'][ix]);a=model._actor({k:v[ix] for k,v in norm.items()},z,model.cfg.actor_std).mean.float();ee.append(qof(a)-d['q'][ix]);zz.append(z-d['z'][ix])
   e=torch.cat(ee);dz=torch.cat(zz);return dict(latent_rms=float(dz.square().mean().sqrt()),arm_rms_deg=float(torch.rad2deg(e[:,22:].square().mean().sqrt())),nonarm_rms_deg=float(torch.rad2deg(e[:,:22].square().mean().sqrt())),per_group=[dict(group=i,arm_rms_deg=float(torch.rad2deg(e[g,22:].square().mean().sqrt())),latent_rms=float(dz[g].square().mean().sqrt())) for i,g in enumerate(groups)])
  before=metrics()
  for step in range(plan['steps'][rnd]):
   check_deadline();chosen=torch.randint(len(groups),(plan['batch'],),generator=rng,device=dev);ix=torch.stack([groups[int(g)][torch.randint(len(groups[int(g)]),(1,),generator=rng,device=dev)[0]] for g in chosen])
   with torch.enable_grad():
    opt.zero_grad();z=policy_latent(model,ends,net(d['h'][ix]),d['gate'][ix]);lz=(z-d['z'][ix]).square().mean()
    if method=='alignment' and not(rnd==0 and step<500):
     a=model._actor({k:v[ix] for k,v in norm.items()},z,model.cfg.actor_std).mean.float();e=qof(a)-d['q'][ix];la=e[:,22:].square().mean()+.1*e[:,:22].square().mean();loss=la+.01*lz
    else:la=torch.zeros((),device=dev);loss=lz
    assert torch.isfinite(loss);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1));opt.step()
   if step%250==0:
    row=dict(step=step,loss=float(loss),q_loss=float(la),z_loss=float(lz),gradient_norm=grad);logs.append(row);print('M5_TRAIN',rnd,method,row,flush=True)
  after=metrics();ck=out/(method+'-final.pt');torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},round=rnd,total_updates=sum(plan['steps'][:rnd+1]),method=method,input='1425D causal native/force history; no privilege',fixed_final=True),ck)
  reports[method]=dict(before=before,after=after,updates=plan['steps'][rnd],total_updates=sum(plan['steps'][:rnd+1]),checkpoint=str(ck),sha256=sha(ck),logs=logs);write(out,'training.json',reports)
 return dict(status='completed',student_optimizer_updates=2*plan['steps'][rnd],reports=reports,aggregation_round=rnd,samples=len(d['h']))
