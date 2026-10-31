"""Conditional single Projected-z extension; no heldout data and no BFM update."""
import gzip,json,hashlib
from pathlib import Path
import numpy as np,torch
from alpha_student import student_input
from alpha_memory import ForceMemory
from contact_projector import constrained,check_deadline
from m1_interface import endpoints,policy_latent
from m1_learning import load_net
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs');RUN=P/'m3-continuous-body-20261031A';M1=P/'m1-state-response-20261031A/aggregate-attempt-01/outputs'
def read(p):return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
def write(out,n,x):(out/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def done(stage):
 a=[a for a in read(RUN/'compute-ledger.json')['attempts'] if a['stage']==stage and a.get('result_status')=='completed'];assert len(a)==1;return Path(a[0]['path'])/'outputs'
def selected():return read(Path(__file__).with_name('m3_training_plan.json'))['teacher_sources']
def support(env,model,contract):
 dev=env.standing.device;scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev)
 return dev,lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
def project(model,obs,base,gate,target,qof):
 norm={k:v.detach() for k,v in model._normalize(obs).items()}
 actor=lambda z:model._actor(norm,z,model.cfg.actor_std).mean.float()
 with torch.enable_grad():
  raw=torch.zeros(len(base),256,device=base.device,requires_grad=True);opt=torch.optim.Adam([raw],lr=.05)
  for _ in range(160):
   check_deadline();opt.zero_grad();z=constrained(model,base,raw,gate)[0];e=qof(actor(z))-target;loss=e[:,22:].square().mean()+.1*e[:,:22].square().mean();loss.backward();opt.step()
  z=constrained(model,base,raw,gate)[0];e=qof(actor(z))-target
 return z.detach(),e.detach()
def representation(env,model,contract,out):
 dev,qof=support(env,model,contract);records=[]
 for source in selected():
  ep=read(done('qualification')/source);cs=ep['control']
  for t in [13.,15.,16.8,17.6]:
   i=int(np.argmin(abs(np.array([c['time'] for c in cs])-t)));b=cs[i:i+5];obs={k:torch.tensor([c['bfm_obs'][k][0] for c in b],device=dev) for k in env.keys};base=torch.tensor([c['executed_z'][0] for c in b],device=dev);gate=torch.tensor([c['local_gate'] for c in b],device=dev);q=torch.tensor([c['q_PD'] for c in b],device=dev);z,e=project(model,obs,base,gate,q,qof);delta=q-torch.tensor([c['base_q_PD'] for c in b],device=dev)
   arm=float(e[:,22:].square().mean().sqrt());nonarm=float(e[:,:22].square().mean().sqrt());ref=float(delta[:,22:].square().mean().sqrt());ok=arm<=max(.5*ref,.001) and nonarm<=.005
   row=dict(source=source,time=t,gate=gate.tolist(),arm_rms_rad=arm,nonarm_rms_rad=nonarm,teacher_delta_arm_rms_rad=ref,passed=ok,norm_error=float(abs(z.norm(dim=-1)-16).max()));records.append(row);print('M3_REPRESENTATION',row,flush=True)
 result=dict(status='completed' if all(r['passed'] for r in records) else 'capability-stopped',student_optimizer_updates=0,blocks=records,criterion='OriginalM1 localactionexpression: arm<=max(.5*teacherincrement,.001rad), nonarm<=.005rad; finitefixed160steps, actualgates/50Hzframes. No model updates.')
 write(out,'representation.json',result);return result
def build(env,model,out):
 dev=env.standing.device;plan=read(Path(__file__).with_name('m3_training_plan.json'));rows=[];manifest=[]
 sources=[(done('qualification')/s,'teacher') for s in selected()]+[(done(stage)/name,'negative') for stage,name in plan['negative_sources']]
 old=torch.load(M1/'dataset.pt',map_location=dev,weights_only=True);oldz=torch.load(M1/'projected-global-z.pt',map_location=dev,weights_only=True);assert set(m['configuration'] for m in old['metadata'])=={'standing','raised'};group_offset=int(old['group'].max())+1
 for group,(path,kind) in enumerate(sources,group_offset):
  e=read(path);memory=ForceMemory(320);hist=[memory.append(c['measured_force_xyz']).copy() for c in e['control']];cs=e['control'];ready=e['control_ready_s'];on=e['condition']['onset'];rel=e['condition']['release'];ix=[]
  for i,c in enumerate(cs):
   t=c['time']
   if t<ready-1e-8:continue
   dense=kind=='teacher' and on-1e-8<=t<rel+2-1e-8
   if dense or i%5==0:ix.append(i)
  obs={k:torch.tensor([cs[i]['bfm_obs'][k][0] for i in ix],device=dev) for k in env.keys};h=student_input(model,obs,[hist[i] for i in ix],env.keys).detach();assert h.shape[1]==1425
  for j,i in enumerate(ix):
   c=cs[i];rows.append(dict(h=h[j],obs={k:obs[k][j] for k in env.keys},base=c['executed_z'][0],q=c['q_PD'],base_q=c['base_q_PD'],action=c['action'],gate=c['local_gate'],group=group,configuration=e['configuration'],kind=kind,time=c['time']))
  manifest.append(dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),samples=len(ix),kind=kind,group=group))
 new=dict(h=torch.stack([r['h'] for r in rows]),obs={k:torch.stack([r['obs'][k] for r in rows]) for k in env.keys},metadata=[{k:r[k] for k in ['configuration','kind','time','group']} for r in rows])
 for k in ['base','q','base_q','action','gate']:new[k]=torch.tensor([r[k] for r in rows],device=dev,dtype=torch.float32)
 new['group']=torch.tensor([r['group'] for r in rows],device=dev)
 write(out,'dataset-manifest.json',dict(old_dataset=str(M1/'dataset.pt'),old_sha=hashlib.sha256((M1/'dataset.pt').read_bytes()).hexdigest(),old_z_sha=hashlib.sha256((M1/'projected-global-z.pt').read_bytes()).hexdigest(),old_samples=len(old['h']),new_sources=manifest,new_samples=len(rows),selection=plan,sampling='Teachercontact andfirst2srelease50Hz; precontact/late8srelease/negative10Hz. Features reconstructed fromALLactual50Hzhistory, neverdownsamplehistory. Old2600targets reused notrefit. Noheldoutorweakgoal.5labels.'))
 return old,oldz,new
def train(env,model,contract,out):
 rp=read(done('representation')/'representation.json');assert all(b['passed'] for b in rp['blocks']);qa=read(RUN/'analysis/query-parity.json');assert len(qa)==6 and all(r['query_within_original_bounds'] for r in qa)
 dev,qof=support(env,model,contract);ends=endpoints(env);old,oldz,new=build(env,model,out);zs=[];errs=[]
 for start in range(0,len(new['h']),32):
  ix=slice(start,min(start+32,len(new['h'])));z,e=project(model,{k:v[ix] for k,v in new['obs'].items()},new['base'][ix],new['gate'][ix],new['q'][ix],qof);zs.append(z);errs.append(e)
  if start%256==0:print('M3_LABELS',start,len(new['h']),float(e.square().mean().sqrt()),flush=True)
 nz=torch.cat(zs);err=torch.cat(errs);groups=[]
 for g in new['group'].unique():
  m=new['group']==g;arm=float(err[m,22:].square().mean().sqrt());nonarm=float(err[m,:22].square().mean().sqrt());ref=float((new['q'][m,22:]-new['base_q'][m,22:]).square().mean().sqrt());groups.append(dict(group=int(g),arm_rms_rad=arm,nonarm_rms_rad=nonarm,teacher_delta_arm_rms_rad=ref,passed=arm<=max(.5*ref,.001) and nonarm<=.005))
 write(out,'label-projection.json',dict(groups=groups,steps=160,old_targets_reused=True))
 if not all(g['passed'] for g in groups):return dict(status='capability-stopped',student_optimizer_updates=0,reason='Newfull-dataactionprojectionfailed; noStudentfit',groups=groups)
 d={k:torch.cat([old[k],new[k]]) for k in ['h','base','q','base_q','action','gate','group']};d['obs']={k:torch.cat([old['obs'][k],new['obs'][k]]) for k in env.keys};d['metadata']=old['metadata']+new['metadata'];targets=torch.cat([oldz,nz]);torch.save({k:{kk:vv.cpu() for kk,vv in v.items()} if k=='obs' else v.cpu() if torch.is_tensor(v) else v for k,v in d.items()},out/'dataset.pt');torch.save(targets.cpu(),out/'projected-global-z.pt')
 plan=read(Path(__file__).with_name('m3_training_plan.json'));init=M1/'distill-final.pt';assert hashlib.sha256(init.read_bytes()).hexdigest()==plan['initial_sha256'];net=load_net(init,dev);net.train().requires_grad_(True);opt=torch.optim.Adam(net.parameters(),lr=.0003);rng=torch.Generator(device=dev).manual_seed(0);gg=[(d['group']==g).nonzero().flatten() for g in d['group'].unique()];norm={k:v.detach() for k,v in model._normalize(d['obs']).items()};logs=[]
 def metrics():
  ee=[];zz=[]
  with torch.no_grad():
   for start in range(0,len(d['h']),64):
    ix=slice(start,min(start+64,len(d['h'])));z=policy_latent(model,ends,net(d['h'][ix]),d['gate'][ix]);a=model._actor({k:v[ix] for k,v in norm.items()},z,model.cfg.actor_std).mean.float();ee.append(qof(a)-d['q'][ix]);zz.append(z-targets[ix])
  e=torch.cat(ee);dz=torch.cat(zz);return dict(latent_rms=float(dz.square().mean().sqrt()),arm_rms_rad=float(e[:,22:].square().mean().sqrt()),nonarm_rms_rad=float(e[:,:22].square().mean().sqrt()),per_group=[dict(group=i,latent_rms=float(dz[g].square().mean().sqrt()),arm_rms_rad=float(e[g,22:].square().mean().sqrt()),nonarm_rms_rad=float(e[g,:22].square().mean().sqrt())) for i,g in enumerate(gg)])
 before=metrics()
 for step in range(1500):
  check_deadline();chosen=torch.randint(len(gg),(16,),generator=rng,device=dev);ix=torch.stack([gg[int(g)][torch.randint(len(gg[int(g)]),(1,),generator=rng,device=dev)[0]] for g in chosen])
  with torch.enable_grad():
   opt.zero_grad();z=policy_latent(model,ends,net(d['h'][ix]),d['gate'][ix]);loss=(z-targets[ix]).square().mean();assert torch.isfinite(loss);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(net.parameters(),10));opt.step()
  if step%100==0:row=dict(step=step,loss=float(loss),gradient_norm_before_clip=grad);logs.append(row);print('M3_TRAIN',row,flush=True)
 after=metrics();ck=out/'projected-z-final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},updates_this_stage=1500,total_updates=3000,method='Projected-z extension',initial_sha256=plan['initial_sha256'],student_input='1425Ddeployablehistory',fixed_final=True),ck);torch.save(opt.state_dict(),out/'optimizer-final.pt')
 result=dict(status='completed',student_optimizer_updates=1500,steps=1500,batch=16,seed=0,lr=.0003,old_samples=len(old['h']),new_samples=len(new['h']),before=before,after=after,logs=logs,checkpoint=str(ck),sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),no_checkpoint_selection=True,no_student_visit_aggregation_this_stage=True)
 write(out,'training.json',result);return result
