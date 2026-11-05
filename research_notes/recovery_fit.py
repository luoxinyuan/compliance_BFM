"""One matched C0/C1 comparison. Effective recovery data + bounded function retention.
Recorded five-substep observations are fixed data, not differentiable physics.
"""
import json,gzip,time,os,subprocess,copy
from pathlib import Path
import transfer_student_validated as runner
OLD=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/latent-prior-audit-20261105')
RUN=OLD.parent/'prior-recovery-20261105'
def read(p):return json.load(gzip.open(p,'rt'))
def study(env,model,contract,out):
 import torch,numpy as np
 from clean_policy import Expert,direct_command
 from geometry_policy import expand
 from transition_env import Task,install_frame
 from m11_runtime_speed import install
 from m39b_replay import interp,grad
 install_frame();import geometry_env,reach_contact;geometry_env.install_contact=reach_contact.install
 runner.write(out/'runtime-speed.json',install(env));torch.set_num_threads(1);torch.set_grad_enabled(False);model.eval().requires_grad_(False)
 cfg=json.loads(Path(__file__).with_name('recovery_protocol.json').read_text());decision=json.loads((RUN/'recovery-decision.json').read_text());assert decision['proceed'];arm=os.environ['RECOVERY_ARM'];assert arm in ['C0','C1'];device=env.standing.device;task=Task(env,model,contract,4301)
 ident=cfg['models']['I'];assert runner.sha(ident['path'])==ident['sha256'];ck=torch.load(ident['path'],map_location=device,weights_only=True)
 def load(ck):
  n=Expert(1088,'direct').to(device);expand(n,ck);return n.eval().requires_grad_(False)
 net=load(ck);initial=load(ck);b=load(torch.load(cfg['models']['B']['path'],map_location=device,weights_only=True));rows=[];groups=[];kinds=[];manifest=[]
 for m in json.loads((OLD/'fit2-B/initialization.json').read_text())['manifest']:
  rr=read(m['source']);ix=np.linspace(0,len(rr)-1,m['samples']).round().astype(int);selected=[rr[i] for i in ix];groups.append(torch.arange(len(rows),len(rows)+len(selected),device=device));kinds.append(m['kind']);rows+=selected;manifest.append(m)
 original=list(rows);recovery=[]
 for p in decision['accepted_label_files']:recovery+=read(p)
 assert recovery
 walk=groups[kinds.index('walk')];replace=walk[::2];ix=np.linspace(0,len(recovery)-1,len(replace)).round().astype(int)
 if arm=='C1':
  for j,i in zip(replace.cpu().tolist(),ix):rows[j]=recovery[i]
 x=torch.tensor([r['x'] for r in rows],device=device);z=torch.tensor([r['z'] for r in rows],device=device);targetnorm=initial.distribution(x).mean.norm(dim=-1).detach()
 # Retention only measured reliable I startup and genuinely recovered I suffixes.
 reliable=[r for r in original if r['kind']=='walk' and 4.-1e-7<=r['time']<=7.5]+recovery
 rx=torch.tensor([r['x'] for r in reliable],device=device);rz=torch.tensor([r['z'] for r in reliable],device=device);prev=torch.tensor([r['previous_z'] for r in reliable],device=device)
 assert (rx[:,565:821]*16-prev).abs().max()<1e-5
 qt=torch.tensor([[c['q_PD'] for c in r['sequence']] for r in reliable],device=device);at=torch.tensor([[c['action'] for c in r['sequence']] for r in reliable],device=device)
 ob={k:torch.tensor([[c['bfm_obs'][k][0] for c in r['sequence']] for r in reliable],device=device) for k in contract['bfm_inputs']}
 normalized=model._normalize({k:v.reshape(-1,*v.shape[2:]) for k,v in ob.items()});ob={k:v.detach().reshape(len(reliable),5,*v.shape[1:]) for k,v in normalized.items()}
 scale=torch.tensor(contract['q_PD_scale'],device=device);offset=torch.tensor(contract['q_PD_offset'],device=device)
 def action(z,ix):
  zz=interp(prev[ix],z);o={k:v[ix].reshape(-1,*v.shape[2:]) for k,v in ob.items()};a=model._actor(o,zz.reshape(-1,256),model.cfg.actor_std).mean.float();q=offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5;return a.reshape(-1,5,29),q.reshape(-1,5,29)
 ixall=torch.arange(len(reliable),device=device);identity=[]
 for ii in ixall.split(32):
  aa,qq=action(rz[ii],ii);identity.append(dict(action_max=float((aa-at[ii]).abs().max()),qpd_max_rad=float((qq-qt[ii]).abs().max())))
 assert max(r['qpd_max_rad'] for r in identity)<1e-4,identity
 # Fixed calibration at existing B, where old-function error is nonzero. No update.
 b.actor.requires_grad_(True);cal=[];g=torch.Generator(device=device).manual_seed(4401)
 with torch.enable_grad():
  for kind,ii in zip(kinds,groups):
   jj=ii[torch.randint(len(ii),(64,),device=device,generator=g)];rr=ixall[torch.randint(len(ixall),(32,),device=device,generator=g)]
   raw=b.distribution(x[jj]).mean;ls=(direct_command(raw,env.standing)-z[jj]).square().mean()+.05*(raw.norm(dim=-1)-targetnorm[jj]).square().mean();gs=grad(ls,b)
   _,qq=action(direct_command(b.distribution(rx[rr]).mean,env.standing),rr);la=(qq-qt[rr]).square().mean();ga=grad(la,b)
   assert ga.norm()>1e-10 and torch.isfinite(ga).all();cal.append(dict(kind=kind,supervised_gradient=float(gs.norm()),retention_gradient=float(ga.norm()),retention_qpd_rad2=float(la)))
 b.actor.requires_grad_(False)
 coefficient=float(np.clip(.25*np.median([r['supervised_gradient'] for r in cal])/np.median([r['retention_gradient'] for r in cal]),1e-3,1e4)) if arm=='C1' else 0.
 runner.write(out/'calibration.json',dict(coefficient=coefficient,target_gradient_ratio=.25,batches=cal,replay_identity=identity,scope='same-state five measured BFM substeps, frozen physics; gradient through frozen actor to high policy',records=len(reliable)))
 # Fresh optimizer, exact C0 seed/order. Extra retention sampling uses separate RNG.
 torch.manual_seed(4301);net.actor.requires_grad_(True);opt=torch.optim.Adam(net.actor.parameters(),lr=1e-5);curve=[];prevparam=torch.cat([p.detach().reshape(-1) for p in net.actor.parameters()]).clone();trainstart=time.monotonic()
 with torch.enable_grad():
  for step in range(1500):
   if time.monotonic()-runner.START>runner.ARGS.max_seconds-220:raise TimeoutError('reserved physical development time')
   gi=step%len(groups);ii=groups[gi];ix=ii[torch.randint(len(ii),(128,),device=device)];raw=net.distribution(x[ix]).mean;zz=direct_command(raw,env.standing);lz=(zz-z[ix]).square().mean();ln=(raw.norm(dim=-1)-targetnorm[ix]).square().mean();loss=lz+.05*ln;ret=torch.zeros((),device=device)
   if arm=='C1':
    rr=ixall[torch.randint(len(ixall),(16,),device=device,generator=g)];_,qq=action(direct_command(net.distribution(rx[rr]).mean,env.standing),rr);ret=(qq-qt[rr]).square().mean();loss=loss+coefficient*ret
   opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(net.actor.parameters(),1));opt.step()
   if step%100==0 or step==1499:
    param=torch.cat([p.detach().reshape(-1) for p in net.actor.parameters()]);delta=float((param-prevparam).norm());prevparam=param.clone();row=dict(update=step+1,kind=kinds[gi],z_loss=float(lz),norm_loss=float(ln),retention_qpd_rad2=float(ret),weighted_retention=float(coefficient*ret),gradient_preclip=gn,param_change_since_log=delta,elapsed_s=time.monotonic()-trainstart);curve.append(row);runner.write(out/'curve.json',curve);print('RECOVERY_FIT',arm,row,flush=True)
 net.eval().requires_grad_(False);runtime=subprocess.check_output(['git','-C',str(runner.ROOT),'rev-parse','HEAD'],text=True).strip();path=out/(arm+'.pt');torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},input_dim=1088,kind='direct',updates=1500,initial_sha=ident['sha256'],seed=4301,runtime=runtime,arm=arm),path)
 groupcheck=[]
 for kind,ii in zip(kinds,groups):
  raw=net.distribution(x[ii]).mean;zz=direct_command(raw,env.standing);groupcheck.append(dict(kind=kind,samples=len(ii),z_mse=float((zz-z[ii]).square().mean()),mean_z_angle_deg=float(torch.rad2deg(torch.acos((zz*z[ii]).sum(-1).div(256).clamp(-1,1))).mean()),raw_norm=float(raw.norm(dim=-1).mean())))
 _,qq=action(direct_command(net.distribution(rx).mean,env.standing),ixall)
 ckB=torch.load(cfg['models']['B']['path'],map_location='cpu',weights_only=True);same=max(float((v.cpu()-ckB['state_dict'][k]).abs().max()) for k,v in net.state_dict().items() if k.startswith('actor.'))
 runner.write(out/'training.json',dict(arm=arm,updates=1500,samples=len(rows),manifest=manifest,unique_recovery_records=len(recovery),replaced_walk_slots=len(replace) if arm=='C1' else 0,retention_records=len(reliable),retention_weight=coefficient,group_checks=groupcheck,retention_qpd_rms_deg=float(torch.rad2deg(qq-qt).square().mean().sqrt()),C0_actor_max_difference_from_old_B=same if arm=='C0' else None,checkpoint=str(path),sha256=runner.sha(path),runtime=runtime,seed=4301,init_sha=ident['sha256'],scope='C1 bundles valid recovery coverage and bounded action retention; not isolating either contribution'))
 # Fixed endpoint only, no checkpoint selection by physical validation.
 specs=json.loads(Path(__file__).with_name('prior_training.json').read_text())['dev'];specs=copy.deepcopy(specs);specs['small8']=cfg['specs']['small'];
 for name,d,v in [('middle20',.20,.09),('long55',.55,.13)]:
  ramp=1.5*d/v;specs[name]=dict(specs['walk'],distance=d,onset=4.55,ramp=ramp,release=4.55+ramp+2,end=4.55+ramp+6)
 summ=[]
 for name,s in specs.items():
  task.reset(s,record=True,label=arm+'__'+name)
  while not task.done:task.step(net.distribution(task.features()).mean)
  r=task.save(out);summ.append(r);runner.write(out/'summaries.json',summ);print('RECOVERY_DEV',arm,name,r['error_cm'],r['force_rms'],r['boundary'],flush=True)
 return dict(status='completed',supervised_updates=1500,rl_decisions=0,episodes=len(summ),elapsed_s=time.monotonic()-runner.START)
runner.study=study
if __name__=='__main__':runner.main()
