"""Same-I, equal-data-count/equal-update initialization comparison, no RL here."""
import json,gzip,time,os,subprocess
from pathlib import Path
import transfer_student_validated as runner
RUN=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/latent-prior-audit-20261105')
def read(p):return json.load(gzip.open(p,'rt'))
def study(env,model,contract,out):
 import numpy as np,torch
 from clean_policy import Expert,direct_command
 from geometry_policy import expand
 from transition_env import Task,PLAN,install_frame
 from m11_runtime_speed import install
 from locomotion_rollout import spherical
 install_frame();import geometry_env,reach_contact;geometry_env.install_contact=reach_contact.install
 runner.write(out/'runtime-speed.json',install(env));torch.set_num_threads(1);torch.set_grad_enabled(False)
 cfg=json.loads(Path(__file__).with_name('prior_protocol.json').read_text());decision=json.loads((RUN/'qualification-decision.json').read_text());assert decision['proceed'];cap=decision['cap'];kinds=decision['kinds'];arm=os.environ['PRIOR_ARM'];src=RUN/'qualify-02';task=Task(env,model,contract,4301);device=env.standing.device
 ident=cfg['models']['I'];assert runner.sha(ident['path'])==ident['sha256'];ck=torch.load(ident['path'],map_location=device,weights_only=True)
 def load():
  n=Expert(1088,'direct').to(device);expand(n,ck);return n.eval().requires_grad_(False)
 net=load();initial=load();data=[];groups=[];manifest=[]
 for kind in kinds:
  paths=[src/f'{cap}_I_{kind}-labels.json.gz', (src/f'{cap}_I_{kind}-labels.json.gz' if kind in ['zero','walk'] else RUN/'source2'/f'{cap}_source_{kind}-labels.json.gz')];rr=[read(p) for p in paths]
  # Source walking is I itself; exclude its unrequested pre-contact actions from
  # diverse targets, rather than labeling an identical standing history twice.
  # Both corpora now share actual I precontact histories; no prefix label exclusion.
  n=min(map(len,rr));rows=rr[0 if arm=='A' else 1];ix=np.linspace(0,len(rows)-1,n).round().astype(int);selected=[rows[i] for i in ix];groups.append(torch.arange(len(data),len(data)+n,device=device));data+=selected
  manifest.append(dict(kind=kind,samples=n,source=str(paths[0 if arm=='A' else 1]),sha256=runner.sha(paths[0 if arm=='A' else 1])))
 x=torch.tensor([r['x'] for r in data],device=device);z=torch.tensor([r['z'] for r in data],device=device);targetnorm=initial.distribution(x).mean.norm(dim=-1).detach()
 # Exact repeated deployable/privileged inputs must not carry incompatible z.
 exact={};conflicts=[]
 for i,r in enumerate(data):
  key=np.asarray(r['x'],dtype=np.float32).tobytes()
  if key in exact and np.max(abs(np.array(r['z'])-data[exact[key]]['z']))>1e-4:conflicts.append([exact[key],i])
  exact[key]=i
 assert not conflicts,conflicts[:5]
 torch.manual_seed(4301);net.actor.requires_grad_(True);opt=torch.optim.Adam(net.actor.parameters(),lr=1e-5);curve=[]
 with torch.enable_grad():
  for step in range(1500):
   g=groups[step%len(groups)];ix=g[torch.randint(len(g),(128,),device=device)];raw=net.distribution(x[ix]).mean;zz=direct_command(raw,env.standing);lz=(zz-z[ix]).square().mean();ln=(raw.norm(dim=-1)-targetnorm[ix]).square().mean();loss=lz+.05*ln;opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(net.actor.parameters(),1));opt.step()
   if step%100==0 or step==1499:curve.append(dict(update=step+1,z_loss=float(lz),raw_norm_loss=float(ln),gradient=gn));runner.write(out/'curve.json',curve)
 net.eval().requires_grad_(False);runtime=subprocess.check_output(['git','-C',str(runner.ROOT),'rev-parse','HEAD'],text=True).strip();path=out/(arm+'.pt');torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},input_dim=1088,kind='direct',updates=1500,initial_sha=ident['sha256'],seed=4301,runtime=runtime,arm=arm),path)
 checks=[];gen=torch.Generator(device=device).manual_seed(4303)
 for i in np.linspace(0,len(data)-1,min(128,len(data))).round().astype(int):
  r=data[i];xx=x[i:i+1];raw=net.distribution(xx).mean;zz=direct_command(raw,env.standing);p=torch.tensor(r['previous_z'],device=device)[None];sq=[]
  for j,c in enumerate(r['sequence']):
   ob={k:torch.tensor(v,device=device) for k,v in c['bfm_obs'].items()};q=task.qof(model.act(ob,spherical(p,zz,(j+1)/5),mean=True));sq.append(q[0]-torch.tensor(c['q_PD'],device=device))
  noisy=direct_command(raw+net.log_std.exp()*torch.randn((32,256),device=device,generator=gen),env.standing);angles=torch.rad2deg(torch.acos(((noisy*zz).sum(-1)/256).clamp(-1,1)))
  checks.append(dict(kind=r['kind'],z_angle_deg=float(torch.rad2deg(torch.acos((zz*z[i:i+1]).sum().div(256).clamp(-1,1)))),qpd_rms_deg=float(torch.rad2deg(torch.stack(sq)).square().mean().sqrt()),raw_norm=float(raw.norm()),sample_angle_mean_deg=float(angles.mean())))
 runner.write(out/'initialization.json',dict(arm=arm,updates=1500,samples=len(data),manifest=manifest,exact_input_conflicts=conflicts,checkpoint=str(path),sha256=runner.sha(path),runtime=runtime,checks=checks,normalization_and_critic='unchanged I, only actor trained',A_scope='old I self-supervision control; same data count and updates, not same nonzero gradient work'))
 # Actual physical development after the fixed endpoint; no checkpoint picking.
 summaries=[]
 specs={}
 for name,dist,ramp in [('zero',0.,7.8),('arm5',.05,3.),('arm15',.15,3.),('middle30',.30,7.8),('walk',.80,7.8)]:
  s=dict(PLAN['dev']['stand_walk'],distance=dist,ramp=ramp,onset=4.25,release=4.25+ramp+2,end=4.25+ramp+6,anchor_mode='onset_actual',total_cap=cap);specs[name]=s
 specs['no_contact']=dict(specs['zero'],enabled=False)
 for name,angle in [('left',15.),('right',-15.)]:specs[name]=dict(specs['zero'],geometry='torso_pair',rotation_deg=angle,anchor_mode='legacy')
 for name,s in specs.items():
  task.reset(s,record=True,label=arm+'_'+name)
  while not task.done:task.step(net.distribution(task.features()).mean)
  r=task.save(out);summaries.append(r);runner.write(out/'summaries.json',summaries);print('PRIOR_INIT',arm,name,r['error_cm'],r['force_rms'],r['actual_torso_turn_deg'],r['boundary'],flush=True)
 return dict(status='completed',rl_decisions=0,supervised_updates=1500,episodes=len(summaries),elapsed_s=time.monotonic()-runner.START)
runner.study=study
if __name__=='__main__':runner.main()
