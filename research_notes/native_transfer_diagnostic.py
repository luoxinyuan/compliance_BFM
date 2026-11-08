"""Frozen-main BC diagnosis + prebudgeted single-training-episode overfit probe.
No final conditions, no checkpoint reselection, no PPO, no old actor initialization.
"""
import argparse,json,gzip,os,sys,hashlib,copy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
S=ROOT.parent
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def study(e,m,contract,out):
 import torch,numpy as np
 import native_env
 from native_env import Task,install
 from clean_policy import Expert,direct_command
 from locomotion_rollout import spherical
 from rolling_backend import install as backend
 from m11_runtime_speed import install as speed
 from native_transfer import phase_report
 torch.set_num_threads(1);torch.set_grad_enabled(False);install();write(out/'speed.json',speed(e));write(out/'backend.json',backend(e,m,S/'compatibility-prefix'))
 native_env.RUN=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/native-contact-20261106')
 task=Task(e,m,contract,5001);dev=e.standing.device
 data=torch.load(S/'runs/data-prepare-01/dataset.pt',map_location='cpu',weights_only=True)
 rows=[r for r in data['rows'] if r['valid'] and r['split']=='train'];assert len(rows)==311
 nets={};identities={}
 for role in ['P','S']:
  p=S/'runs/bc-fit-01'/(role+'-4000.pt');ck=torch.load(p,map_location=dev,weights_only=True);n=Expert(ck['input_dim'],'direct').to(dev);n.load_state_dict(ck['state_dict']);n.eval().requires_grad_(False);nets[role]=n;identities[role]=sha(p)
 labelmaps={}
 for row in data['manifest']:
  if row['split']!='train':continue
  qs=json.loads((Path(row['source']).parent/'searches.json').read_text())
  for q in qs:
   if q['label']!=row['episode']:continue
   c=q['candidates'][q['selected']];kind=c['kind']+(':'+c['parameters'].get('base','motion') if c['kind']=='sequence' else '')
   labelmaps[(row['episode'],round(q['time']*10))]=kind
 alignment=[]
 for role,net in nets.items():
  for row in rows:
   xx=row['privileged' if role=='P' else 'restricted'].to(dev)[None];zs=direct_command(net.distribution(xx).mean,e.standing);zt=row['z'].to(dev)[None];prev=row['previous_z'].to(dev)[None];ds=[]
   for k,c in enumerate(row['control']):
    ob={key:torch.tensor(c['bfm_obs'][key],device=dev,dtype=torch.float32) for key in e.keys}
    qt=task.qof(m.act(ob,spherical(prev,zt,(k+1)/5),mean=True));qs=task.qof(m.act(ob,spherical(prev,zs,(k+1)/5),mean=True));ds.append(float(torch.rad2deg(qs-qt).square().mean().sqrt()))
   alignment.append(dict(role=role,episode=row['episode'],time=row['time'],source_kind=labelmaps[(row['episode'],round(row['time']*10))],z_angle_deg=float(torch.rad2deg(torch.acos(((zs*zt).sum()/256).clamp(-1,1)))),qpd_mean_deg=float(np.mean(ds)),qpd_max_deg=max(ds)))
 write(out/'main4000-train-alignment.json',alignment);write(out/'main-identities.json',identities)
 trainS=torch.stack([r['restricted'] for r in rows]).to(dev);ns=nets['S'];trainSn=ns.normalized(trainS);summary=[]
 def run(label,s,net,role):
  task.reset(s,record=True,label=label);queries=[]
  while not task.done:
   x=task.features() if role=='P' else task.student_features();raw=net.distribution(x).mean;z=direct_command(raw,e.standing)
   q=dict(time=float(e.tether.time),input=x[0].cpu(),actual_previous_z=task.last_command[0].cpu(),command_z=z[0].cpu())
   if role=='S':q['nearest_main_train_normalized_RMS']=float(torch.cdist(ns.normalized(x),trainSn).min()/trainS.shape[-1]**.5)
   queries.append(q);task.step(z/16)
  row=task.save(out);row.update(phase_report(task),role=role);summary.append(row);write(out/'summaries.json',summary);torch.save(queries,out/(label+'-actual-queries.pt'));print('FROZEN_DIAGNOSTIC',label,row['all']['error_cm'],row['all']['force_rms'],row['boundary'],flush=True)
 specs={r['episode']:r['condition'] for r in data['manifest'] if r['split']=='train'}
 for name in ['rolling_continue','rolling_stop','short08']:run('S4000_'+name,specs[name],ns,'S')
 no=copy.deepcopy(specs['short08']);no.update(enabled=False,distance=0.,ramp=4.,release=11.2,end=15.2);run('S4000_no_actual_contact',no,ns,'S')
 # Exploratory single-episode diagnostic only. Does not revise main selection/final.
 single=[r for r in rows if r['episode']=='rolling_continue'];target=torch.stack([r['z'] for r in single]).to(dev);rng=np.random.default_rng(5102);order=[rng.integers(len(single),size=128) for _ in range(1000)]
 for role,key in [('P','privileged'),('S','restricted')]:
  torch.manual_seed(5102);x=torch.stack([r[key] for r in single]).to(dev);net=Expert(x.shape[-1],'direct').to(dev);net.center.copy_(x.mean(0));net.scale.copy_(x.std(0,unbiased=False).clamp_min(.05));net.actor[-1].bias.data.copy_(e.standing[0]/16);net.requires_grad_(False);net.actor.requires_grad_(True);opt=torch.optim.Adam(net.actor.parameters(),lr=3e-4);curve=[]
  with torch.enable_grad():
   for step,ix in enumerate(order,1):
    ix=torch.tensor(ix,device=dev);zz=direct_command(net.distribution(x[ix]).mean,e.standing);loss=(zz-target[ix]).square().mean();opt.zero_grad();loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(net.actor.parameters(),1));opt.step()
    if step%100==0:
     with torch.no_grad():zz=direct_command(net.distribution(x).mean,e.standing);angle=torch.rad2deg(torch.acos(((zz*target).sum(-1)/256).clamp(-1,1)))
     curve.append(dict(step=step,loss=float(loss.detach()),angle_mean=float(angle.mean()),angle_max=float(angle.max()),gradient=grad))
  net.eval().requires_grad_(False);p=out/(role+'-single-1000.pt');torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},input_dim=x.shape[-1],role=role,seed=5102,steps=1000,runtime=RUNTIME,scope='exploratory single-training-episode only, not new dev-selected baseline'),p);write(out/(role+'-single-training.json'),dict(samples=len(single),steps=1000,curve=curve,sha256=sha(p),old_actor_inherited=False,trained_on_final=False))
  run(role+'single_train_only',specs['rolling_continue'],net,role)
 return dict(status='completed',new_BC_updates=2000,formal_main_selection_unchanged=True,new_final_evaluations=0,Student_PPO=0,episodes=len(summary))
def main():
 global RUNTIME
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 import subprocess
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 RUNTIME=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();write(a.output/'launch.json',dict(runtime=RUNTIME,source_sha256=sha(Path(__file__)),scope='frozen train-domain diagnosis and protocol-reserved1000-step single-episode probes, no final-based selection'))
 os.environ['BFM_RECOVERY_OUTPUT']=str(a.output);os.chdir(a.output)
 import m10_physics,m10_stage
 m10_physics.OUT=a.output;m10_physics.MODEL=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view');m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
