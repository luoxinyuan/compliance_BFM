"""Authorized clean-label continuation of Issue53; no archived artifact mutation."""
import argparse,copy,json,os,sys,time,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
S=ROOT.parent;P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');PLAN=json.loads(Path(__file__).with_name('interactive_increment_protocol.json').read_text());OLD=Path(PLAN['old_stage'])
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def baseline():
 p=OLD/PLAN['baseline']['path'];assert sha(p)==PLAN['baseline']['sha256'];return p

def fit(out):
 import torch,numpy as np
 from clean_policy import Expert,direct_command
 torch.set_num_threads(1);torch.manual_seed(5301);dev='cuda';cfg=PLAN['bc']
 d=torch.load(P/'experiments/native-transfer-20261107/runs/data-prepare-01/dataset.pt',map_location='cpu',weights_only=True);old=[r for r in d['rows'] if r['valid'] and r['split']=='train'];assert len(old)==311
 fresh=OLD/'runs/final-analysis-01/corrected-new-labels.pt';assert sha(fresh)=='3062cf5c02a6f5b37d19c38e9f0fa285649378e007a3bb2ddac5ce303316b657'
 f=torch.load(fresh,map_location='cpu',weights_only=True);new=f['rows'];assert len(new)==34 and all(r['valid'] for r in new);assert len({r['state_fingerprint'] for r in new})==34
 start=P/'experiments/native-transfer-20261107/runs/bc-fit-01/S-4000.pt';assert sha(start)=='8f5e7287e974deedba1a8d841dd65d7692e944e278bd4a091f1ec224f167a479';baseline()
 x=torch.stack([r['restricted'] for r in old+new]).to(dev);z=torch.stack([r['z'] for r in old+new]).to(dev);assert x.shape==(345,1639) and torch.isfinite(x).all() and torch.isfinite(z).all();standing=d['standing'].to(dev)[None]
 # Preserve the EXACT original old-data sample orders, replacing only the prescribed45 slots.
 orders=np.load(OLD/'runs/bc-fit-01/matched-orders.npz')['old_data_only'].copy();assert orders.shape==(1000,128)
 groups=[np.array([311+j for j,r in enumerate(new) if r['episode'].startswith(prefix)]) for prefix in ['shadow_normal72','Teacher_replan_2s_normal72']]
 # Group by physical recovery window, not every correlated frame as a separate episode.
 groups=[np.array([311+j for j,r in enumerate(new) if r['time']<5.3 or 'normal72' in r['episode']]),np.array([311+j for j,r in enumerate(new) if 'normal72' not in r['episode']])];assert sorted(map(len,groups))==[14,20]
 rng=np.random.default_rng(5301);mixed=orders.copy();k=round(128*.35)
 for row in mixed:row[:k]=rng.choice(groups[int(rng.integers(2))],k,replace=True)
 ck=torch.load(start,map_location=dev,weights_only=True);net=Expert(1639,'direct').to(dev);net.load_state_dict(ck['state_dict']);net.requires_grad_(False);net.actor.requires_grad_(True);opt=torch.optim.Adam(net.actor.parameters(),lr=1e-4);curve=[];beg=time.monotonic()
 def save(step):
  p=out/f'interactive-{step}.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},input_dim=1639,kind='direct',role='clean-label ideal-force interactive Student',updates=step,seed=5301,runtime=RUNTIME,start_sha256=sha(start),data_sha256=sha(fresh),normalizer='S4000 unchanged',actor_only_BC=True,no_privileged_actor=True),p)
 save(0)
 for step,ix in enumerate(mixed,1):
  with torch.enable_grad():
   ix=torch.tensor(ix,device=dev);pred=direct_command(net.distribution(x[ix]).mean,standing);loss=(pred-z[ix]).square().mean();opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(net.actor.parameters(),1));opt.step()
  if step%100==0:
   curve.append(dict(step=step,loss=float(loss.detach()),gradient=gn,seconds=time.monotonic()-beg));write(out/'curve.json',curve);print('CLEAN_INTERACTIVE_BC',step,float(loss.detach()),flush=True)
  if step in [500,1000]:save(step)
 assert torch.equal(net.center,ck['state_dict']['center']) and torch.equal(net.scale,ck['state_dict']['scale'])
 np.savez_compressed(out/'sample-orders.npz',old_reference=orders,interactive=mixed)
 return dict(status='completed',supervised_updates=1000,RL_decisions=0,full_queries=0,old_labels=311,new_labels=34,independent_new_source_windows=2,window_sizes=list(map(len,groups)),batch=128,new_samples_per_batch=k,actor_parameters=sum(p.numel() for p in net.actor.parameters()),frozen_normalizer=True,source_sha256=sha(start),data_sha256=sha(fresh),baseline_sha256=sha(baseline()),seconds=time.monotonic()-beg)

def study(e,m,contract,out):
 import torch,numpy as np
 from native_env import Task,spec,install
 from clean_policy import Expert
 from native_transfer import phase_report
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 install();torch.set_num_threads(1);torch.set_grad_enabled(False);write(out/'speed.json',speed(e));write(out/'backend.json',backend(e,m,P/'experiments/native-rolling-h100-20261107/compatibility-prefix'))
 task=Task(e,m,contract,5301);rows=[];phys=reset=0
 plan=json.loads((ROOT/'research_notes/interactive_protocol.json').read_text());dev={'short08':spec(.08,1.6,5.2),'normal72':spec(.72,4.8,5.),'stop66':spec(.66,4.4,5.4,stop=8.1)};reg={'mid24':spec(.24,3,5.3),'none':spec(0,3,5.2,enabled=False)};final={}
 for name,c in plan['sealed_cases'].items():
  ss=spec(c['distance'],c['ramp'],c['onset'])
  if 'stop_time'in c:ss.update(stop_time=c['stop_time'],deceleration=c['deceleration'],release=c['stop_time']+c['deceleration']+2,end=c['stop_time']+c['deceleration']+6)
  final[name]=ss
 def run(role,path,name,ss):
  nonlocal phys,reset
  ck=torch.load(path,map_location='cuda',weights_only=True);n=Expert(1639,'direct').cuda();n.load_state_dict(ck['state_dict']);n.eval().requires_grad_(False);task.reset(ss,record=True,label=role+'_'+name);reset+=600+len(task.prefix_data['physics'])
  while not task.done:
   _,_,_,st=task.step(n.distribution(task.student_features()).mean);phys+=st['physics_steps']
  r=copy.deepcopy(task.save(out));r.update(phase_report(task));r.update(role=role,case=name,model=dict(path=str(path),sha256=sha(path)),permission='restricted1639 ideal-force only',exposure='original53 heldout reused for requested fixed evaluation' if name in final else 'development/regression');rows.append(r);write(out/'summaries.json',rows);print('INCREMENT_REAL',role,name,r['all']['error_cm'],r['all']['force_rms'],r['boundary'],flush=True);return r
 def cost(r):a=r['all'];return a['force_rms']+.03*(a['error_cm'] or 0)+a['cap_fraction']
 def score(rs):cs=[cost(r) for r in rs];return float(np.mean(cs)+.1*max(cs))
 def severe(r,b):
  a,c=r['all'],b['all'];reasons=[]
  if r['boundary']:reasons.append('physics boundary')
  if a['error_cm'] is not None and c['error_cm'] is not None and a['error_cm']>c['error_cm']+max(5,.5*c['error_cm']):reasons.append('large error regression')
  if a['force_rms']>c['force_rms']+2:reasons.append('force +2N')
  if r['case'] in ['short08','near06','none'] and r['root_max_displacement']>b['root_max_displacement']+.10:reasons.append('small/none root +10cm')
  return reasons
 if ARGS.mode=='eval':
  base={name:run('BC500',baseline(),name,ss) for name,ss in dev.items()};candidates=[]
  for step in [500,1000]:
   path=S/f'runs/fit-01/interactive-{step}.pt';rr=[run(f'interactive{step}',path,name,ss) for name,ss in dev.items()];bad=[dict(case=r['case'],reasons=severe(r,base[r['case']])) for r in rr if severe(r,base[r['case']])];candidates.append(dict(step=step,model=rr[0]['model'],score=score(rr),severe=bad,rows=rr))
  eligible=[c for c in candidates if not c['severe']];chosen=min(eligible or candidates,key=lambda c:c['score']);write(out/'selection-before-final.json',dict(selected=chosen,candidates=candidates,baseline_score=score(list(base.values())),rule=PLAN['selection'],checkpoint_frozen_before_final=True))
  selected=Path(chosen['model']['path']);allbase={};allnew={}
  for name,ss in {**reg,**final}.items():
   allbase[name]=run('BC500',baseline(),name,ss);allnew[name]=run('Sinteractive',selected,name,ss)
  primary=list(final);bad=[dict(case=k,reasons=severe(allnew[k],allbase[k])) for k in primary if severe(allnew[k],allbase[k])];gains=[]
  for k in primary:
   a,b=allnew[k]['all'],allbase[k]['all'];gains.append((a['error_cm']<=.95*b['error_cm'] and a['force_rms']<=1.05*b['force_rms'])or(a['force_rms']<=.95*b['force_rms'] and a['error_cm']<=1.05*b['error_cm']))
  noninferior=not bad and all(cost(allnew[k])<=cost(allbase[k])+.25 for k in primary)
  gate=dict(status='evaluated',selected_interactive=chosen['model'],baseline=dict(path=str(baseline()),sha256=sha(baseline())),serious_regressions=bad,meaningful_cases=gains,aggregate_score_new=score([allnew[k] for k in primary]),aggregate_score_baseline=score([allbase[k] for k in primary]),useful=not bad and sum(gains)>=2 and score([allnew[k] for k in primary])<=.95*score([allbase[k] for k in primary]),interactive_noninferior=noninferior,PPO_start=chosen['model'] if noninferior else dict(path=str(baseline()),sha256=sha(baseline())),PPO_allowed=not any(r['boundary'] for r in allbase.values()),checkpoint_selection_after_final=False,physics_steps=phys,reset_steps=reset,RL_decisions=0)
  write(out/'route-gate.json',gate);return gate
 else:
  paths={'frozen_PPOstart':Path(json.loads((S/'runs/eval-01/route-gate.json').read_text())['PPO_start']['path'])}
  for seed in [5401,5402]:paths[f'RL{seed}_batch8']=S/f'runs/ppo-{seed}/candidate-8.pt'
  for role,path in paths.items():
   for name,ss in {**reg,**final}.items():run(role,path,name,ss)
  return dict(status='completed',physics_steps=phys,reset_steps=reset,episodes=len(rows),checkpoint_selection_after_final=False,RL_decisions=0)

def main():
 global ARGS,RUNTIME
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fit','eval','final'],required=True);p.add_argument('--output',type=Path,required=True);ARGS=p.parse_args();ARGS.output.mkdir(exist_ok=False)
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain']);RUNTIME=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();write(ARGS.output/'launch.json',dict(runtime=RUNTIME,source_sha256=sha(__file__),mode=ARGS.mode))
 if ARGS.mode=='fit':write(ARGS.output/'result.json',fit(ARGS.output));return
 os.environ['BFM_RECOVERY_OUTPUT']=str(ARGS.output);os.chdir(ARGS.output)
 import m10_stage,m10_physics
 m10_physics.OUT=ARGS.output;m10_physics.MODEL=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view';m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
