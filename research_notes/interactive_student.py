"""Matched continued BC vs verified interactive BC; restricted-only real execution."""
import argparse,copy,json,os,sys,time,hashlib,gzip
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
S=ROOT.parent;P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');OLD=P/'experiments/native-transfer-20261107'
PLAN=json.loads((ROOT/'research_notes/interactive_protocol.json').read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fit(out):
 import torch,numpy as np
 from clean_policy import Expert,direct_command
 torch.set_num_threads(1);dev='cuda';cfg=PLAN['bc'];start=P/PLAN['source_student']['checkpoint'];assert sha(start)==PLAN['source_student']['sha256']
 diag=S/'runs/diagnose-01';assert json.loads((diag/'gate.json').read_text())['BC_allowed']
 d=torch.load(OLD/'runs/data-prepare-01/dataset.pt',map_location='cpu',weights_only=True);old=[r for r in d['rows'] if r['valid'] and r['split']=='train']
 fresh=torch.load(diag/'interactive-labels.pt',map_location='cpu',weights_only=True);valid=[r for r in fresh['rows'] if r['valid']];unique={}
 for r in valid:
  key=r['state_fingerprint']
  if key in unique:
   assert torch.equal(unique[key]['z'],r['z']) and torch.equal(unique[key]['restricted'],r['restricted']);unique[key].setdefault('additional_roles',[]).append(r['source'])
  else:unique[key]=r
 new=list(unique.values());assert new
 allrows=old+new;x=torch.stack([r['restricted'] for r in allrows]).to(dev);z=torch.stack([r['z'] for r in allrows]).to(dev);standing=d['standing'].to(dev)[None]
 oldgroups=[np.array([j for j,r in enumerate(old) if r['episode']==e]) for e in sorted(set(r['episode'] for r in old))]
 def region(r):return r['episode'].replace('shadow_','').replace('Teacher_replan_2s_','')
 newgroups=[np.array([len(old)+j for j,r in enumerate(new) if region(r)==e]) for e in sorted(set(region(r) for r in new))]
 rng=np.random.default_rng(cfg['seed']);orders={a:[] for a in cfg['arms']}
 for i in range(cfg['updates_each']):
  og=oldgroups[int(rng.integers(len(oldgroups)))];ng=newgroups[int(rng.integers(len(newgroups)))];ix=rng.choice(og,size=cfg['batch'],replace=True)
  orders['old_data_only'].append(ix);mix=ix.copy();k=round(cfg['batch']*cfg['new_fraction']);mix[:k]=rng.choice(ng,size=k,replace=True);orders['interactive'].append(mix)
 ck=torch.load(start,map_location=dev,weights_only=True);log=[]
 for arm in cfg['arms']:
  torch.manual_seed(cfg['seed']);net=Expert(1639,'direct').to(dev);net.load_state_dict(ck['state_dict']);net.requires_grad_(False);net.actor.requires_grad_(True);opt=torch.optim.Adam(net.actor.parameters(),lr=cfg['lr']);curve=[];beg=time.monotonic()
  def save(j):
   p=out/(arm+'-'+str(j)+'.pt');torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},input_dim=1639,kind='direct',role='ideal-force restricted Student',updates=j,seed=cfg['seed'],runtime=RUNTIME,start_sha256=sha(start),data_sha256=sha(diag/'interactive-labels.pt'),actor_only_BC=True,normalizer='S4000 unchanged',teacher='verified native causal rolling query',not_privileged_actor=True),p);return sha(p)
  save(0)
  with torch.enable_grad():
   for j,ix in enumerate(orders[arm],1):
    ix=torch.tensor(ix,device=dev);pred=direct_command(net.distribution(x[ix]).mean,standing);loss=(pred-z[ix]).square().mean();opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(net.actor.parameters(),1));opt.step()
    if j%100==0:
     curve.append(dict(step=j,loss=float(loss.detach()),gradient=gn,seconds=time.monotonic()-beg));write(out/(arm+'-curve.json'),curve);print('STUDENT_BC',arm,j,float(loss.detach()),flush=True)
    if j in cfg['checkpoints']:save(j)
  assert torch.equal(net.center,ck['state_dict']['center']) and torch.equal(net.scale,ck['state_dict']['scale'])
  log.append(dict(arm=arm,updates=cfg['updates_each'],original_valid=len(old),new_valid=len(new),raw_valid_roles=len(valid),deduplicated_same_state_roles=len(valid)-len(new),old_independent_episodes=len(oldgroups),new_query_regions=len({(r['episode'],r['time']) for r in new}),new_trajectory_groups=len(newgroups),new_role_counts={k:sum(r['source']==k for r in new) for k in set(r['source'] for r in new)},training_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),seconds=time.monotonic()-beg,curve=curve,restricted_input_dim=1639))
 np.savez_compressed(out/'matched-orders.npz',**{k:np.stack(v) for k,v in orders.items()});write(out/'training.json',log)
 return dict(status='completed',student_optimizer_updates=2000,RL_decisions=0,models=log)
def study(e,m,contract,out):
 import torch,numpy as np
 import native_env
 from native_env import Task,spec,install
 from clean_policy import Expert,direct_command
 from native_transfer import phase_report
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 torch.set_num_threads(1);torch.set_grad_enabled(False);install();write(out/'speed.json',speed(e));write(out/'backend.json',backend(e,m,P/'experiments/native-rolling-h100-20261107/compatibility-prefix'))
 task=Task(e,m,contract,5301);dev=e.standing.device
 # Archive every predicted candidate z from frozen source parameters; never use unexecuted nodes as labels.
 from locomotion_rollout import spherical
 from m9_cyclic import sample
 rr=json.loads((P/'experiments/native-rolling-h100-20261107/runs/reward-inference-01/reward-candidates.json').read_text());rz={r['name']:torch.tensor(r['z'],device=dev,dtype=torch.float32).reshape(1,256) for r in rr['candidates']}
 with (S/'runs/diagnose-01/searches.json').open() as ff:queries=json.load(ff)
 predicted=[];selected_max=0.
 for qi,q in enumerate(queries):
  for ci,crow in enumerate(q['candidates']):
   c=crow['proposal'];zs=[]
   for j in range(24):
    if c['kind']=='standing':z=e.standing
    elif c['kind']=='goal_arm15':z=task.native_z['goal_arm15']
    elif c['kind']=='reward':z=rz[c['name']]
    else:
     alpha=float(np.interp(j/23,np.linspace(0,1,4),c['alpha']));base=rz[c['base']] if c.get('base') in rz else sample(task.refs['straight'],q['phase_start']+j*.1*c['rate']/task.refs['straight']['period']);z=spherical(spherical(e.standing,base,alpha),task.native_z['goal_arm15'],c.get('arm',0.))
    zs.append(z[0].cpu())
   if ci==q['selected']:selected_max=max(selected_max,float((zs[0]-torch.tensor(q['selected_z'])).abs().max()))
   fscore=None
   if 'shadow' in q:
    detail=q['shadow']['F'];ff=detail['all_F_scores'];extras=detail['all_available_extra'];fscore=ff[ci] if ci<7 else ff[7+next(i for i,v in enumerate(extras) if v['name']==c['name'])]
   predicted.append(dict(query_index=qi,candidate_index=ci,source=c,z_sequence=torch.stack(zs),F_score=fscore,physical=crow['physical'],physical_score=crow['score'],scope='unexecuted predictions except separately recorded true first actions; reconstructed deterministic source, not training data'))
 assert selected_max<1e-6,selected_max
 torch.save(dict(rows=predicted,selected_first_reconstruction_max=selected_max,not_labels=True),out/'candidate-z-archive.pt');write(out/'candidate-archive-audit.json',dict(candidates=len(predicted),selected_first_max_abs=selected_max,source='same frozen original references and saved candidate parameters; GPU FP32 reconstruction; no new physics/query',no_unexecuted_labels=True))
 specs={'short08':spec(.08,1.6,5.2),'normal72':spec(.72,4.8,5.),'stop66':spec(.66,4.4,5.4,stop=8.1),'mid24':spec(.24,3,5.3),'none':spec(0,3,5.2,enabled=False)}
 final={}
 for name,c in PLAN['sealed_cases'].items():
  ss=spec(c['distance'],c['ramp'],c['onset'])
  if 'stop_time' in c:ss.update(stop_time=c['stop_time'],deceleration=c['deceleration'],release=c['stop_time']+c['deceleration']+2,end=c['stop_time']+c['deceleration']+6)
  final[name]=ss
 summaries=[];f=S/'runs/bc-fit-01';nets={};identities={}
 def load(path):
  ck=torch.load(path,map_location=dev,weights_only=True);assert ck['input_dim']==1639;n=Expert(1639,'direct').to(dev);n.load_state_dict(ck['state_dict']);n.eval().requires_grad_(False);identities[id(n)]=dict(path=str(path),sha256=sha(path));return n
 def run(name,s,n,sample=False):
  task.reset(s,record=True,label=name);start=time.monotonic();rawstats=[];g=torch.Generator(device=dev).manual_seed(535301)
  while not task.done:
   x=task.student_features();assert x.shape[-1]==1639
   dist=n.distribution(x);mean=dist.mean;raw=mean+dist.stddev*torch.randn(mean.shape,device=dev,generator=g) if sample else mean
   zm=direct_command(mean,e.standing);zz=direct_command(raw,e.standing);rawstats.append(dict(norm=float(raw.norm()),noise_angle_deg=float(torch.rad2deg(torch.acos(((zm*zz).sum()/256).clamp(-1,1))))));task.step(raw)
  r=task.save(out);r.update(phase_report(task));r.update(model=identities[id(n)],sample=sample,wall_s=time.monotonic()-start,permission='restricted ideal-force Student; no privileged input',command_statistics={k:float(np.mean([x[k] for x in rawstats])) for k in rawstats[0]});summaries.append(r);write(out/'summaries.json',summaries);print('STUDENT_REAL',name,round(r['all']['error_cm'] or 0,3),round(r['all']['force_rms'],3),r['boundary'],flush=True);return r
 def score(rows):
  severe=any(r['boundary'] for r in rows);costs=[r['all']['force_rms']+.03*(r['all']['error_cm'] or 0)+r['all']['cap_fraction'] for r in rows]
  return dict(score=float(np.mean(costs)+.1*max(costs)),severe=severe,case_costs=costs)
 selection={};basenet=load(P/PLAN['source_student']['checkpoint']);base=[]
 for name in ['short08','normal72','stop66']:base.append(run('S4000_dev_'+name,specs[name],basenet))
 base_score=score(base);write(out/'baseline.json',base)
 for arm in PLAN['bc']['arms']:
  candidates=[dict(step=0,assessment=base_score,model=identities[id(basenet)],rows=base)]
  for j in [500,1000]:
   n=load(f/(arm+'-'+str(j)+'.pt'));rows=[run(arm+'_'+str(j)+'_dev_'+name,specs[name],n) for name in ['short08','normal72','stop66']];candidates.append(dict(step=j,assessment=score(rows),model=identities[id(n)],rows=rows))
  ok=[c for c in candidates if not c['assessment']['severe']];best=min(ok or candidates,key=lambda c:c['assessment']['score']);selection[arm]=dict(selected=best,candidates=candidates,rule='safety first; mean physical cost+.1worst across3 dev, no final consulted');write(out/'selection.json',selection)
 # Selection fixed. New heldout never used for reselection. Include S4000 for same-condition comparison.
 selected_nets={'S4000':basenet}
 for arm in PLAN['bc']['arms']:selected_nets[arm]=load(Path(selection[arm]['selected']['model']['path']))
 for role,n in selected_nets.items():
  for name in ['mid24','none']:run(role+'_regression_'+name,specs[name],n)
  for name,ss in final.items():run(role+'_FINAL_'+name,ss,n)
 # Same actual state, same last-z and complete five 50Hz observations. No differentiable-physics claim.
 from locomotion_rollout import spherical
 verified=torch.load(S/'runs/diagnose-01/interactive-labels.pt',map_location='cpu',weights_only=True)
 align=[];ep_cache={}
 for lab in [r for r in verified['rows'] if r['valid']][::3]:
  name=lab['episode'].replace('shadow_','Teacher_replan_2s_');path_ep=S/'runs/diagnose-01'/(name+'.json.gz')
  if not path_ep.exists():continue
  if name not in ep_cache:
   with gzip.open(path_ep,'rt') as ff:ep_cache[name]=json.load(ff)
  cs=[c for c in ep_cache[name]['control'] if lab['time']-1e-7<=c['time']<lab['time']+.1-1e-7]
  if len(cs)!=5:continue
  prev=lab['previous_z'].to(dev)[None];zt=lab['z'].to(dev)[None]
  for role,n in selected_nets.items():
   zs=direct_command(n.distribution(lab['restricted'].to(dev)[None]).mean,e.standing);diff=[];parity=[]
   for k,c in enumerate(cs):
    ob={kk:torch.tensor(v,device=dev,dtype=torch.float32) for kk,v in c['bfm_obs'].items()};qt=task.qof(m.act(ob,spherical(prev,zt,(k+1)/5),mean=True));qs=task.qof(m.act(ob,spherical(prev,zs,(k+1)/5),mean=True))
    diff.append(float(torch.rad2deg(qs-qt).square().mean().sqrt()));parity.append(float((qt-torch.tensor(c['q_PD'],device=dev)).abs().max()))
   align.append(dict(role=role,query_index=lab['query_index'],time=lab['time'],five_BFM_substep_qpd_RMSE_deg=diff,actual_teacher_recompute_max_rad=max(parity),scope='same-state fixed observations; not a multistep physical alternative rollout'))
 write(out/'same-state-alignment.json',align)
 # Mean vs sampled execution preserved independently of development selection.
 chosen=selected_nets['interactive'];run('interactive_sample_normal72',specs['normal72'],chosen,True)
 original=base;candidate=selection['interactive']['selected']['rows'];meaningful=[]
 for b,r in zip(original,candidate):
  am,bm=b['all'],r['all'];meaningful.append((bm['error_cm']<=.95*am['error_cm'] and bm['force_rms']<=am['force_rms']) or (bm['force_rms']<=.95*am['force_rms'] and bm['error_cm']<=am['error_cm']))
 stable=not any(r['boundary'] for r in summaries if r['label'].startswith('interactive'))
 gate=dict(safe=stable,meaningful_family_improvement=any(meaningful),PPO_allowed=stable and any(meaningful) and selection['interactive']['selected']['step']>0,selected=selection['interactive']['selected']['model'],evaluation='dev gate only; final does not choose checkpoint',Student_PPO_started=False)
 write(out/'ppo-gate.json',gate)
 return dict(status='completed',student_optimizer_updates=0,episodes=len(summaries),selection=selection,PPO_gate=gate)
def main():
 global RUNTIME,ARGS
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fit','eval'],required=True);p.add_argument('--output',type=Path,required=True);ARGS=p.parse_args();out=ARGS.output;out.mkdir(exist_ok=False)
 import subprocess
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain']);RUNTIME=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();write(out/'launch.json',dict(runtime=RUNTIME,source_sha256=sha(__file__),protocol_sha256=sha(ROOT/'research_notes/interactive_protocol.json'),mode=ARGS.mode))
 if ARGS.mode=='fit':write(out/'result.json',fit(out));return
 os.environ['BFM_RECOVERY_OUTPUT']=str(out);os.chdir(out)
 import m10_stage,m10_physics
 m10_physics.OUT=out;m10_physics.MODEL=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view';m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
