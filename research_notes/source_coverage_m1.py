"""M1: frozen source/contact qualification and full F/physics pairs. No training."""
import argparse,copy,gzip,hashlib,json,os,sys,time,subprocess,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=ROOT.parent;ROLL=P/'experiments/native-rolling-h100-20261107'
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gz(p,x):
 with gzip.open(p,'wt',compresslevel=3) as f:json.dump(x,f)
def study(e,m,contract,out):
 import numpy as np,torch,mujoco
 import geometry_contact
 from native_env import Task,spec,install
 from clean_policy import Expert,direct_command
 from interactive_snapshot import CrossResetState
 from locomotion_rollout import spherical
 from m9_cyclic import sample
 from native_transfer import metrics,phase_report
 from retention_eval import phase_quality,MODELS
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 from frozen_bfm_placement import tensor_digest
 torch.set_num_threads(1);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 install();write(out/'speed.json',speed(e));write(out/'backend.json',backend(e,m,ROLL/'compatibility-prefix'))
 assert contract['model_sha256']=='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361'
 task=Task(e,m,contract,6101);dev=e.standing.device;path,digest=MODELS['BC500'];assert sha(path)==digest
 ck=torch.load(path,map_location=dev,weights_only=True);net=Expert(1639,'direct').to(dev);net.load_state_dict(ck['state_dict']);net.eval().requires_grad_(False)
 assert torch.equal(net.center,ck['state_dict']['center']) and torch.equal(net.scale,ck['state_dict']['scale'])
 protocol=json.loads((ROOT/'research_notes/source_coverage_m1_protocol.json').read_text());cells=[c for c in protocol['pilot']['cells'] if c['source']==ARGS.source]
 counts=dict(full_queries=0,extra_FB=0,prediction_physics=0,prediction_commands=0,actual_Teacher_physics=0,actual_Teacher_commands=0,actual_BC_physics=0,actual_native_physics=0,reset_prefix_physics=0,source_selection_physics=0,RL_decisions=0,training_updates=0,F_fits=0,F_calls=0)
 summaries=[];queries=[];labels=[];start=time.monotonic()
 def persist():
  write(out/'counts.json',counts);write(out/'summaries.json',summaries);write(out/'searches.json',queries)
  assert time.monotonic()-start<ARGS.job_limit,'owned worker job limit'
  assert counts['full_queries']<=dict(standing=23,goal_arm5=42,motion_walk_task=23)[ARGS.source]
 def fixed_rng():
  torch.manual_seed(6101);np.random.seed(6101);random.seed(6101);task.rng=np.random.default_rng(6101)
 def reset(ss,label):
  fixed_rng();task.reset(ss,record=True,label=label);counts['reset_prefix_physics']+=600+len(task.prefix_data['physics'])
 def bc():
  _,r,d,st=task.step(net.distribution(task.student_features()).mean);counts['actual_BC_physics']+=st['physics_steps'];return r
 def native():
  _,r,d,st=task.step(task.native(ARGS.source,float(e.tether.time))/16);counts['actual_native_physics']+=st['physics_steps'];return r
 # Fixed state-stratum selection without exposing any contact outcome.
 limits=protocol['source_selection'][ARGS.source]['interval'];reset(spec(0,4,20,ARGS.source,3,False), 'source_selection')
 basew=np.array(task.read_point_state()[0])-np.array(task.initial.get('shoulder_xyz',task.read_point_state()[0]));seen=[];chosen=None;fallback=None
 while e.tether.time<=limits[1]+1e-7 and not task.done:
  t=round(float(e.tether.time),6);v=e.sim.data.qvel[:3].copy();loads=np.array(e.last_support);row=task.physics[-1] if task.physics else task.initial
  w=np.array(row['wrist'])-np.array(row['shoulder_xyz']);dx=float(w[0]-basew[0]);both=bool(np.all(loads>=20));single=bool(np.any(loads<20))
  entry=dict(time=t,root_velocity=v.tolist(),support=loads.tolist(),wrist_shoulder_x_change=dx)
  if t>=limits[0]-1e-7:
   seen.append(entry)
   ok=(both and np.linalg.norm(v[:2])<.08) if ARGS.source=='standing' else (both and np.linalg.norm(v[:2])<.12 and dx>=.015) if ARGS.source=='goal_arm5' else (v[0]>.05 and single)
   if ARGS.source=='motion_walk_task' and v[0]>.05 and fallback is None:fallback=entry
   if ok:chosen=entry;break
  _,_,_,st=task.step(task.native(ARGS.source,t)/16,prefix=True);counts['source_selection_physics']+=st['physics_steps']
 if chosen is None and fallback is not None:chosen=copy.deepcopy(fallback);chosen['explicit_fallback']='positive vx but no low-load support sample in fixed interval'
 write(out/'source-selection.json',dict(chosen=chosen,inspected=seen,boundary=task.boundary,rule=protocol['source_selection'][ARGS.source],group='native-prefix-'+ARGS.source))
 if chosen is None:
  persist();return dict(status='source_stratum_unavailable',counts=counts,source=ARGS.source)
 until=chosen['time'];on=round(until+.6,6)
 # Original7 functions intentionally copied verbatim from interactive_pilot; no candidate additions.
 rewards=json.loads((ROLL/'runs/reward-inference-01/reward-candidates.json').read_text());assert rewards['model_sha256']==contract['model_sha256']
 rz={r['name']:torch.tensor(r['z'],device=dev,dtype=torch.float32).reshape(1,256) for r in rewards['candidates']}
 emb=torch.tensor(next(r['preprojection_embedding'] for r in rewards['candidates'] if r['name']=='forward015_ordinary'),device=dev,dtype=torch.float32)/150000
 fm=m._forward_map;fdigest=tensor_digest(fm)
 for k,h in list(fm._forward_pre_hooks.items()):
  if h.__module__=='frozen_bfm_placement' and 'unused' in h.__qualname__:del fm._forward_pre_hooks[k]
 fm.to(dev);fm.eval().requires_grad_(False);assert tensor_digest(fm)==fdigest
 write(out/'identities.json',dict(BC500=str(path),BC500_SHA=digest,BFM_SHA=contract['model_sha256'],runtime=RUNTIME,native_assets=str(__import__('native_env').RUN/'native-assets.json'),native_assets_sha=sha(__import__('native_env').RUN/'native-assets.json'),reward_candidates_sha=sha(ROLL/'runs/reward-inference-01/reward-candidates.json'),forward_tensor_digest=fdigest,original7_version=sha(ROOT/'research_notes/interactive_pilot.py'),forward_proxy='fixed forward015 unprojected embedding/150000; minhead Fdotembedding; no contact-return claim'))
 def command(c,j,phase):
  if c['kind']=='standing':return e.standing
  if c['kind']=='goal_arm15':return task.native_z['goal_arm15']
  if c['kind']=='reward':return rz[c['name']]
  alpha=float(np.interp(j/23,np.linspace(0,1,4),c['alpha']));base=rz[c['base']] if c.get('base') in rz else sample(task.refs['straight'],phase+j*.1*c['rate']/task.refs['straight']['period'])
  return spherical(spherical(e.standing,base,alpha),task.native_z['goal_arm15'],c.get('arm',0.))
 def propose(rng,prev):
  prior=dict(kind='sequence',base='forward015_ordinary',alpha=[.2,.7,1.,1.],rate=1.,arm=0.)
  if prev and prev['kind']=='sequence':
   prior=copy.deepcopy(prev);xx=np.linspace(0,1,4);prior['alpha']=np.interp(np.minimum(xx+1/23,1),xx,prior['alpha']).tolist()
  cs=[dict(kind='standing'),dict(kind='goal_arm15'),dict(kind='reward',name='forward015_ordinary'),dict(kind='reward',name='forward022_ordinary'),dict(kind='sequence',alpha=[1.]*4,rate=1.,arm=0.),prior]
  cs.append(dict(kind='sequence',base=prior.get('base','motion'),alpha=np.clip(np.array(prior['alpha'])+rng.normal(0,.15,4),0,1).tolist(),rate=float(np.clip(prior['rate']+rng.normal(0,.15),.5,1.4)),arm=float(np.clip(prior.get('arm',0.)+rng.normal(0,.08),0,.2))))
  return cs
 original_frame=geometry_contact.frame;prediction=None
 def frame(s,t):
  if prediction is None:return original_frame(s,t)
  c=prediction;return c['p']+max(0,t-c['t'])*c['v'],c['v'].copy(),c['yaw'],0.
 geometry_contact.frame=frame
 def serial(v):
  if torch.is_tensor(v):return v.detach().cpu()
  if isinstance(v,dict):return {k:serial(x) for k,x in v.items()}
  if isinstance(v,(list,tuple)):return [serial(x) for x in v]
  return v
 def save_state(ss,qdir):
  # Callback objects themselves reconstructed by exact reset+actual command replay recipe.
  state=dict(integration=ss.integration,solver_arrays={k:getattr(ss.data,k).copy() for k in dir(ss.data) if not k.startswith('_') and isinstance(getattr(ss.data,k),np.ndarray)},task=ss.task,env=ss.env,base=ss.base,sim=ss.sim,wrapped=ss.wrapped,tether=ss.tether,history=ss.history,force_history=ss.force_history,origin=ss.origin,torch_rng=ss.rng,cpu_rng=ss.cpu_rng,numpy_rng=np.random.get_state(),python_rng=random.getstate(),task_rng=copy.deepcopy(task.rng.bit_generator.state),spec=copy.deepcopy(task.spec),source_audit=task.prefix_data['audit'],actual_prefix_decisions=copy.deepcopy(task.decision_rows),actual_prefix_controls=copy.deepcopy(task.controls),actual_prefix_physics=copy.deepcopy(task.physics),fingerprint=ss.fingerprint,callback_reconstruction='native Task.reset(saved spec) then replay actual_prefix_decisions raw in order; CrossResetState within live process',physics_identity=contract)
  torch.save(serial(state),qdir/'state.pt')
 def quality(rows,controls):
  mm=metrics(rows,controls);mm['slip_peak_mps']=max(r['contact_tangent_speed_max_mps'] for r in rows);mm['slip_p95_mps']=float(np.quantile([r['contact_tangent_speed_max_mps'] for r in rows],.95));last=rows[-1]['pelvis_matrix'];mm['yaw_final_deg']=float(np.degrees(np.arctan2(last[1][0],last[0][0])));mm['error_xyz_rms_cm']=(100*np.sqrt(np.mean([(np.array(r['anchor'])-r['wrist'])**2 for r in rows if r['active']],axis=0))).tolist() if any(r['active'] for r in rows) else None;return mm
 def search(cell,label,phase=0.,previous=None,rng=None):
  nonlocal prediction
  persist();counts['full_queries']+=1;idx=counts['full_queries'];qdir=out/f'query-{idx:03d}';qdir.mkdir();started=time.monotonic();ss=CrossResetState(task);n0=len(task.physics);c0=len(task.controls);save_state(ss,qdir)
  now=float(e.tether.time);p,v,yaw,_=original_frame(task.spec,now);current_pred=dict(t=now,p=np.array(p).copy(),v=np.array(v).copy(),yaw=yaw)
  cs=propose(rng or np.random.default_rng(6101),previous);sequences=[[command(c,j,phase).detach().clone() for j in range(24)] for c in cs];records=[]
  torch.save(dict(previous_z=task.last_command.cpu(),BFM_obs={k:v.cpu() for k,v in e.obs.items()},privileged=task.features().cpu(),restricted=task.student_features().cpu(),proposals=cs,plans=torch.stack([torch.cat(x) for x in sequences]).cpu(),phase=phase,previous_plan=previous,partner_prediction={k:np.asarray(v).tolist() for k,v in current_pred.items()}),qdir/'plans.pt')
  for i,(c,seq) in enumerate(zip(cs,sequences)):
   ss.restore();z0=spherical(task.last_command,seq[0],.2);action=m.act(e.obs,z0,mean=True);ff=m.forward_map(e.obs,z0,action);counts['F_calls']+=1;score=float((ff*emb).sum(-1).min())
   torch.save(dict(original_F=ff.cpu(),score=score,score_embedding=emb.cpu(),target_z=seq[0].cpu(),first_interpolated_z=z0.cpu(),actual_low_action=action.cpu(),BFM_obs={k:v.cpu() for k,v in e.obs.items()},previous_z=task.last_command.cpu()),qdir/f'candidate-{i}-F.pt')
   prediction=current_pred
   for k in ['stop_time','deceleration','change_time','after_curvature','rotation_onset','rotation_return','rotation_return_ramp']:task.spec.pop(k,None)
   task.spec.update(anchor_mode='fixed_prediction_origin',distance=.72,ramp=1.,curvature=0.,rotation_deg=0.,onset=-1000.,release=100000.,end=now+3.4,enabled=bool(e.tether.active))
   total=0.;sttime=time.monotonic()
   for j,z in enumerate(seq):
    _,r,d,st=task.step(z/16);total+=r;counts['prediction_physics']+=st['physics_steps'];counts['prediction_commands']+=1
    if j==0:
     cfirst=task.controls[c0];assert np.max(np.abs(np.array(cfirst['executed_z'])-z0.cpu().numpy()))<1e-5
     assert np.max(np.abs(np.array(cfirst['action'])-action[0].cpu().numpy()))<1e-6
    if d:break
   rr=task.physics[n0:];cc=task.controls[c0:];mm=quality(rr,cc);rrscore=float(-1e6-abs(total) if task.boundary else total)
   gz(qdir/f'candidate-{i}-prediction.json.gz',dict(identity='prediction_not_actual_not_PPO',cell=cell,query=idx,candidate=i,proposal=c,phase=phase,physics=rr,control=cc,rl_stats=task.stats[-(j+1):],boundary=task.boundary))
   records.append(dict(index=i,proposal=c,F_score=score,physical_score=rrscore,physical=mm,boundary=copy.deepcopy(task.boundary),steps=j+1,wall_s=time.monotonic()-sttime,asset_prefix=f'query-{idx:03d}/candidate-{i}',source_version='original7 unchanged',target_first_z_vs_actual_interpolated='separately saved in F.pt; all24 plans in plans.pt',phase_compatibility=dict(proposal_phase=phase,actual_last_z=ss.task['last_command'][0].cpu().tolist(),note='coldphase0 on Student state; actual Teacher phase updated only from own commands; no borrowed phase')))
  prediction=None;ss.restore();assert ss.digest()==ss.fingerprint
  best=max(range(7),key=lambda i:records[i]['physical_score']);row=dict(cell=cell,query=idx,label=label,time=now,group='native-prefix-'+ARGS.source,split='train',state_fingerprint=ss.fingerprint,state_asset=f'query-{idx:03d}/state.pt',phase=phase,previous_plan=previous,selected=best,candidates=records,wall_s=time.monotonic()-started,scope='7 predictions; actual use tracked separately',future_available=False)
  write(qdir/'summary.json',row);queries.append(row);persist();print('QUERY',ARGS.source,idx,label,now,best,records[best]['physical']['error_cm'],records[best]['physical']['force_rms'],flush=True)
  return sequences[best],cs[best],row
 def save(label,identity,cell):
  task.label=label;row=task.save(out);row.update(phase_report(task));row.update(quality=phase_quality(task),all_quality=quality(task.physics,task.controls),identity=identity,cell=cell,source_group='native-prefix-'+ARGS.source,BC500_SHA=digest,runtime=RUNTIME)
  row['rl_decisions']=0;row['actual_high_commands']=len(task.decision_rows);row['not_PPO']=True;summaries.append(row);persist();return row
 # No-contact native continuation: same source, no freezing limbs, full duration.
 reset(spec(0,4,on,ARGS.source,until,False),'native_no_contact')
 while not task.done:native()
 save('native_no_contact','actual_native_continuation_no_contact','native_no_contact')
 for cell in cells:
  sspec=spec(cell['distance'],cell['ramp'],on,ARGS.source,until,True,on+cell['stop_offset'] if cell['stop_offset'] is not None else None)
  reset(sspec,cell['id']+'_BC');initial=CrossResetState(task);write(out/(cell['id']+'-initial.json'),dict(source_audit=task.prefix_data['audit'],fingerprint=initial.fingerprint,source_stratum=chosen))
  querytime=round(on+.1,6);snapshot=None
  while not task.done:
   if abs(float(e.tether.time)-querytime)<1e-6:snapshot=CrossResetState(task)
   bc()
  bcrow=save(cell['id']+'_BC','actual_Student_frozen_BC500',cell['id']);assert snapshot is not None
  # Native control uses exact same onset state prefix as BC until handoff; full native suffix.
  initial.restore();assert initial.digest()==initial.fingerprint
  while not task.done:native()
  save(cell['id']+'_native','actual_native_continuation_contact_control',cell['id'])
  # Verify snapshot restore reproduces unchanged Student next command across intervening branch.
  snapshot.restore();finger=snapshot.fingerprint;raw=net.distribution(task.student_features()).mean;_,_,_,st=task.step(raw);counts['actual_BC_physics']+=st['physics_steps'];nextraw=copy.deepcopy(task.physics[-st['physics_steps']:]);snapshot.restore();bc();assert task.physics[-st['physics_steps']:]==nextraw;snapshot.restore()
  rng=np.random.default_rng(6101);seq,proposal,q=search(cell['id'],'Student_visited_first',rng=rng)
  for z in seq:
   _,_,d,st=task.step(z/16);counts['actual_Teacher_physics']+=st['physics_steps'];counts['actual_Teacher_commands']+=1
   if d:break
  while not task.done:bc()
  save(cell['id']+'_best_sequence','actual_selected_plan24_then_BC_aftermath',cell['id'])
  if cell['rolling']:
   snapshot.restore();assert snapshot.digest()==finger;phase=0.;prev=None;actual_queries=[]
   for j in range(20):
    if task.done:break
    if j==0:plan,prev,qr=seq,proposal,q
    else:plan,prev,qr=search(cell['id'],'Teacher_own_visited',phase,prev,rng)
    z=plan[0];sx=task.student_features().detach().cpu();px=task.features().detach().cpu();previousz=task.last_command.detach().cpu();t=float(e.tether.time)
    _,_,d,st=task.step(z/16);counts['actual_Teacher_physics']+=st['physics_steps'];counts['actual_Teacher_commands']+=1;actual_queries.append(qr['query'])
    labels.append(dict(cell=cell['id'],group=cell['group'],query=qr['query'],time=t,restricted=sx,privileged=px,previous_z=previousz,z=z.cpu(),identity='actual_Teacher_first_command',valid_for_positive_imitation=False,qualification='pending complete gate, not auto label'))
    phase+=.1*prev.get('rate',0.)/task.refs['straight']['period'] if prev['kind']=='sequence' else 0.
   endwindow=float(e.tether.time)
   while not task.done:bc()
   rr=save(cell['id']+'_rolling','actual_Teacher20_firstcommands_then_BC_aftermath',cell['id']);write(out/(cell['id']+'-window.json'),dict(start=querytime,end=endwindow,queries=actual_queries,actual_identity=rr['identity'],full_episode_is_not_all_Teacher=True))
   torch.save(dict(rows=labels,scope='actual Teacher only; pending final absolute qualification'),out/'actual-teacher-labels.pt')
  print('CELL_COMPLETED',cell['id'],flush=True)
 assert sha(path)==digest and tensor_digest(m._forward_map)==fdigest
 persist();return dict(status='completed',counts=counts,source=ARGS.source,independent_native_prefix_groups=1,cells=len(cells),no_training=True)
def main():
 global ARGS,RUNTIME
 p=argparse.ArgumentParser();p.add_argument('--source',choices=['standing','goal_arm5','motion_walk_task'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--job-limit',type=float,default=4500);ARGS=p.parse_args();out=ARGS.output;out.mkdir(exist_ok=False)
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain']);RUNTIME=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 write(out/'launch.json',dict(runtime=RUNTIME,protocol_sha=sha(ROOT/'research_notes/source_coverage_m1_protocol.json'),source=ARGS.source,scope='pure qualification and paired acquisition'))
 os.environ['CLEAN_KIND']='direct';os.environ['P4000_CANONICAL_ROW_ACTOR']='1';os.environ['BFM_RECOVERY_OUTPUT']=str(out);os.chdir(out)
 import m10_stage,m10_physics
 m10_physics.OUT=out;m10_physics.MODEL=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view';m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
