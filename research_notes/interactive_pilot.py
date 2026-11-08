"""Issue53 bounded Student-state recovery + frozen FB shadow. No Expert PPO."""
import argparse,copy,gzip,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
S=ROOT.parent;P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');OLD=P/'experiments/native-transfer-20261107';ROLL=P/'experiments/native-rolling-h100-20261107'
PLAN=json.loads((ROOT/'research_notes/interactive_protocol.json').read_text())
def write(p,x):
 p=Path(p);p.write_text(json.dumps(x,indent=2,allow_nan=False))
def read(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else Path(p).open() as f:return json.load(f)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def study(e,m,contract,out):
 import numpy as np,torch,mujoco
 import native_env,geometry_contact
 from native_env import Task,spec,install
 from clean_policy import Expert,direct_command
 from rolling_snapshot import BranchState
 from locomotion_rollout import spherical
 from m9_cyclic import sample
 from native_transfer import metrics,phase_report
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 torch.set_num_threads(1);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 install();write(out/'speed.json',speed(e));write(out/'backend.json',backend(e,m,ROLL/'compatibility-prefix'))
 assert contract['model_sha256']=='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361'
 assert set(e.keys)=={'state','last_action','history_actor'}
 task=Task(e,m,contract,5301);dev=e.standing.device
 def load_s(path,digest):
  assert sha(path)==digest;ck=torch.load(path,map_location=dev,weights_only=True);assert ck['input_dim']==1639
  n=Expert(1639,'direct').to(dev);n.load_state_dict(ck['state_dict']);n.eval().requires_grad_(False);return n
 student=load_s(P/PLAN['source_student']['checkpoint'],PLAN['source_student']['sha256'])
 identity=dict(S4000=PLAN['source_student'],S_init=dict(path=str(OLD/'runs/bc-fit-01/S-0.pt'),sha256=sha(OLD/'runs/bc-fit-01/S-0.pt')),historic_S0=dict(path=str(P/'bridge-v2-owned/runs/direct-transfer-20261103/A-prepare-01/job/ppo15-student-r0.pt'),sha256=sha(P/'bridge-v2-owned/runs/direct-transfer-20261103/A-prepare-01/job/ppo15-student-r0.pt')))
 assert identity['S_init']['sha256']==PLAN['s_init_sha256'] and identity['historic_S0']['sha256']==PLAN['historic_s0_sha256'];write(out/'models.json',identity)
 case_specs={'normal72':spec(.72,4.8,5.),'short08':spec(.08,1.6,5.2),'mid24':spec(.24,3.,5.3),'stop66':spec(.66,4.4,5.4,stop=8.1),'none':spec(0,3,5.2,enabled=False)}
 # Sensor installation, not activation mask. Never concatenate privileged feature slices.
 task.reset(case_specs['normal72'],record=True,label='audit')
 assert task.student_features().shape==(1,1639) and task.features().shape==(1,1088) and task.pro.shape[-1]==465
 write(out/'permissions.json',dict(roles=PLAN['student_actor_audit'],bfm_selected_keys=list(e.keys),metadata_excluded=['time','privileged_state'],source_actor='S4000 restricted; no online privileged neural teacher',ideal_force_only=True))
 counts=dict(full_queries=0,FB_extra_sequences=0,candidate_physics_steps=0,actual_Student_physics_steps=0,actual_Teacher_physics_steps=0,counterfactual_first_physics_steps=0,reset_prefill_physics_steps=600+len(task.prefix_data['physics']),supervised_updates=0,Student_PPO_decisions=0)
 query_rows=[];labels=[];summaries=[];cache={};snapshot_times=[]
 rewards=read(ROLL/'runs/reward-inference-01/reward-candidates.json');assert rewards['model_sha256']==contract['model_sha256']
 rz={r['name']:torch.tensor(r['z'],device=dev,dtype=torch.float32).reshape(1,256) for r in rewards['candidates']};assert len(rz)==6
 emb=torch.tensor(next(r['preprojection_embedding'] for r in rewards['candidates'] if r['name']=='forward015_ordinary'),device=dev,dtype=torch.float32)/150000
 original_frame=geometry_contact.frame;prediction=None
 def frame(s,t):
  if prediction is None:return original_frame(s,t)
  c=prediction;return c['p']+max(0,t-c['t'])*c['v'],c['v'].copy(),c['yaw'],0.
 geometry_contact.frame=frame
 f_status=None
 def init_f():
  nonlocal f_status
  if f_status is not None:return f_status['status']=='available'
  from frozen_bfm_placement import tensor_digest
  st=time.monotonic();fm=m._forward_map;digest=tensor_digest(fm)
  # Remove only the local inference-placement guard for this explicitly authorized shadow module.
  hooks=[]
  for k,h in list(fm._forward_pre_hooks.items()):
   if h.__module__=='frozen_bfm_placement' and 'unused' in h.__qualname__:hooks.append(k);del fm._forward_pre_hooks[k]
  fm.to(dev);fm.eval().requires_grad_(False);assert tensor_digest(fm)==digest
  f_status=dict(status='available',seconds=time.monotonic()-st,guard_removed_local_ids=hooks,frozen_tensor_digest=digest,source='same loaded BFM forward_map; original normalize; no new checkpoint',cfg= m.cfg.archi.f.model_dump())
  write(out/'F-contract.json',f_status);return True
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
 def forward_scores(candidates,phase):
  init_f();ts=time.monotonic();scores=[];shapes=[]
  for c in candidates:
   z=spherical(task.last_command,command(c,0,phase),.2);action=m.act(e.obs,z,mean=True)
   ff=m.forward_map(e.obs,z,action);assert torch.is_tensor(ff) and ff.shape[-1]==256
   vv=(ff*emb).sum(-1);scores.append(float(vv.min()));shapes.append(list(ff.shape))
  return scores,dict(seconds=time.monotonic()-ts,forward_output_shapes=shapes,obs_keys=list(e.obs),low_action_dim=int(action.shape[-1]),z_norm=float(z.norm()),proxy='fixed robot-state forward015 embedding; no external contact value claim')
 def persist():
  write(out/'counts.json',counts);write(out/'searches.json',query_rows)
 def search(label,phase=0.,previous=None,shadow=False,rng=None):
  nonlocal prediction
  assert counts['full_queries']<96;counts['full_queries']+=1
  rng=rng or np.random.default_rng(5301);t0=time.monotonic();ss=BranchState(task);snapshot_s=time.monotonic()-t0;n0=len(task.physics);c0=len(task.controls)
  now=float(e.tether.time);p,v,yaw,_=original_frame(task.spec,now);current_pred=dict(t=now,p=np.array(p).copy(),v=np.array(v).copy(),yaw=yaw)
  candidates=propose(rng,previous);nom_s=time.monotonic()-t0-snapshot_s;fs=None;extras=[];f_detail=None
  if shadow:
   all_extra=[];duplicates=[]
   pool_first=[command(c,0,phase) for c in candidates]
   for name in rz:
    if name in ['forward015_ordinary','forward022_ordinary']:continue
    c=dict(kind='reward',name=name);zz=command(c,0,phase)
    if any(float((zz-z0).abs().max())<1e-5 for z0 in pool_first):duplicates.append(name);continue
    all_extra.append(c);pool_first.append(zz)
   assert len(all_extra)>=2,'fewer than two distinct shadow proposals; preserve budget and diagnose'
   fs,f_detail=forward_scores(candidates+all_extra,phase);rank=sorted(range(len(all_extra)),key=lambda j:fs[7+j],reverse=True)
   extras=[all_extra[j] for j in rank[:2]];f_detail['all_available_extra']=all_extra;f_detail['all_F_scores']=fs;f_detail['duplicate_exclusions']=duplicates
  records=[];seqs=[];restores=[]
  for i,c in enumerate(candidates+extras):
   if i>=7:
    assert counts['FB_extra_sequences']<40;counts['FB_extra_sequences']+=1
   rt=time.monotonic();ss.restore();restores.append(time.monotonic()-rt);prediction=current_pred
   for k in ['stop_time','deceleration','change_time','after_curvature','rotation_onset','rotation_return','rotation_return_ramp']:task.spec.pop(k,None)
   task.spec.update(anchor_mode='fixed_prediction_origin',distance=.72,ramp=1.,curvature=0.,rotation_deg=0.,onset=-1000.,release=100000.,end=now+3.4,enabled=bool(e.tether.active))
   st=time.monotonic();total=0.;seq=[]
   for j in range(24):
    z=command(c,j,phase);seq.append(z.detach().clone());_,r,d,stat=task.step(z/16);counts['candidate_physics_steps']+=stat['physics_steps'];total+=r
    if d:break
   rr=task.physics[n0:];cc=task.controls[c0:];mm=metrics(rr,cc)
   records.append(dict(index=i,proposal=c,score=float(-1e6-abs(total) if task.boundary else total),boundary=copy.deepcopy(task.boundary),physical=mm,wall_s=time.monotonic()-st,steps=len(seq),BFM_calls=len(cc),scope='prediction, not actual trajectory'))
   seqs.append(seq)
  prediction=None;rt=time.monotonic();ss.restore();restores.append(time.monotonic()-rt);assert ss.digest()==ss.fingerprint
  best=max(range(7),key=lambda i:records[i]['score']);z=seqs[best][0]
  row=dict(label=label,time=now,state_fingerprint=ss.fingerprint,phase_start=phase,previous_plan=previous,query_kind='full_original7',selected=best,candidates=records,wall_s=time.monotonic()-t0,nomination_s=nom_s,snapshot_s=snapshot_s,restore_s=restores,privileged_input=task.features()[0].cpu().tolist(),restricted_input=task.student_features()[0].cpu().tolist(),previous_z=task.last_command[0].cpu().tolist(),selected_z=z[0].cpu().tolist(),no_actual_future=True,first_action_scope='counterfactual query until explicitly executed')
  if shadow:
   pool=[0,1,7,8,4,5,6];newbest=max(pool,key=lambda i:records[i]['score']);originalbest=records[best]['score'];gain=records[newbest]['score']-originalbest
   rank7=sorted(range(7),key=lambda i:fs[i],reverse=True)
   row['shadow']=dict(nominated=extras,F=f_detail,original_pool=list(range(7)),replacement_pool=pool,original_best=best,replacement_best=newbest,replacement_gain=gain,nominee_wins=bool(newbest>=7 and gain>1e-8),F_top4_retains_physical_top7=best in rank7[:4],F_rank7=rank7,physical_rank7=sorted(range(7),key=lambda i:records[i]['score'],reverse=True),equal_candidate_budget=True)
  query_rows.append(row);persist();print('QUERY',counts['full_queries'],label,round(now,2),'best',best,'wall',round(row['wall_s'],2),'extra',len(extras),'err/F',round(records[best]['physical']['error_cm'] or 0,2),round(records[best]['physical']['force_rms'],2),flush=True)
  return z,candidates[best],row
 def student_step():
  raw=student.distribution(task.student_features()).mean;_,r,d,st=task.step(raw);counts['actual_Student_physics_steps']+=st['physics_steps'];return r
 def reset(name):
  task.reset(case_specs[name],record=True,label='S4000_'+name);counts['reset_prefill_physics_steps']+=600+len(task.prefix_data['physics'])
 def reach(name,t):
  reset(name)
  while e.tether.time<t-1e-7 and not task.done:student_step()
  assert not task.done and abs(float(e.tether.time)-t)<1e-6
 def save(label,scope):
  task.label=label;row=task.save(out);row.update(phase_report(task));row['scope']=scope;summaries.append(row);write(out/'summaries.json',summaries);return row
 def make_label(row,source,qualified):
  return dict(restricted=torch.tensor(row['restricted_input']),privileged=torch.tensor(row['privileged_input']),z=torch.tensor(row['selected_z']),previous_z=torch.tensor(row['previous_z']),episode=row['label'],time=row['time'],query_index=query_rows.index(row),source=source,valid=qualified,state_fingerprint=row['state_fingerprint'])
 if ARGS.mode=='moving-cost':
  # Real moving Student state after a documented, already executed early assistance prefix.
  ep=read(S/'runs/diagnose-01/Teacher_replan_2s_normal72_4.7.json.gz');task.reset(ep['condition'],record=True,label='actual_moving_replay');parity=dict(qpos=0.,qvel=0.,qpd=0.);j=0
  while float(e.tether.time)<7.3-1e-7:
   c=ep['control'][5*j];parity['qpos']=max(parity['qpos'],float(np.max(abs(e.sim.data.qpos-np.array(c['pre_qpos'])))));parity['qvel']=max(parity['qvel'],float(np.max(abs(e.sim.data.qvel-np.array(c['pre_qvel'])))))
   raw=torch.tensor(ep['decisions'][j]['raw'],device=dev,dtype=torch.float32)[None];_,_,_,st=task.step(raw);counts['actual_Student_physics_steps']+=st['physics_steps'];parity['qpd']=max(parity['qpd'],float(np.max(abs(np.array([c['q_PD'] for c in task.controls[-5:]])-np.array([c['q_PD'] for c in ep['control'][5*j:5*j+5]])))));j+=1
  assert max(parity.values())==0.,parity
  old_counts=read(S/'runs/diagnose-01/counts.json');assert old_counts['full_queries']+1<=96
  state_motion=dict(root_velocity=e.sim.data.qvel[:3].tolist(),foot_loads=list(e.last_support),previous_assistance='Teacher4.7-6.7s; actual Student6.7-7.3s, full own executed history replayed',no_borrowed_MPC_phase=True)
  z,_,q=search('moving_student_after_assistance',phase=0.,previous=None,shadow=False);_,_,_,st=task.step(z/16);counts['counterfactual_first_physics_steps']+=st['physics_steps'];save('moving_student_first_Teacher_command','actual first100ms only, no full moving recovery claim, never training label')
  write(out/'moving-cost.json',dict(full_prefix_replay_errors=parity,current_actual_motion=state_motion,query_wall_s=q['wall_s'],selected=q['selected'],scope='one cost-only Student-moving query after documented actual assistance; no qualification label',counts=counts));persist();return dict(status='completed',counts=counts,student_optimizer_updates=0)
 if ARGS.mode=='diagnose':
  # Four first cost checks. Three are also members of the predetermined shadow cohort.
  for c in PLAN['cost_first_states']:
   reach(c['case'],c['time']);_,_,q=search('cost_'+c['case']+'_'+str(c['time']),shadow=False);cache[(c['case'],c['time'])]=q
  cost=[q['wall_s'] for q in query_rows];write(out/'first-cost.json',dict(p50=float(np.median(cost)),p90=float(np.quantile(cost,.9)),queries=len(cost),scope='actual H100 current backend first four; same7/24 physics'))
  # Shadows are evaluated anew so each 7vs7 comparison shares exact state; cost queries retained separately.
  snapshots={}
  for name,times in PLAN['shadow_states'].items():
   reset(name)
   while not task.done:
    tt=round(float(e.tether.time),6)
    if any(abs(tt-t)<1e-6 for t in times):
     _,_,q=search('shadow_'+name+'_'+str(tt),shadow=True);snapshots[(name,tt)]=BranchState(task);snapshot_times.append(dict(case=name,time=tt,fingerprint=q['state_fingerprint']))
    student_step()
   save('S4000_'+name,'actual Student complete rollout; no Teacher actions executed')
  write(out/'visited-states.json',snapshot_times)
  # Full exact-prefix three-arm 2s takeover, then Student remainder separately saved.
  recoveries=[]
  for c in PLAN['takeover']['candidates']:
   ss=snapshots[(c['case'],c['time'])];ss.restore();n0=len(task.physics);c0=len(task.controls);results={};branch_prefix={};newlabels=[]
   for arm in ['Student_continue','first_Teacher_then_Student','Teacher_replan_2s']:
    ss.restore();branch_prefix[arm]=ss.digest();phase=0.;prev=None;rew=0.;rng=np.random.default_rng(5301);actual_ids=[];start=time.monotonic()
    for j in range(20):
     if arm=='Student_continue' or (arm=='first_Teacher_then_Student' and j>0):rew+=student_step()
     else:
      # j0 uses already evaluated same-state original7 query, avoiding duplicate full searches.
      if j==0:
       q=next(q for q in query_rows if q['label']=='shadow_'+c['case']+'_'+str(c['time']));z=torch.tensor(q['selected_z'],device=dev,dtype=torch.float32)[None];prev=copy.deepcopy(q['candidates'][q['selected']]['proposal'])
       assert ss.digest()==q['state_fingerprint']
      else:z,prev,q=search(arm+'_'+c['case']+'_'+str(c['time']),phase,prev,rng=rng)
      actual_ids.append(query_rows.index(q));newlabels.append(make_label(q,'actual Teacher takeover first command',False)) if arm=='Teacher_replan_2s' else None
      _,r,d,st=task.step(z/16);rew+=r
      key='actual_Teacher_physics_steps' if arm=='Teacher_replan_2s' else 'counterfactual_first_physics_steps';counts[key]+=st['physics_steps']
      phase+=.1*prev.get('rate',0.)/task.refs['straight']['period'] if prev['kind']=='sequence' else 0.
     if task.done:break
    rs=task.physics[n0:];cc=task.controls[c0:];results[arm]=dict(reward=rew,physical=metrics(rs,cc),boundary=copy.deepcopy(task.boundary),query_indices=actual_ids,wall_s=time.monotonic()-start,rows=len(rs),prefix=ss.fingerprint)
    # Complete physical aftermath uses frozen Student, clearly distinct from closed-loop Teacher window.
    while not task.done:student_step()
    save(arm+'_'+c['case']+'_'+str(c['time']),'actual same-prefix branch; Teacher replans only first2s, thereafter frozen Student; not full-episode Teacher')
   assert len(set(branch_prefix.values()))==1
   a=results['Student_continue'];b=results['Teacher_replan_2s'];am=a['physical'];bm=b['physical']
   qualified=(b['boundary'] is None and b['reward']>a['reward']+1e-6 and bm['force_rms']<=am['force_rms'] and (am['error_cm'] is None or bm['error_cm']<=am['error_cm']))
   for lab in newlabels:lab['valid']=qualified
   # Only original visited start is a qualified Student-state query; descendants remain separate expert trajectories.
   baseq=next(q for q in query_rows if q['label']=='shadow_'+c['case']+'_'+str(c['time']));labels.append(make_label(baseq,'Student visited counterfactual, recovery checked2s',qualified));labels+=newlabels
   recoveries.append(dict(case=c['case'],time=c['time'],prefix_equal=True,arms=results,qualified=qualified,qualification_scope='only tested start and actually executed2s expert descendants; not all Student trajectory states'))
   write(out/'recoveries.json',recoveries);torch.save(dict(rows=labels,scope='verified labels only; exact roles retained'),out/'interactive-labels.pt');persist();print('RECOVERY',c['case'],c['time'],qualified,'Student',am['error_cm'],am['force_rms'],'Teacher',bm['error_cm'],bm['force_rms'],flush=True)
  early=[r for r in recoveries if r['time']<=5.4];gate=any(r['qualified'] for r in early)
  valid=[r for r in labels if r['valid']];shadow=[q['shadow'] for q in query_rows if 'shadow' in q];wins=sum(x['nominee_wins'] for x in shadow)
  write(out/'gate.json',dict(Teacher_early_recovery_any=gate,qualified_starts=sum(r['qualified'] for r in recoveries),tested_starts=len(recoveries),valid_labels=len(valid),qualified_Student_visited_starts=sum(r['qualified'] for r in recoveries),FB_comparisons=len(shadow),FB_wins=wins,FB_pause_current_proposer=len(shadow)>=20 and wins==0,BC_allowed=gate and bool(valid),Student_PPO_started=False,not_new_Expert_PPO=True))
  from frozen_bfm_placement import tensor_digest
  assert tensor_digest(m._forward_map)==f_status['frozen_tensor_digest']
  persist();return dict(status='completed',counts=counts,recovery_qualified=sum(r['qualified'] for r in recoveries),FB_wins=wins,student_optimizer_updates=0)
 raise ValueError(ARGS.mode)
def main():
 global ARGS
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['diagnose','moving-cost'],required=True);p.add_argument('--output',type=Path,required=True);ARGS=p.parse_args();out=ARGS.output;out.mkdir(exist_ok=False)
 import subprocess
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 write(out/'launch.json',dict(runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),protocol_sha256=sha(ROOT/'research_notes/interactive_protocol.json'),source_sha256=sha(__file__),mode=ARGS.mode))
 os.environ['BFM_RECOVERY_OUTPUT']=str(out);os.chdir(out)
 import m10_physics,m10_stage
 m10_physics.OUT=out;m10_physics.MODEL=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view';m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
