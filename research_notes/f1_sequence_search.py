"""Bounded F1 sequence/causal-feedback diagnostics. No optimizer, no sealed inputs."""
import copy,gzip,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from dual_track_runtime import TrainingBranch
from diagnostic_frozen import portable
from f1_mjdata_contract import exact,same,metadata
from frl_v3_risk import score,metrics
from clean_policy import direct_command
from locomotion_rollout import spherical
from m9_cyclic import sample

def write(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def install_post_velocity(e):
 import mujoco
 old=e._original_step
 def measured():
  old();m,d=e.sim.model,e.sim.data;rows=[]
  for con in d.contact[:d.ncon]:
   ids=[int(m.geom_bodyid[con.geom1]),int(m.geom_bodyid[con.geom2])]
   if 0 not in ids:continue
   body=max(ids);name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,body) or ''
   if 'ankle' not in name and 'foot' not in name:continue
   jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,con.pos,body)
   rows.append(dict(body=name,post_tangent_mps=float(np.linalg.norm(con.frame.reshape(3,3)[1:]@(jac@d.qvel)))))
  assert len(rows)==len(e.contact_sample['stage_foot_contacts'])
  e.contact_sample['f1_post_contacts']=rows
 e._original_step=measured
 return old

def semantics(ph):
 # Preserve ALL old events; post-speed is separate diagnostic, not a filter.
 old=0;transient=0;sust=0;normal=[];run={};maxrun={};travel={};impulse={}
 for r in ph:
  present=set()
  for a,b in zip(r['stage_foot_contacts'],r.get('f1_post_contacts',[])):
   assert a['body']==b['body'];k=a['body'];present.add(k)
   loaded=a['normal_N']>1e-6;over=loaded and a['tangent_speed_mps']>.45
   if over:old+=1;normal.append(a['normal_N'])
   if over and b['post_tangent_mps']<=.45:transient+=1
   if over and b['post_tangent_mps']>.45:sust+=1
   run[k]=run.get(k,0)+.005 if over and b['post_tangent_mps']>.45 else 0.
   maxrun[k]=max(maxrun.get(k,0.),run[k]);travel[k]=travel.get(k,0.)+(b['post_tangent_mps']*.005 if loaded else 0.)
   impulse[k]=impulse.get(k,0.)+(a['normal_N']*.005 if over and b['post_tangent_mps']<=.45 else 0.)
  for k in set(run)-present:run[k]=0.
 return dict(old_loaded_overlimit_contact_rows=old,loaded_landing_transient_rows=transient,persistent_pre_post_rows=sust,max_consecutive_persistent_s=maxrun,loaded_point_speed_integral_m=travel,transient_normal_impulse_Ns=impulse,event_normal_N_min=min(normal) if normal else None,event_normal_N_max=max(normal) if normal else None,threshold_N=1e-6,threshold_mps=.45,scope='all contacts retained; speed integral is point travel proxy, not tracked-material-point displacement')

class Search:
 def __init__(self,t,e,m,nets,out,args,identities):
  self.t=t;self.e=e;self.m=m;self.nets=nets;self.out=out;self.args=args;self.identities=identities;self.rows=[];self.queries=[];self.checks=[];self.counts=dict(prediction_ticks=0,actual_ticks=0,queries=0,training_updates=0,L3=0);self.start=time.monotonic()
  self.protocol=json.loads(Path(__file__).with_name('f1_l1_l2_protocol.json').read_text());self.plan=json.loads((Path(__file__).resolve().parents[2]/'prep/state-plan.json').read_text())
  import geometry_contact
  self.original_frame=geometry_contact.frame;self.prediction=None
  def frame(s,tm):
   if self.prediction is None:return self.original_frame(s,tm)
   p=self.prediction;return p['position']+max(0.,tm-p['time'])*p['velocity'],p['velocity'].copy(),p['angle'],0.
  geometry_contact.frame=frame
  # Observation only; must be installed after archived prefix verification.
  self.post_installed=False
 def includes(self,sid):return self.args.layer=='L1' or sid in self.protocol['L2_states']
 def x(self):return torch.cat([self.t.student_features(),self.t.features()],-1)
 def action(self,key):
  if key=='BC500':return self.nets[key].base.distribution(self.t.student_features()).mean
  return self.nets[key].distribution(self.x()).mean
 def candidate_pool(self,horizon,small=False):
  t=self.t;e=self.e;nodes={k:direct_command(self.action(k),e.standing).clone() for k in self.nets}
  last=t.last_command.clone();ref=t.refs['straight']
  # Current last-z phase match only. Never another trajectory's MPC phase.
  ps=[j/8 for j in range(8)];p=min(ps,key=lambda p:float((sample(ref,p)-last).square().sum()))
  pool=[dict(name=k,kind='policy',key=k) for k in ('BC500','B64','B128','v3-32','v3-128')]
  pool += [dict(name='hold_current',kind='static',z=last),dict(name='standing',kind='static',z=e.standing.clone()),dict(name='arm5',kind='static',z=t.native_z['goal_arm5'].clone()),dict(name='arm15',kind='static',z=t.native_z['goal_arm15'].clone())]
  for alpha,rate in (([.25,.5,.75],.5),([.5,1.,1.],1.),([1.,.5,0.],.5)):
   pool.append(dict(name='motion_'+str(len(pool)),kind='motion',alpha=alpha,rate=rate,phase=p,phase_match_squared_distance=float((sample(ref,p)-last).square().sum())))
  for a,b in [('BC500','B128'),('B128','BC500'),('BC500','v3-128'),('v3-128','BC500')]:
   pool.append(dict(name=a+'_to_'+b,kind='nodes',zs=[nodes[a],spherical(nodes[a],nodes[b],.5),nodes[b]]))
  if small:pool=[pool[i] for i in (0,2,4,5,6,7,9,10)]
  return pool
 def command(self,c,j,horizon):
  if c['kind']=='policy':return self.action(c['key'])
  if c['kind']=='static':return c['z']/16
  u=j/max(1,round(horizon*10)-1)
  if c['kind']=='nodes':
   v=u*2;i=min(1,int(v));return spherical(c['zs'][i],c['zs'][i+1],v-i)/16
  a=float(np.interp(u,[0,.5,1],c['alpha']));z=sample(self.t.refs['straight'],c['phase']+j*.1*c['rate']/self.t.refs['straight']['period'])
  return spherical(self.e.standing,z,a)/16
 def describe(self,c):
  return {k:([x.detach().cpu().tolist() for x in v] if k=='zs' else v.detach().cpu().tolist() if torch.is_tensor(v) else v) for k,v in c.items()}
 def execute(self,c,horizon,identity,spec=None):
  t=self.t;pi=len(t.physics);ci=len(t.controls);at=float(self.e.tether.time);inputs=[];raws=[];targets=[];previous=[]
  for j in range(round(horizon*10)):
   if t.done:break
   inputs.append(self.x().cpu().clone());previous.append(t.last_command.cpu().clone());raw=self.command(c,j,horizon);raws.append(raw.cpu().clone());targets.append(direct_command(raw,self.e.standing).cpu());t.step(raw)
  ph=copy.deepcopy(t.physics[pi:]);co=copy.deepcopy(t.controls[ci:]);assert ph
  # All rollouts use lambda0 runtime and both locked scores reconstructed once.
  sc=score(ph,co,at);s=copy.deepcopy(spec if spec is not None else t.spec);s.setdefault('demand_family','mid')
  met=metrics(ph,co,s);met['low_slip_task_success']=bool(met['low_slip_task_success'] and t.boundary is None and len(ph)>=round(horizon*200)-1)
  met['torso_yaw_final_deg']=math.degrees(math.atan2(ph[-1]['torso_matrix'][1][0],ph[-1]['torso_matrix'][0][0]))
  active=[r for r in ph if r['active']]
  met['error_xyz_rms_cm']=(100*np.sqrt(np.mean([(np.array(r['anchor'])-r['wrist'])**2 for r in active],0))).tolist() if active else [0,0,0]
  ret=dict(identity=identity,physics=ph,controls=co,inputs=inputs,raws=raws,target_z=targets,previous_z=previous,score=sc,metrics=met,boundary=copy.deepcopy(t.boundary),slip_semantics=semantics(ph),horizon_requested=horizon,actual_seconds=len(ph)*.005)
  self.counts['prediction_ticks' if identity=='candidate_prediction' else 'actual_ticks']+=len(ph)
  return ret
 def save(self,label,r):
  path=self.out/(label+'.pt');torch.save(r,path)
  return dict(artifact=path.name,artifact_bytes=path.stat().st_size,artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 def summarize(self,r):return {k:r[k] for k in ('score','metrics','boundary','slip_semantics','actual_seconds')}
 def permission(self,tier,meta,oldrow,horizon,wall,n,selected):
  return dict(state_parent_id=meta['parent_id'],state_id=meta['state_id'],state_snapshot_sha=oldrow['state_snapshot_sha256'],source_branch=meta['state_id'],source_policy_sha=oldrow['model_sha256'],runtime_sha=self.args.runtime,actual_runtime_sha=self.args.runtime,phase=meta['phase'],stop_continue_role=meta.get('paired_role','original'),alias_group_id=None,information_tier=tier,actor_observation_schema='1639D ideal-force restricted' if tier=='C-S' else '2727D privileged or native commands',candidate_generator_inputs=['current real proprio/action/last-z/contact history']+([] if tier=='C-S' else ['current anchor and partner velocity']),allowed_partner_fields=[] if tier=='C-S' else ['current anchor','current velocity','current active','current yaw'],actual_partner_fields_read=[] if tier=='C-S' else ['current anchor','current velocity','current active','current yaw']+(['exact future scenario'] if tier=='O-F' else []),partner_predictor_version={'C-S':'none-fixed-BC500','C-P':'constant_current_velocity_v1','O-F':'original_exact_future'}[tier],partner_predictor_inputs=[] if tier=='C-S' else ['current position/velocity/active/yaw']+(['actual scenario future'] if tier=='O-F' else []),predictor_input_fields=[] if tier=='C-S' else ['current position/velocity/active/yaw'],score_features=['independent reporter only'] if tier=='C-S' else ['perpoint force/error','relative velocity','heading','continuity','slip','tilt','idle','v3 risk'],score_component_fields=['original physics reward','v3 event .33'],simulator_state_origin='L0 verified real complete prefix plus RNG/solver',future_anchor_accessible=tier=='O-F',future_anchor_consumed=tier=='O-F',future_trajectory_accessible=tier=='O-F',future_trajectory_consumed=tier=='O-F',anchor_error_used_in_selection=tier!='C-S',true_anchor_error_in_selection=tier!='C-S',selection_horizon=horizon,planning_horizon=horizon,physical_eval_horizon=horizon if tier=='O-F' else .1,physical_horizon=horizon if tier=='O-F' else .1,query_wall_seconds=wall,cumulative_job_seconds=time.monotonic()-self.start,number_of_simulated_candidates=n,selected_candidate_id=selected,independent_reporter_fields='all physical truth after execution; never returned to fixed BC500',proposal_source='finite frozen policy/native/time-node pool',raw_to_legal_z='original direct_command norm16 then original five substep spherical interpolation')
 def selection(self,rs):
  # Same locked v3 return. Hard boundary candidates rejected; no new reward.
  safe=[i for i,r in enumerate(rs) if r['boundary'] is None]
  return max(safe or range(len(rs)),key=lambda i:rs[i]['score']['v3_discounted_return'])
 def state(self,meta,owner,oldrow):
  original_step=install_post_velocity(self.e)
  import frl_v3_task
  frl_v3_task.RISK_COEFFICIENT=0.
  # Extra measurements do not enter actor/history or alter physics.
  try:
   if self.args.layer=='L1':self.l1(meta,owner,oldrow)
   else:self.l2(meta,owner,oldrow)
  finally:self.e._original_step=original_step
  write(self.out/'candidate-scores.json',self.rows);write(self.out/'checks.json',self.checks);write(self.out/'counts.json',self.counts)
 def l1(self,meta,owner,oldrow):
  t=self.t;ss=TrainingBranch(t);pool=self.candidate_pool(3.);baseline=portable(t);md=metadata(self.e.sim.model,self.e.sim.data);rs=[];start=time.monotonic()
  for i,c in enumerate(pool):
   ss.restore();assert same(baseline,portable(t),md,metadata(self.e.sim.model,self.e.sim.data))
   r=self.execute(c,3.,'candidate_prediction');record=dict(layer='L1',state_id=meta['state_id'],parent_id=meta['parent_id'],phase=meta['phase'],candidate=i,proposal=self.describe(c),**self.summarize(r),**self.save(meta['state_id']+f'-L1-{i:02}',r));rs.append(r);self.rows.append(record)
  selected=self.selection(rs);self.queries.append(self.permission('O-F',meta,oldrow,3.,time.monotonic()-start,len(pool),selected));self.counts['queries']+=1
  write(self.out/'query-ledger.json',self.queries);ss.restore();print('L1_STATE',meta['state_id'],'good',sum(r['metrics']['low_slip_task_success'] for r in rs),'best',selected,'error',rs[selected]['metrics']['error_cm'],'slip',rs[selected]['metrics']['loaded_slip_peak'],flush=True)
 def predict(self,horizon):
  t=self.t;now=float(self.e.tether.time)
  # Current snapshot values only. Current position relative to current origin.
  pred=dict(time=now,position=np.array(self.e.tether.anchor)-np.array(t.geometry_origin),velocity=np.array(self.e.tether.partner_velocity).copy(),angle=float(self.e.tether.geometry_target_yaw-t.geometry_yaw),active=bool(self.e.tether.active))
  self.prediction=pred
  # Replace IN PLACE so all closure references see identical sanitized spec.
  # distance is causal predicted endpoint displacement for existing heading cost.
  s=dict(geometry='wrist',enabled=pred['active'],onset=-1000.,release=100000.,end=now+horizon+1.,distance=float(np.linalg.norm(pred['position']+horizon*pred['velocity'])),ramp=1.,curvature=0.,heading_deg=0.,rotation_deg=0.,family='walking',source='standing',source_until=0.,handoff_time=-1000.,total_cap=10.,anchor_mode='fixed_prediction_origin',cached_observation_reads=True,demand_family='causal_prediction')
  t.spec.clear();t.spec.update(s)
  return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in pred.items()}
 def query(self,meta,oldrow,index,audit=False):
  t=self.t;ss=TrainingBranch(t);base=portable(t);md=metadata(self.e.sim.model,self.e.sim.data);pool=self.candidate_pool(2.,small=True);rs=[];start=time.monotonic();prediction=None
  try:
   for i,c in enumerate(pool):
    ss.restore();assert same(base,portable(t),md,metadata(self.e.sim.model,self.e.sim.data))
    prediction=self.predict(2.);r=self.execute(c,2.,'candidate_prediction');rs.append(r)
    self.rows.append(dict(layer='L2-prediction',state_id=meta['state_id'],parent_id=meta['parent_id'],replan=index,candidate=i,proposal=self.describe(c),prediction=prediction,**self.summarize(r),**self.save(meta['state_id']+f'-L2-q{index:02}-{i}',r)))
    self.prediction=None
   selected=self.selection(rs)
   if audit:
    # Future-hidden adversarial intervention: identical state/history, different
    # inaccessible scenario. Sanitization must erase it, exact prediction result.
    ss.restore();t.spec.update(distance=9.,ramp=.013,stop_time=float(self.e.tether.time)+.001,release=float(self.e.tether.time)+.003,end=float(self.e.tether.time)+.004)
    self.predict(2.);repeat=self.execute(pool[0],2.,'candidate_prediction')
    assert exact(rs[0],repeat),'future-spec intervention changed causal predicted rollout'
    self.checks.append(dict(state_id=meta['state_id'],test='future-spec invariance',passed=True,compared='all physics/control/inputs/raw/z/scores byte-exact'))
  finally:self.prediction=None;ss.restore()
  q=self.permission('C-P',meta,oldrow,2.,time.monotonic()-start,len(pool),selected);q.update(replan=index,time=float(self.e.tether.time),prediction=prediction,initial_previous_z=t.last_command.cpu().tolist(),current_restricted_input_sha256=hashlib.sha256(t.student_features().cpu().numpy().tobytes()).hexdigest(),current_privileged_input_sha256=hashlib.sha256(self.x().cpu().numpy().tobytes()).hexdigest(),prediction_reward_distance_rule='norm(current relative anchor + 2s current velocity), no final requested distance')
  self.queries.append(q);self.counts['queries']+=1;write(self.out/'query-ledger.json',self.queries)
  return rs[selected]['raws'][0].to(self.e.standing.device)
 def l2(self,meta,owner,oldrow):
  ss=TrainingBranch(self.t);spec=copy.deepcopy(self.t.spec)
  first=self.l2_once(meta,owner,oldrow)
  if meta['state_id']==self.protocol['paired_stop_state']:
   from frl_v2_task import continuation
   ss.restore();self.t.spec.clear();self.t.spec.update(continuation(spec))
   second=self.l2_once(dict(meta,state_id=meta['state_id']+'-continue',paired_role='continue'),owner,oldrow)
   cut=spec['stop_time']
   a=[r for r in first['physics'] if r['force_time']<cut-1e-7];b=[r for r in second['physics'] if r['force_time']<cut-1e-7]
   aa=[r for r in first['decisions'] if r['time']<cut-1e-7];bb=[r for r in second['decisions'] if r['time']<cut-1e-7]
   assert exact(a,b) and exact(aa,bb),'real stop/continue prefix differs'
   self.checks.append(dict(test='real C-P stop/continue prefix',parent=meta['parent_id'],physics_ticks=len(a),commands=len(aa),passed=True))
  ss.restore()
 def l2_once(self,meta,owner,oldrow):
  t=self.t;ss=TrainingBranch(t);spec=copy.deepcopy(t.spec)
  r=self.execute(dict(kind='policy',key='BC500'),3.,'fixed_restricted_execution');self.rows.append(dict(layer='L2-C-S',state_id=meta['state_id'],parent_id=meta['parent_id'],**self.summarize(r),**self.save(meta['state_id']+'-L2-CS',r)));self.queries.append(self.permission('C-S',meta,oldrow,0.,0.,0,'BC500'))
  ss.restore();pi=len(t.physics);ci=len(t.controls);at=float(self.e.tether.time);actual=[]
  for j in range(30):
   if t.done:break
   raw=self.query(meta,oldrow,j,audit=(j==0));actual.append(dict(time=float(self.e.tether.time),raw=raw.cpu(),input=self.x().cpu(),previous_z=t.last_command.cpu()));t.step(raw);self.counts['actual_ticks']+=20
   print('L2_STEP',meta['state_id'],j,flush=True)
  ph=copy.deepcopy(t.physics[pi:]);co=copy.deepcopy(t.controls[ci:]);rr=dict(identity='real_causal_feedback_execution',physics=ph,controls=co,decisions=actual,score=score(ph,co,at),metrics=metrics(ph,co,spec),boundary=copy.deepcopy(t.boundary),slip_semantics=semantics(ph),actual_seconds=len(ph)*.005)
  rr['metrics']['low_slip_task_success']=bool(rr['metrics']['low_slip_task_success'] and t.boundary is None and len(ph)>=599)
  self.rows.append(dict(layer='L2-C-P',state_id=meta['state_id'],parent_id=meta['parent_id'],**self.summarize(rr),**self.save(meta['state_id']+'-L2-CP',rr)))
  ss.restore();return rr
 def finish(self):
  write(self.out/'candidate-scores.json',self.rows);write(self.out/'query-ledger.json',self.queries);write(self.out/'checks.json',self.checks);write(self.out/'counts.json',self.counts)
  result=dict(status='completed',layer=self.args.layer,runtime=self.args.runtime,states=sorted(set(r['state_id'] for r in self.rows)),parents=len(set(r['parent_id'] for r in self.rows)),rows=len(self.rows),counts=self.counts,wall_s=time.monotonic()-self.start,C_S_planner='not implemented; fixed BC500 only',L3='not started and not queued')
  write(self.out/'complete.json',result);return result
