"""Issue60 additive authorization 6801908284: matched future-forecast diagnostic."""
import copy, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from f1_sequence_search import Search as Base, write, semantics
from dual_track_runtime import TrainingBranch
from diagnostic_frozen import portable
from f1_mjdata_contract import exact, same, metadata
from frl_v3_risk import score, metrics

STATES=('new_tail-hold-97ff10c4dcd9','new_stop-sustained-5f6380cdec75')
class Search(Base):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw)
  self.oracle=False;self.oracle_spec=None
  self.archive=Path(__file__).resolve().parents[2]/'runs/l2-01'
  self.counts['L3']=1
  import geometry_contact
  causal_frame=geometry_contact.frame
  def frame(spec,tm):
   if self.oracle_spec is not None:return self.original_frame(self.oracle_spec,tm)
   return causal_frame(spec,tm)
  geometry_contact.frame=frame
 def includes(self,sid):return sid in STATES
 def predict(self,horizon):
  future=copy.deepcopy(self.t.spec) if self.oracle else None
  pred=super().predict(horizon)
  if self.oracle:
   # Retain exactly the causal heading-distance scale, source metadata and
   # candidate pool. Only partner trajectory/active schedule is oracle.
   self.oracle_spec=future;self.prediction=None
   for key in ('enabled','onset','release','physical_force_off'):
    if key in future:self.t.spec[key]=future[key]
   pred.update(predictor='exact_future_partner_program',future_spec=future,
               reward_distance=self.t.spec['distance'],future_used_for_heading_scale=False)
  return pred
 def query(self,*a,**kw):
  start=len(self.rows)
  try:
   raw=super().query(*a,**kw)
   if self.oracle:
    q=self.queries[-1]
    for key in ('future_anchor_accessible','future_anchor_consumed','future_trajectory_accessible','future_trajectory_consumed'):q[key]=True
    q.update(information_tier='O-F',partner_predictor_version='original_exact_future_matched_v1',
      actual_partner_fields_read=['current anchor/velocity/active/yaw','future partner program and active/release schedule'],
      partner_predictor_inputs=['complete actual partner spec'],predictor_input_fields=['complete actual partner spec'],
      allowed_partner_fields=['current and future partner trajectory/active'],physical_eval_horizon=.1,physical_horizon=.1,
      actual_execution_identity='oracle_feedback_execution',comparison='same8/same2s/samev3; only partner forecast permission',
      future_used_for_heading_scale=False)
    for r in self.rows[start:]:r.update(layer='L3-O-F-prediction',information_tier='O-F')
    write(self.out/'query-ledger.json',self.queries)
   return raw
  finally:self.oracle_spec=None;self.prediction=None
 def reuse_check(self,meta,oldrow):
  sid=meta['state_id'];ss=TrainingBranch(self.t)
  saved=torch.load(self.archive/(sid+'-query-state-00.pt'),map_location='cpu',weights_only=False)
  assert same(portable(self.t),saved['state'],metadata(self.e.sim.model,self.e.sim.data),saved['state_validity']),('C-P initial state mismatch',sid)
  assert torch.equal(self.x().cpu(),saved['input']) and torch.equal(self.t.last_command.cpu(),saved['previous_z'])
  pool=self.candidate_pool(2.,small=True)
  oldrows=json.loads((self.archive/'candidate-scores.json').read_text())
  oldq=[r for r in oldrows if r['layer']=='L2-prediction' and r['state_id']==sid and r['replan']==0]
  assert len(oldq)==8
  for i,c in enumerate(pool):assert exact(self.describe(c),oldq[i]['proposal'])
  self.oracle=False;self.query(meta,oldrow,0,audit=False)
  for r in self.rows[-8:]:
   archived=torch.load(self.archive/(sid+f"-L2-q00-{r['candidate']}.pt"),map_location='cpu',weights_only=False)
   actual=torch.load(self.out/r['artifact'],map_location='cpu',weights_only=False)
   assert exact(actual,archived),('C-P first query full trace differs',sid,r['candidate'])
   oldpath=self.out/r['artifact'];newpath=oldpath.with_name('reuse-'+oldpath.name);oldpath.rename(newpath)
   r.update(layer='L3-C-P-reuse-check',information_tier='C-P',artifact=newpath.name)
  self.queries[-1]['purpose']='C-P archive reuse verification, not new scientific condition'
  oldpath=self.out/self.queries[-1]['current_state_artifact'];newpath=oldpath.with_name('reuse-'+oldpath.name);oldpath.rename(newpath);self.queries[-1]['current_state_artifact']=newpath.name
  ss.restore();original=torch.load(self.archive/(sid+'-L2-CP.pt'),map_location='cpu',weights_only=False)
  pi=len(self.t.physics);ci=len(self.t.controls)
  for d in original['decisions']:
   assert torch.equal(self.x().cpu(),d['input']) and torch.equal(self.t.last_command.cpu(),d['previous_z'])
   self.t.step(d['raw'].to(self.e.standing.device));self.counts['actual_ticks']+=20
  assert exact(self.t.physics[pi:],original['physics']) and exact(self.t.controls[ci:],original['controls']),('C-P replay differs',sid)
  self.checks.append(dict(test='strict C-P reuse: full initial state/pool/all8 first-query branches/full actual trajectory',state_id=sid,passed=True,
    archive_runtime='ad93901e9b75791ae918dd37664b7d6f8b8e86bb',
    actual_artifact_sha256=hashlib.sha256((self.archive/(sid+'-L2-CP.pt')).read_bytes()).hexdigest()))
  ss.restore();write(self.out/'checks.json',self.checks)
 def l2(self,meta,owner,oldrow):
  ss=TrainingBranch(self.t);spec=copy.deepcopy(self.t.spec)
  self.l3_once(meta,owner,oldrow)
  if meta['state_id']==STATES[1]:
   from frl_v2_task import continuation
   ss.restore();self.t.spec.clear();self.t.spec.update(continuation(spec))
   self.l3_once(dict(meta,state_id=meta['state_id']+'-continue',paired_role='continue'),owner,oldrow)
  ss.restore()
 def l3_once(self,meta,owner,oldrow):
  t=self.t;self.reuse_check(meta,oldrow);ss=TrainingBranch(t);spec=copy.deepcopy(t.spec)
  pi=len(t.physics);ci=len(t.controls);at=float(self.e.tether.time);actual=[];self.oracle=True
  for j in range(30):
   if t.done:break
   raw=self.query(meta,oldrow,j,audit=False)
   actual.append(dict(time=float(self.e.tether.time),raw=raw.cpu(),input=self.x().cpu(),previous_z=t.last_command.cpu()))
   assert self.oracle_spec is None and self.prediction is None
   t.step(raw);self.counts['actual_ticks']+=20
   print('L3_STEP',meta['state_id'],j,flush=True)
  self.oracle=False
  ph=copy.deepcopy(t.physics[pi:]);co=copy.deepcopy(t.controls[ci:])
  rr=dict(identity='real_oracle_feedback_execution',physics=ph,controls=co,decisions=actual,score=score(ph,co,at),metrics=metrics(ph,co,spec),boundary=copy.deepcopy(t.boundary),slip_semantics=semantics(ph),actual_seconds=len(ph)*.005)
  rr['metrics']['low_slip_task_success']=bool(rr['metrics']['low_slip_task_success'] and t.boundary is None and len(ph)>=599)
  self.rows.append(dict(layer='L3-O-F',state_id=meta['state_id'],parent_id=meta['parent_id'],**self.summarize(rr),**self.save(meta['state_id']+'-L3-OF',rr)))
  write(self.out/'candidate-scores.json',self.rows);write(self.out/'counts.json',self.counts);ss.restore()
 def finish(self):
  r=super().finish();r.update(L3='completed; F1 scientific work stops',S1='not authorized',C_S_planner='missing')
  write(self.out/'complete.json',r);return r
