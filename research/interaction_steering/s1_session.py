"""Bounded S1 actual execution; explicit legacy physical-worker boundary only.
This module imports archived physics helpers inside the dedicated worker process.
No search, optimizer, sealed task generator or predictor is constructed.
"""
import copy,json,time,math,hashlib
from pathlib import Path
import numpy as np
import torch
from .contracts import CausalObservation,Guidance,Tier
from .source_adapter import ProgressToLatent,ProgressMemory
from .guidance import ForwardAdmittance,AdmittanceParameters

class Session:
 def __init__(self,t,e,m,nets,out,args,identities):
  self.t=t;self.e=e;self.m=m;self.nets=nets;self.out=out;self.args=args;self.identities=identities
  self.rows=[];self.checks=[];self.counts=dict(actual_ticks=0,decisions=0,training_updates=0,search_queries=0);self.start=time.monotonic()
  self.targets=json.loads((Path(__file__).parents[2]/'research_notes/s1_state_draft.json').read_text())['states']
  self.ids={r['state_id'] for r in self.targets};self.prefix={};self.post_original=None
  r=t.refs['straight'];digest=hashlib.sha256(json.dumps(r['z'],separators=(',',':')).encode()).hexdigest()
  assert digest=='f3bb1f490aa5f90a4a78653e3f3f78af79d47ddadc92d502632deb8c5550e907' and r['period']==1.42
  self.adapter=ProgressToLatent(r['tensor'],e.standing,1.42,.214,digest)
  self.admittance=ForwardAdmittance(AdmittanceParameters(10.,30.,.22))
 def write(self,name,data):
  p=self.out/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(p)
 def includes(self,sid):return sid in self.ids
 def obs(self):
  # Copy only declared causal fields. No Task/spec object crosses into Guidance.
  d=self.e.sim.data;R=d.xmat[1].reshape(3,3);yaw=math.atan2(R[1,0],R[0,0]);c,s=math.cos(yaw),math.sin(yaw);Y=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
  co=self.t.controls[-4:];assert len(co)==4
  states=[np.asarray(x['bfm_obs']['state']).reshape(64).tolist() for x in co]
  actions=[x['action'] for x in co]
  ph=self.t.physics[-100:];assert len(ph)>=97
  forces=[(Y.T@np.asarray(x['force'])).tolist() for x in ph[-100::4]];assert len(forces)==25
  forces[-1]=(Y.T@np.asarray(self.e.tether.force)).tolist()
  o=CausalObservation(tuple(d.qpos[7:]),tuple(d.qvel[6:]),tuple(R.T@np.array([0.,0.,-1.])),tuple(d.qvel[3:6]),tuple(map(tuple,actions)),tuple(map(tuple,states)),tuple(map(tuple,forces)),tuple(self.t.last_command.detach().cpu().reshape(-1).tolist()),'right_wrist_yaw_link',(0.,0.,0.),'forward_guidance')
  return o,Y
 def action(self,kind,memory,velocity):
  o,Y=self.obs()
  if kind=='BC500':return self.nets['BC500'].base.distribution(self.t.student_features()).mean,memory,velocity,None,o
  if kind=='CS':u,velocity=self.admittance.step(o,velocity)
  elif kind=='CP':
   # The environment exposes CURRENT partner velocity only, never its spec/program.
   v=float((Y.T@np.asarray(self.e.tether.partner_velocity))[0]);u=Guidance(max(0.,min(.22,v)),0.,0.,True,Tier.CP)
  else:u=Guidance(float(kind.split(':')[1]),0.,0.,True,Tier.CS)
  z,memory=self.adapter.map(u,memory);return z/16,memory,velocity,u,o
 def execute(self,meta,kind,rep):
  from frl_v3_risk import score,metrics
  from f1_sequence_search import semantics
  from dataclasses import asdict
  t=self.t;start=float(self.e.tether.time);pi=len(t.physics);ci=len(t.controls);spec=copy.deepcopy(t.spec);memory=ProgressMemory(0.);velocity=0.;dec=[];at=time.monotonic()
  extend=rep==0 and (meta['state_id'].startswith('new_stop-') or meta['state_id'].startswith('new_near-')) and not kind.startswith('fixed:')
  end=max(start+3.,min(start+12.,float(spec['release'])+2.)) if extend else start+3.
  while float(self.e.tether.time)<end-1e-7 and not t.done:
   assert self.counts['actual_ticks']<45000 and self.counts['decisions']<2250
   raw,memory,velocity,u,o=self.action(kind,memory,velocity)
   dec.append(dict(time=float(self.e.tether.time),raw=raw.detach().cpu().clone(),previous_z=t.last_command.detach().cpu().clone(),guidance=asdict(u) if u else None,phase=memory.phase,causal=asdict(o),input=t.student_features().cpu().clone() if kind=='BC500' else None))
   before=len(t.physics);t.step(raw);self.counts['actual_ticks']+=len(t.physics)-before;self.counts['decisions']+=1
  ph=copy.deepcopy(t.physics[pi:]);co=copy.deepcopy(t.controls[ci:]);assert ph
  # Three-second primary endpoint remains matched despite release extensions.
  p3=[r for r in ph if r['force_time']<start+3.-1e-8];c3=[r for r in co if r['time']<start+3.-1e-8]
  def met(p,c):
   sp=copy.deepcopy(spec);sp.setdefault('demand_family','none' if not sp['enabled'] else 'mid');v=metrics(p,c,sp)
   active=[r for r in p if r['active']];v['force_xyz_rms_N']=np.sqrt(np.mean([np.asarray(r['force'])**2 for r in active],0)).tolist() if active else [0.,0.,0.]
   v['error_xyz_rms_cm']=(100*np.sqrt(np.mean([(np.asarray(r['anchor'])-r['wrist'])**2 for r in active],0))).tolist() if active else None
   v['root_dx_m']=float(p[-1]['qpos'][0]-p[0]['qpos'][0]);v['root_mean_vx_mps']=float(np.mean([r['qvel'][0] for r in p]));v['physical_boundary']=copy.deepcopy(t.boundary)
   v['low_slip_task_success']=bool(v['low_slip_task_success'] and t.boundary is None and len(p)>=599)
   return v
  rr=dict(identity='actual_controller_execution',state_id=meta['state_id'],parent_id=meta['parent_id'],controller=kind,rep=rep,spec=spec,physics=ph,controls=co,decisions=dec,primary=met(p3,c3),full=met(ph,co),score3=score(p3,c3,start),slip_semantics=semantics(p3),boundary=copy.deepcopy(t.boundary),actual_seconds=len(ph)*.005,requested_end=end,start=start)
  path=self.out/(meta['state_id']+'-'+kind.replace(':','_')+'-'+str(rep)+'.pt');torch.save(rr,path)
  row={k:rr[k] for k in ('state_id','parent_id','controller','rep','primary','full','score3','slip_semantics','boundary','actual_seconds','requested_end','start')};row.update(artifact=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,job_s=time.monotonic()-at)
  self.rows.append(row);self.write('results.json',self.rows);self.write('counts.json',self.counts)
  print('S1_BRANCH',meta['state_id'],kind,rep,len(ph),row['primary']['error_cm'],row['primary']['force_rms'],flush=True)
  return rr
 def state(self,meta,owner,oldrow):
  from dual_track_runtime import TrainingBranch
  from diagnostic_frozen import portable
  from f1_mjdata_contract import exact,metadata
  from f1_sequence_search import install_post_velocity
  from frl_v2_task import continuation
  import frl_v3_task
  ss=TrainingBranch(self.t);originalrisk=frl_v3_task.RISK_COEFFICIENT;frl_v3_task.RISK_COEFFICIENT=0.
  snapshot=dict(state=portable(self.t),state_validity=metadata(self.e.sim.model,self.e.sim.data),previous_z=self.t.last_command.cpu(),meta=meta)
  torch.save(snapshot,self.out/(meta['state_id']+'-initial.pt'))
  try:
   # Diagnostic hook is read-only and is removed before next archived reset/reconstruction.
   original=install_post_velocity(self.e)
   specs=[(meta,copy.deepcopy(self.t.spec))]
   if meta['state_id'].startswith('new_stop-'):specs.append((dict(meta,state_id=meta['state_id']+'-continue'),continuation(self.t.spec)))
   for mm,sp in specs:
    kinds=['BC500','CS','CP']+(['fixed:0','fixed:0.09','fixed:0.18'] if mm['state_id']=='native-standing' else [])
    for kind in kinds:
     records=[]
     for rep in (0,1):
      ss.restore();self.t.spec.clear();self.t.spec.update(copy.deepcopy(sp));self.e._original_step=original;install_post_velocity(self.e)
      records.append(self.execute(mm,kind,rep))
     a,b=records;cut=a['start']+3.-1e-8
     for key,tkey in [('physics','force_time'),('controls','time'),('decisions','time')]:
      aa=[r for r in a[key] if r[tkey]<cut];bb=[r for r in b[key] if r[tkey]<cut];assert exact(aa,bb),('repeat mismatch',mm['state_id'],kind,key)
     self.checks.append(dict(test='repeat first3s physics/control/causal_inputs/commands exact',state=mm['state_id'],controller=kind,passed=True))
     if mm['state_id'].startswith('new_stop-'):
      key=(kind,'continue' if mm['state_id'].endswith('-continue') else 'stop');self.prefix[key]=a
   if meta['state_id'].startswith('new_stop-'):
    cut=specs[0][1]['stop_time']
    for kind in ['BC500','CS','CP']:
     a,b=self.prefix[(kind,'stop')],self.prefix[(kind,'continue')]
     for key,tkey in [('physics','force_time'),('controls','time'),('decisions','time')]:assert exact([r for r in a[key] if r[tkey]<cut-1e-8],[r for r in b[key] if r[tkey]<cut-1e-8]),('future leakage or prefix mismatch',kind,key)
     self.checks.append(dict(test='stop/continue pre-observable-branch exact',controller=kind,cut=cut,passed=True))
  finally:
   ss.restore();self.e._original_step=original;frl_v3_task.RISK_COEFFICIENT=originalrisk
   self.write('checks.json',self.checks)
 def native(self,P,read,sha):
  from diagnostic_frozen import portable
  from f1_mjdata_contract import same,metadata
  import frl_v3_task
  N=P/'experiments/fb-native-audit-20261110/runs/native-inventory-01';a=read(N/'standing-actual.json.gz');torch.manual_seed(56017);np.random.seed(56017);frl_v3_task.RISK_COEFFICIENT=0.;self.t.reset(a['spec'],record=True,label='native-standing')
  j=0
  while float(self.e.tether.time)<6.-1e-8:
   target=a['targets'][j];assert abs(float(self.e.tether.time)-target['time'])<1e-7;self.t.step(torch.tensor(target['z'],device='cuda',dtype=torch.float32).reshape(1,-1)/16);j+=1
  row=next(x for x in json.loads((P/'experiments/f1-l0-array-audit-20261111/runs/l0-complete-02/state-manifest.json').read_text()) if x['state_id']=='native-standing')
  prior=torch.load(P/'experiments/f1-l0-array-audit-20261111'/row['state_snapshot'],map_location='cpu',weights_only=False)
  actual=portable(self.t);md=metadata(self.e.sim.model,self.e.sim.data)
  if not same(actual,prior['state'],md,prior['state_validity']):
   adjusted=copy.deepcopy(actual);adjusted['mjdata_arrays']['efc_JT']=prior['state']['mjdata_arrays']['efc_JT'].copy();assert same(adjusted,prior['state'],md,prior['state_validity']), 'native portable mismatch beyond archived efc_JT'
   self.e.sim.data.efc_JT[:]=prior['state']['mjdata_arrays']['efc_JT']
  assert same(portable(self.t),prior['state'],md,prior['state_validity'])
  self.checks.append(dict(test='native-standing archived full state exact',passed=True))
  self.state(row,'native_standing',row)
 def finish(self):
  self.write('checks.json',self.checks);self.write('counts.json',self.counts)
  return dict(status='completed',branches=len(self.rows),parents=len({x['parent_id'] for x in self.rows}),counts=self.counts,training=False)
