"""S1 isolated legacy worker: archived reset/physical helpers only; no planner or training.
Process-local imports/CFG configuration are confined to this entry. Pure interfaces do not import legacy scripts."""
import argparse,copy,gzip,json,os,sys,time,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];ST=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ.setdefault('TMPDIR',str(ST/'tmp'));os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(ST/'torch-cache'))
import m11_ipc
P=ST.parents[1];V=P/'experiments/frl-v3-risk-20261111';OLD=P/'experiments/frl-v2-fb-repair-20261110'
def read(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else Path(p).open() as f:return json.load(f)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def study(e,m,c,out):
 import numpy as np,torch,mujoco
 from native_env import install
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 from frl_v2_high import build
 from frl_v3_counterfactual import tensorsame
 from f1_mjdata_contract import metadata,canonical,same,install_restore_guard,exact
 install_restore_guard()
 assert read(P/'experiments/f1-l0-array-audit-20261111/runs/array-audit-03/checks.json')['status']=='passed'
 from dual_track_runtime import TrainingBranch
 from diagnostic_frozen import portable
 import frl_v3_task
 from frl_v3_risk import score
 install();speed(e);torch.set_num_threads(1);torch.set_grad_enabled(False)
 cold=time.monotonic();(out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix');backend_wall=time.monotonic()-cold
 frl_v3_task.configure(.33);task=frl_v3_task.Task(e,m,c,5801)
 paths={'BC500':OLD/'runs/frl-formal-B-01/candidate-0.pt','B64':OLD/'runs/frl-formal-B-01/candidate-64.pt','B128':OLD/'runs/frl-formal-B-01/candidate-128.pt','v3-32':V/'runs/formal-v3-lambda033-02/candidate-32.pt','v3-128':V/'runs/formal-v3-lambda033-02/candidate-128.pt'}
 nets={};identities={}
 for k,p in paths.items():
  n=build().eval().requires_grad_(False);n.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model'],strict=True);nets[k]=n;identities[k]={'sha256':sha(p),'path':str(p.relative_to(P))}
 assert c['model_sha256']=='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361'
 plan=read(ST/'prep/state-plan.json');results=[];timings=[];queries=[];prefixchecks=[];native_refs=[]
 from research.interaction_steering.s1_session import Session as Search
 search=Search(task,e,m,nets,out,ARGS,identities)
 def xnow():return torch.cat([task.student_features(),task.features()],-1)
 def capture(meta,owner,archive=None):
  x=xnow();pp=portable(task);md=metadata(e.sim.model,e.sim.data)
  oldrow=next(r for r in read(P/'experiments/f1-l0-array-audit-20261111/runs/l0-complete-02/state-manifest.json') if r['state_id']==meta['state_id'])
  prior=torch.load(P/'experiments/f1-l0-array-audit-20261111'/oldrow['state_snapshot'],map_location='cpu',weights_only=False)
  assert sha(P/'experiments/f1-l0-array-audit-20261111'/oldrow['state_snapshot'])==oldrow['state_snapshot_sha256']
  assert torch.equal(x.cpu(),prior['input']) and torch.equal(task.last_command.cpu(),prior['previous_z'])
  if not same(pp,prior['state'],md,prior['state_validity']):
   diffpath=out/(meta['state_id']+'-restore-difference.pt')
   torch.save(dict(actual=pp,archived=prior['state'],actual_meta=md,archived_meta=prior['state_validity']),diffpath)
   # Conservatively restore this allocated array, NEVER mask it or relax equality.
   check=copy.deepcopy(pp);check['mjdata_arrays']['efc_JT']=prior['state']['mjdata_arrays']['efc_JT'].copy()
   assert same(check,prior['state'],md,prior['state_validity']),('unclassified complete-state difference',meta['state_id'])
   assert e.sim.data.efc_JT.shape==prior['state']['mjdata_arrays']['efc_JT'].shape
   before=TrainingBranch(task);oldrisk=frl_v3_task.RISK_COEFFICIENT;frl_v3_task.RISK_COEFFICIENT=0.
   def probe():
    pi=len(task.physics);ci=len(task.controls);tm=float(e.tether.time);xx=[];rr=[]
    for _ in range(30):
     if task.done:break
     xx.append(xnow().cpu().clone());raw=nets['BC500'].base.distribution(task.student_features()).mean;rr.append(raw.cpu().clone());task.step(raw)
    ph=copy.deepcopy(task.physics[pi:]);co=copy.deepcopy(task.controls[ci:])
    return dict(physics=ph,controls=co,inputs=xx,raws=rr,score=score(ph,co,tm))
   aa=probe();before.restore();e.sim.data.efc_JT[:]=prior['state']['mjdata_arrays']['efc_JT'];assert same(portable(task),prior['state'],md,prior['state_validity'])
   fixed=TrainingBranch(task);bb=probe();assert exact(aa,bb),'efc_JT replay-cache restoration changed measured control/physics'
   torch.save(dict(before=aa,after=bb,equal=True),out/(meta['state_id']+'-efcJT-intervention.pt'))
   fixed.restore();frl_v3_task.RISK_COEFFICIENT=oldrisk;assert same(portable(task),prior['state'],md,prior['state_validity'])
   search.checks.append(dict(state_id=meta['state_id'],test='explicit archived efc_JT restore; all other state exact',passed=True,physics_control_score_bitwise_equal=True,seconds=len(aa['physics'])*.005,compared_array_shape=list(e.sim.data.efc_JT.shape),array_excluded=False))
   print('CACHE_RESTORED',meta['state_id'],len(aa['physics']),flush=True)
  search.checks.append(dict(test='L0 complete reconstructed state exact',state_id=meta['state_id'],passed=True))
  if not search.includes(meta['state_id']):return
  completed=set()
  for name in ARGS.reuse:
   path=ST/'runs'/name/'candidate-scores.json'
   if path.exists():completed.update(r['state_id'] for r in read(path) if r.get('layer')=='L1')
  if meta['state_id'] not in completed:search.state(meta,owner,oldrow)
 # Recorded raw prefixes rebuild original callback/partner graph, never pose-only reset.
 for row in plan['states']:
  if row['source_kind']=='p1_snapshot':
   a=torch.load(P/row['source_snapshot'],map_location='cpu',weights_only=False);owner=a['meta']['frozen_owner'];spec=a['spec'];raws=a['actual_raw_prefix'];frl_v3_task.RISK_COEFFICIENT=0.
  else:
   a=read(P/row['episode_path']);owner=row['policy'];spec=row['spec'];raws=[d['raw'] for d in a['decisions'][:row['decision_index']]];frl_v3_task.RISK_COEFFICIENT=.33 if owner.startswith('v3') else 0.
  torch.manual_seed(56017);np.random.seed(56017);task.reset(spec,record=True,label=row['state_id'])
  for raw in raws:task.step(torch.tensor(raw,device='cuda',dtype=torch.float32).reshape(1,-1))
  x=xnow()
  if row['source_kind']=='p1_snapshot':
   assert torch.equal(x.cpu(),a['input']);assert torch.equal(task.last_command.cpu(),a['previous_z'])
   integ=np.empty(mujoco.mj_stateSize(e.sim.model,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(e.sim.model,e.sim.data,integ,mujoco.mjtState.mjSTATE_INTEGRATION);assert np.array_equal(integ,a['state']['integration'])
  else:
   decision=a['decisions'][row['decision_index']];assert torch.equal(x[:,1639:].cpu(),torch.tensor(decision['obs'])[None]);assert torch.equal(nets[owner].distribution(x).mean.cpu(),torch.tensor(decision['raw'])[None]);assert task.prefix_data['audit']['sha256']==row['source_state_sha256']
   original=[r for r in a['physics'] if r['force_time']<float(e.tether.time)-1e-8]
   assert len(original)==len(task.physics)
   for p,q in zip(original,task.physics):
    for key in ('qpos','qvel','force','anchor','support','feet_xyz'):assert p[key]==q[key],('archived physics mismatch',key,row['state_id'])
  capture(row,owner,a)
 write(out/'model-identities.json',identities)
 search.native(P,read,sha)
 return search.finish()
def main():
 global ARGS
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--layer',choices=['S1'],required=True);p.add_argument('--reuse',action='append',default=[]);ARGS=p.parse_args();ARGS.runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain']);ARGS.output.mkdir(exist_ok=False,parents=True)
 write(ARGS.output/'launch.json',dict(runtime=ARGS.runtime,pid=os.getpid(),started=time.time(),training=False,stage=ARGS.layer))
 os.environ.update(CLEAN_KIND='direct',P4000_CANONICAL_ROW_ACTOR='1',BFM_RECORD_NOMINAL_DIAGNOSTIC='0',BFM_RECOVERY_OUTPUT=str(ARGS.output));os.chdir(ARGS.output)
 import m10_stage,m10_physics
 m10_physics.OUT=ARGS.output;m10_stage.run_study=study
 try:m10_physics.main()
 except BaseException as exc:
  import traceback
  write(ARGS.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
