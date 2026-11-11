"""Issue60 L0 only: archive recovery, frozen timing probes, no training/search stage."""
import argparse,copy,gzip,json,os,sys,time,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ST=ROOT.parent
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
 def xnow():return torch.cat([task.student_features(),task.features()],-1)
 def capture(meta,owner,archive=None):
  x=xnow();snap=TrainingBranch(task);t0=time.monotonic();pp=portable(task)
  dst=out/(meta['state_id']+'.pt');torch.save({'state':pp,'spec':copy.deepcopy(task.spec),'actual_raw_prefix':[r['raw'] for r in task.decision_rows],'input':x.cpu(),'previous_z':task.last_command.cpu(),'meta':meta,'fingerprint':snap.fingerprint,'runtime':ARGS.runtime},dst);savewall=time.monotonic()-t0
  t0=time.monotonic();loaded=torch.load(dst,map_location='cpu',weights_only=False);readwall=time.monotonic()-t0
  assert torch.equal(loaded['input'],x.cpu())
  snaps=[]
  for _ in range(3):
   t0=time.monotonic();snap.restore();snaps.append(time.monotonic()-t0);assert snap.digest()==snap.fingerprint
  row={**meta,'state_snapshot':str(dst.relative_to(ST)),'state_snapshot_sha256':sha(dst),'restored_fingerprint':snap.fingerprint,'model_sha256':identities[owner]['sha256'] if owner in identities else c['model_sha256'],'source_policy':owner,'actual_runtime_sha':ARGS.runtime,'time_actual':float(e.tether.time),'input_dim':x.shape[-1],'restricted_dim':task.student_features().shape[-1],'save_seconds':savewall,'read_seconds':readwall,'snapshot_bytes':dst.stat().st_size,'restore_seconds':snaps,'restore_verified':True}
  results.append(row);write(out/'state-manifest.json',results)
  print('L0_STATE',row['state_id'],row['time_actual'],flush=True)
  if meta['state_id'] not in ['P1-1','P1-13'] and not (meta['role']=='new_tail' and meta['phase']=='sustained' and meta['parent_id'].startswith('5c67db5a')):return
  def execute(horizon,proposal='feedback'):
   start=len(task.physics);cc=len(task.controls);stamp=float(e.tether.time);inputs=[];raws=[]
   hold=task.last_command.clone()/16
   for _ in range(round(horizon*10)):
    if task.done:break
    x=xnow();raw=nets[owner].distribution(x).mean if proposal=='feedback' else hold
    inputs.append(x.cpu().clone());raws.append(raw.cpu().clone());task.step(raw)
   ph=copy.deepcopy(task.physics[start:]);ctrl=copy.deepcopy(task.controls[cc:]);return {'physics':ph,'controls':ctrl,'inputs':inputs,'raws':raws,'scores':score(ph,ctrl,stamp),'state_after':portable(task),'actual_seconds':len(ph)*.005}
  for horizon in (1.,3.):
   pairs=[]
   for rep in range(2):
    snap.restore();st=time.monotonic();r=execute(horizon);torch.cuda.synchronize();wall=time.monotonic()-st;pairs.append(r)
    dst=out/(meta['state_id']+f'-timing-{horizon}-{rep}.pt');ts=time.monotonic();torch.save(r,dst);io=time.monotonic()-ts
    timings.append({'state_id':meta['state_id'],'horizon':horizon,'rep':rep,'candidate_wall_s':wall,'save_wall_s':io,'bytes':dst.stat().st_size,'actual_physics_seconds':r['actual_seconds'],'information_tier':'O-F','purpose':'frozen timing/recovery control only; exact future scenario used; no L1 candidate search'})
   # All captured physical, control and RNG/history tensors must agree; no tolerance relaxation.
   assert tensorsame(pairs[0],pairs[1]),'repeat branch not bitwise identical'
  snap.restore();start=time.monotonic();ranking=[]
  for proposal in ('feedback','hold_last_z'):
   snap.restore();r=execute(1.,proposal);ranking.append((r['scores']['old_discounted_return'],proposal))
  chosen=max(ranking)[1];snap.restore();raw=nets[owner].distribution(xnow()).mean if chosen=='feedback' else task.last_command/16;task.step(raw);torch.cuda.synchronize();wall=time.monotonic()-start
  queries.append(dict(state_parent_id=meta['parent_id'],state_snapshot_sha=row['state_snapshot_sha256'],actual_runtime_sha=ARGS.runtime,information_tier='O-F',actor_observation_schema='2727D privileged branch; 1639D restricted prefix',candidate_generator_inputs=['current real obs','current last-z'],partner_predictor_version='original scenario exact future: ORACLE',partner_predictor_inputs=['full current spec and exact future program'],score_features=['true future contact force','anchor error','physical quality'],simulator_state_origin='complete actual exposed state',future_anchor_accessible=True,future_anchor_consumed=True,anchor_error_used_in_selection=True,selection_horizon=1.,physical_eval_horizon=.1,query_wall_seconds=wall,number_of_simulated_candidates=2,selected_candidate_id=chosen,purpose='L0 timing only',C_S_eligible=False))
  snap.restore();write(out/'timings.json',timings);write(out/'timing-query-permissions.json',queries)
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
 # Original no-contact source records: execute exact archived legal commands, no new prompts.
 N=P/'experiments/fb-native-audit-20261110/runs/native-inventory-01'
 for kind in ('standing','arm_ready_goal','forward_motion'):
  a=read(N/(kind+'-actual.json.gz'));torch.manual_seed(56017);np.random.seed(56017);frl_v3_task.RISK_COEFFICIENT=0.;task.reset(a['spec'],record=True,label='native-'+kind)
  targets=a['targets'];stop_at=6.;j=0
  while float(e.tether.time)<stop_at-1e-8:
   target=targets[j];assert abs(float(e.tether.time)-target['time'])<1e-7;task.step(torch.tensor(target['z'],device='cuda',dtype=torch.float32).reshape(1,-1)/16);j+=1
  original=[r for r in a['physics'] if r['force_time']<float(e.tether.time)-1e-8];assert len(original)==len(task.physics)
  for p,q in zip(original,task.physics):
   for key in ('qpos','qvel','force','support'):assert p[key]==q[key],('native mismatch',kind,key)
  parent='native-'+kind+'-'+sha(N/(kind+'-actual.json.gz'))[:16]
  meta=dict(state_id='native-'+kind,parent_id=parent,role='source_positive',phase='sustain',exposure='source-positive',archive_path=str((N/(kind+'-actual.json.gz')).relative_to(P)),archive_sha256=sha(N/(kind+'-actual.json.gz')),scope='no-contact behavior only; forward quality limitations retained')
  capture(meta,'native_'+kind)
 # Archived common-prefix checks, separate from controller's information permission.
 checked=set()
 for row in plan['states']:
  if row['source_kind']!='recorded_episode' or 'stop_time' not in row['spec'] or row['parent_id'] in checked:continue
  path=P/row['episode_path'];other=path.with_name(row['parent_id'][:12]+'-continue.json.gz')
  if not other.exists():continue
  a,b=read(path),read(other);cut=row['spec']['stop_time'];pa=[r for r in a['physics'] if r['force_time']<cut-1e-7];pb=[r for r in b['physics'] if r['force_time']<cut-1e-7];assert pa==pb,'archived stop/continue prefix differs'
  prefixchecks.append({'parent_id':row['parent_id'],'pre_divergence_physics_ticks':len(pa),'exact':True,'scope':'archive prefix physics; no future read by runtime actor claim inferred'});checked.add(row['parent_id'])
 write(out/'common-prefix-checks.json',prefixchecks);write(out/'model-identities.json',identities)
 write(out/'checks.json',dict(status='completed',states=len(results),parents=len(set(r['parent_id'] for r in results)),all_replayed_states_verified=True,benchmark_repeated_trajectories_bitwise=True,backend_setup_s=backend_wall,formal_RL_decisions=0,optimizer_updates=0,L1_searches=0,L2_runs=0,L3_runs=0,alias='preregistered only; not constructed or control-relevant',source_inventory='3 original real source prefixes reproduced; no new native capability search'))
 return read(out/'checks.json')
def main():
 global ARGS
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);ARGS=p.parse_args();ARGS.runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain']);ARGS.output.mkdir(exist_ok=False,parents=True)
 write(ARGS.output/'launch.json',dict(runtime=ARGS.runtime,pid=os.getpid(),started=time.time(),training=False,stage='L0'))
 os.environ.update(CLEAN_KIND='direct',P4000_CANONICAL_ROW_ACTOR='1',BFM_RECORD_NOMINAL_DIAGNOSTIC='0',BFM_RECOVERY_OUTPUT=str(ARGS.output));os.chdir(ARGS.output)
 import m10_stage,m10_physics
 m10_physics.OUT=ARGS.output;m10_stage.run_study=study
 try:m10_physics.main()
 except BaseException as exc:
  import traceback
  write(ARGS.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
