"""Issue57 full atomic FB packages and bounded interruption regression.

Sampling is stopped only after an actually executed native-control boundary.
No callbacks/closures are pickled; exact low-level prefix reconstruction creates
the graph, then all physical caches/history and RNG are checked/restored.
"""
import copy,hashlib,json,os,signal,time,uuid
from pathlib import Path
import numpy as np
import torch
from dual_track_fb_online import factory,Collector,run_updates,save_package,load_package,equal,cpu
from dual_track_task import global_rng,restore_global_rng
from dual_track_runtime import TrainingBranch
from dual_track_high import write,sha
from diagnostic_frozen import portable

def initial_noise_audit(m,contract,out):
 """Same archived input only, before update0; not a physical noise ablation."""
 import gzip
 old=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/dual-track-formal-20261109/runs/fb-1m')
 ep=json.load(gzip.open(old/'failures/fb-0.json.gz','rt'));rows=[]
 offset=torch.tensor(contract['q_PD_offset'],device='cuda');scale=torch.tensor(contract['q_PD_scale'],device='cuda')
 bound=contract['clip']/contract['normalize_ratio'];ankle=contract['dof_names'].index('left_ankle_roll_joint')
 for c in ep['control']:
  obs={k:torch.tensor(v,device='cuda',dtype=torch.float32) for k,v in c['bfm_obs'].items()}
  z=torch.tensor(c['executed_z'],device='cuda',dtype=torch.float32)
  mean=m.act(obs,z,mean=True);qpd=offset+mean.clamp(-bound,bound)*scale
  rows.append(dict(time=c['time'],mean_action_ankle=float(mean[0,ankle]),sample_action_ankle=c['action'][ankle],
     mean_qpd_ankle=float(qpd[0,ankle]),sample_qpd_ankle=c['q_PD'][ankle],
     same_state_action_max_delta=float((mean-torch.tensor(c['action'],device='cuda')).abs().max())))
 write(out/'old-preupdate-same-state-noise.json',dict(rows=rows,
   scope='episode0 ended before first FB update; frozen original actor reproduces initialization mean source, no physics counterfactual',
   postupdate_mean_not_recoverable=True))


def tree_digest(obj):
 h=hashlib.sha256()
 def visit(v):
  if torch.is_tensor(v):
   a=v.detach().cpu().contiguous();h.update(str((str(a.dtype),tuple(a.shape))).encode());h.update(a.numpy().tobytes())
  elif isinstance(v,np.ndarray):h.update(str((str(v.dtype),v.shape)).encode());h.update(v.tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
  elif isinstance(v,(list,tuple)):
   for x in v:visit(x)
  else:h.update(repr(v).encode())
 visit(obj);return h.hexdigest()

def fsync_file(path):
 with open(path,'rb') as f:os.fsync(f.fileno())

def publish(path,agent,c,updates,runtime,progress,reason,previous=None):
 """Immutable packages, hash manifest and atomic pointer; no in-place model overwrite."""
 assert not path.exists();pending=path.parent/('.pending-'+uuid.uuid4().hex);begun=time.monotonic()
 save_package(pending,agent,c,updates,runtime)
 extra=dict(progress=copy.deepcopy(progress),reason=reason,episode_rows=copy.deepcopy(c.episode_rows),
  numerical_backend=dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG')),
  task_totals={k:getattr(c.task,k) for k in ['total_prefill','total_steps','total_controls','total_episodes','source_physics_total']},
  source_ledger=copy.deepcopy(c.source_ledger),rng=global_rng(),task_rng=copy.deepcopy(c.task.rng.bit_generator.state),
  physical=cpu(portable(c.task)),actual_last_z=cpu(c.task.last_command),physics_rng=copy.deepcopy(c.task.physics_rng))
 torch.save(extra,pending/'continuation.pt')
 # Immutable identical files may share an inode. Never link mutable live tensors/replay.
 old=json.loads((previous/'manifest.json').read_text()) if previous else None
 for p in list(pending.rglob('*')):
  if not p.is_file() or p.name=='manifest.json':continue
  rel=str(p.relative_to(pending));h=sha(p)
  if old and old['completed_files'].get(rel)==h:
   p.unlink();os.link(previous/rel,p)
  fsync_file(p)
 manifest=json.loads((pending/'manifest.json').read_text())
 manifest.update(checkpoint_protocol='ISSUE57_ATOMIC_FULL_V2',reason=reason,progress=progress,
  complete=True,user_stop_immutable=reason=='user_stop',completed_files={str(p.relative_to(pending)):sha(p) for p in pending.rglob('*') if p.is_file() and p.name!='manifest.json'},
  save_total_s=time.monotonic()-begun,continuation_sha256=sha(pending/'continuation.pt'))
 write(pending/'manifest.json',manifest);fsync_file(pending/'manifest.json')
 fd=os.open(pending,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 os.rename(pending,path)
 fd=os.open(path.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 write(path.parent/'latest-complete.json',dict(package=path.name,manifest_sha256=sha(path/'manifest.json'),reason=reason,progress=progress))
 fsync_file(path.parent/'latest-complete.json')
 fd=os.open(path.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 return manifest

def restore(path,e,m,c):
 agent,col,meta=load_package(path,e,m,c,physical=True)
 assert meta['checkpoint_protocol']=='ISSUE57_ATOMIC_FULL_V2' and meta['complete']
 x=torch.load(path/'continuation.pt',map_location='cpu',weights_only=False)
 assert x['numerical_backend']==dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
 col.episode_rows=x['episode_rows'];col.source_ledger=x['source_ledger']
 for k,v in x['task_totals'].items():setattr(col.task,k,v)
 assert torch.equal(col.task.last_command.cpu(),x['actual_last_z'])
 # Exact actual-action reconstruction must match all saved solver/integration
 # arrays that affect evolution, not only qpos/qvel.
 d=col.task.e.sim.data;checked=[]
 for k in ['qpos','qvel','qacc','qacc_warmstart','qfrc_constraint','ctrl','xfrc_applied','efc_force','solver_niter']:
  expected=x['physical']['mjdata_arrays'][k];actual=getattr(d,k)
  assert np.array_equal(actual,expected),('New-process solver mismatch',k,float(np.max(np.abs(actual-expected))))
  checked.append(k)
 assert tree_digest(col.task.e.base.history_handler.history)==tree_digest(x['physical']['history'])
 col.task.physics_rng=copy.deepcopy(x['physics_rng']);restore_global_rng(x['rng']);col.task.rng.bit_generator.state=copy.deepcopy(x['task_rng'])
 meta['exact_solver_arrays_checked']=checked
 return agent,col,x['progress'],meta

class NormalStop:
 """Signal handlers set intent only. Saving occurs synchronously in the training loop."""
 def __init__(self,folder):
  self.folder=folder;self.requested=False;self.reason=None
  def request(signum,frame):self.requested=True;self.reason=f'signal:{signum}'
  signal.signal(signal.SIGTERM,request);signal.signal(signal.SIGINT,request)
 def pending(self):
  if (self.folder/'STOP_REQUEST.json').exists():self.requested=True;self.reason='request_file'
  return self.requested

def collect_to(c,target,stop,at=None):
 while c.transitions<target:
  if stop.pending():return False
  count=min(5,target-c.transitions)
  if at is not None:count=min(count,at-c.transitions)
  if count<=0:return False
  c.collect(count)
  if at is not None and c.transitions==at:return False
 return True

def honor_external_stop(stop,agent,c,updates,runtime,out):
 """An external request stops before further collection/update, with actual cursor."""
 if not stop.pending():return None
 progress=dict(collection_block_start=(c.transitions//1024)*1024,collection_block_target=(c.transitions//1024+1)*1024,
  collected_within_block=c.transitions%1024,completed_agent_updates=updates,validation=True,formal_training=False)
 path=out/f'user-stop-external-{c.transitions}-{updates}'
 publish(path,agent,c,updates,runtime,progress,'user_stop')
 return dict(status='stopped_by_request',package=str(path),actual_transitions=c.transitions,engineering_FB_updates=updates,formal_training=0)

def fingerprints(agent,c,updates,metrics=None):
 return dict(model=tree_digest(agent._model.state_dict()),optimizers=tree_digest(agent.optimizer_dict),
  zbuffer=tree_digest(dict(storage=agent.z_buffer._storage,idx=agent.z_buffer._idx,full=agent.z_buffer._is_full)),
  replay=tree_digest(c.replay.get_full_buffer()),rng=tree_digest(global_rng()),
  physical=TrainingBranch(c.task).fingerprint,transitions=c.transitions,updates=updates,
  coverage=c.coverage,metrics=metrics)

def execution_metadata(agent):
 return dict(module_training={n:v.training for n,v in agent._model.named_modules()},
  parameter_requires_grad={n:v.requires_grad for n,v in agent._model.named_parameters()},
  normalizer_order=list(agent._model._obs_normalizer._normalizers),
  input_filters={n:v.input_filter.cfg.model_dump() for n,v in agent._model.named_modules() if hasattr(v,'input_filter')},
  config=agent.cfg.model_dump())

def factory_rebuild(agent,c,out):
 """One controlled construction-path intervention; state and RNG remain exact."""
 r=global_rng();before=fingerprints(agent,c,1);oldmeta=execution_metadata(agent)
 fresh,_=factory(compile=False)
 fresh._model.load_state_dict(agent._model.state_dict(),strict=True)
 for n,sd in agent.optimizer_dict.items():getattr(fresh,n).load_state_dict(sd)
 fresh.z_buffer._storage.copy_(agent.z_buffer._storage);fresh.z_buffer._idx=agent.z_buffer._idx;fresh.z_buffer._is_full=agent.z_buffer._is_full
 c.proxy.policy_model=fresh._model;restore_global_rng(r)
 assert before==fingerprints(fresh,c,1)
 write(out/'construction-metadata.json',dict(loaded=oldmeta,factory=execution_metadata(fresh)))
 return fresh

def run(e,m,contract,out,args):
 start=time.monotonic();stop=NormalStop(out)
 if args.mode=='d2control':
  initial_noise_audit(m,contract,out)
  agent,init=factory(compile=False);c=Collector(e,m,contract,agent,2048);c.bootstrap_sources()
  collect_to(c,1024,stop)
  stopped=honor_external_stop(stop,agent,c,0,args.runtime,out)
  if stopped is not None:return stopped
  stat=run_updates(agent,c,0,1,compile=False);updates=1
  stopped=honor_external_stop(stop,agent,c,updates,args.runtime,out)
  if stopped is not None:return stopped
  progress=dict(collection_block_start=1024,collection_block_target=2048,collected_within_block=0,completed_agent_updates=updates,
    validation=True,formal_training=False)
  publish(out/'rolling-1024',agent,c,updates,args.runtime,progress,'rolling')
  # Both paths use the same administrative boundary at1037. Only one restarts.
  collect_to(c,2048,stop,at=1037)
  stopped=honor_external_stop(stop,agent,c,updates,args.runtime,out)
  if stopped is not None:return stopped
  write(out/'STOP_REQUEST.json',dict(reason='controlled interruption during collection',requested_at_transition=c.transitions))
  assert stop.pending();progress['collected_within_block']=13
  publish(out/'user-stop-1037',agent,c,updates,args.runtime,progress,'user_stop',previous=out/'rolling-1024')
  before=fingerprints(agent,c,updates);write(out/'stopped-fingerprints.json',before)
  # No process restart for the control. A new Stop object gives explicit permission
  # only for the regression continuation; it does not restart a formal stage.
  stop.requested=False;stop.pending=lambda:False
 elif args.mode in ('d2resume','d2resume_factory'):
  assert args.package is not None
  agent,c,progress,meta=restore(args.package,e,m,contract);updates=progress['completed_agent_updates']
  before=fingerprints(agent,c,updates);write(out/'restored-fingerprints.json',before)
  reference=json.loads((args.package.parent/'stopped-fingerprints.json').read_text())
  assert before==reference,('Full package readback mismatch',{k:(before[k],reference[k]) for k in before if before[k]!=reference[k]})
  if args.mode=='d2resume_factory':agent=factory_rebuild(agent,c,out)
 else:raise AssertionError(args.mode)
 replay_before=len(c.replay);collect_to(c,1064,stop)
 stopped=honor_external_stop(stop,agent,c,updates,args.runtime,out)
 if stopped is not None:return stopped
 after_collection=fingerprints(agent,c,updates);write(out/'after-collection-fingerprints.json',after_collection)
 stats=run_updates(agent,c,updates,1,compile=False);updates+=1
 stopped=honor_external_stop(stop,agent,c,updates,args.runtime,out)
 if stopped is not None:return stopped
 after=fingerprints(agent,c,updates,stats['last_metrics']);write(out/'after-update-fingerprints.json',after)
 progress=dict(collection_block_start=1024,collection_block_target=2048,collected_within_block=c.transitions-1024,
  completed_agent_updates=updates,validation=True,formal_training=False)
 previous=args.package.parent/'rolling-1064' if args.mode=='d2resume_factory' else args.package if args.mode=='d2resume' else out/'user-stop-1037'
 publish(out/'rolling-1064',agent,c,updates,args.runtime,progress,'rolling',previous=previous)
 if args.mode in ('d2resume','d2resume_factory'):
  reference_path=args.package.parent
  expected=json.loads((reference_path/'after-collection-fingerprints.json').read_text())
  assert after_collection==expected,('Interrupted collection differs from uninterrupted', {k:(after_collection[k],expected[k]) for k in expected if after_collection[k]!=expected[k]})
  expected=json.loads((reference_path/'after-update-fingerprints.json').read_text())
  assert after==expected,('Interrupted update differs from uninterrupted',{k:(after[k],expected[k]) for k in expected if after[k]!=expected[k]})
 return dict(status='completed',mode=args.mode,actual_new_transitions=1064 if args.mode=='d2control' else len(c.replay)-replay_before,
  engineering_FB_updates=2 if args.mode=='d2control' else 1,formal_training=0,wall_s=time.monotonic()-start,
  new_process_resume_exact=args.mode in ('d2resume','d2resume_factory'),save_protocol='atomic immutable full package V2')
