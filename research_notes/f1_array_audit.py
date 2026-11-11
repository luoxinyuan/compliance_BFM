"""Frozen per-array causal interventions; no physics/model/optimizer changes."""
import copy,json,time,hashlib
from pathlib import Path
import f1_l0 as base
import numpy as np,torch,mujoco
from diagnostic_frozen import portable
from dual_track_runtime import TrainingBranch
from frl_v3_counterfactual import tensorsame
from frl_v3_risk import score
from f1_pointer_layout import slot,null_arrays,struct_without,canonical_before
P=base.P; S=base.ST
NAMES=['iefc_frictionloss','efc_AR_rownnz','map_efc2iefc','iefc_D','iefc_J_rownnz','iefc_aref','iefc_J_rowsuper','efc_island','dof_island','iefc_state','iefc_JT_colind','iefc_id','iefc_JT','map_iefc2efc','iefc_J_colind','iefc_force','efc_AR_rowadr','iefc_R','iefc_J_rowadr','iLD','map_idof2dof','iefc_J','iM_colind','iefc_type','map_dof2idof']
def digest(x):
 h=hashlib.sha256()
 def add(v):
  if torch.is_tensor(v):add(v.detach().cpu().numpy())
  elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
  elif isinstance(v,dict):
   for k in sorted(v):h.update(str(k).encode());add(v[k])
  elif isinstance(v,(list,tuple)):
   for q in v:add(q)
  else:h.update(repr(v).encode())
 add(x);return h.hexdigest()
def without(x,names):
 y=dict(x);y['mjdata_arrays']={k:v for k,v in x['mjdata_arrays'].items() if k not in names};return y

def study(e,m,c,out):
 from native_env import install
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 from frl_v2_high import build
 import frl_v3_task
 install();speed(e);torch.set_num_threads(1);torch.set_grad_enabled(False)
 (out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix');frl_v3_task.configure(0.)
 t=frl_v3_task.Task(e,m,c,5801);nets={};rows=[];metas=[]
 for name,i in [('BC500',0),('B64',64),('B128',128)]:
  n=build().eval().requires_grad_(False);n.load_state_dict(torch.load(base.OLD/f'runs/frl-formal-B-01/candidate-{i}.pt',map_location='cpu',weights_only=False)['model']);nets[name]=n
 archives=[torch.load(P/'experiments/f1-l0-20261111/runs/l0-01'/f'P1-1-timing-1.0-{i}.pt',map_location='cpu',weights_only=False)['state_after']['mjdata_arrays'] for i in (0,1)]
 for stateid in ('P1-1','P1-13','native-standing'):
  torch.manual_seed(56017);np.random.seed(56017)
  if stateid.startswith('P1'):
   idx=int(stateid.split('-')[1]);a=torch.load(base.V/f'runs/counterfactual-02/state-{idx:02d}.pt',map_location='cpu',weights_only=False);owner=a['meta']['frozen_owner'];t.reset(a['spec'],record=True,label=stateid)
   for raw in a['actual_raw_prefix']:t.step(torch.tensor(raw,device='cuda',dtype=torch.float32).reshape(1,-1))
   assert torch.equal(torch.cat([t.student_features(),t.features()],-1).cpu(),a['input'])
   assert torch.equal(t.last_command.cpu(),a['previous_z'])
  else:
   a=base.read(P/'experiments/fb-native-audit-20261110/runs/native-inventory-01/standing-actual.json.gz');owner='native';t.reset(a['spec'],record=True,label=stateid)
   for target in a['targets']:
    if e.tether.time>=6.-1e-8:break
    assert abs(e.tether.time-target['time'])<1e-7
    t.step(torch.tensor(target['z'],device='cuda',dtype=torch.float32).reshape(1,-1)/16)
  d=e.sim.data;model=e.sim.model;snap=TrainingBranch(t);initial=portable(t)
  nulls=null_arrays(d);assert all(slot(d,k).value is None for k in NAMES),('non-null audited field',stateid)
  counters={k:int(getattr(d,k)) for k in ('ncon','nefc','nisland','nidof','nJ','nA') if hasattr(d,k)}
  conf={k:int(getattr(model.opt,k)) for k in ('solver','jacobian','cone','integrator','enableflags','disableflags')}
  conf.update(nv=int(model.nv),nC=int(model.nC),mujoco=mujoco.__version__,is_sparse=bool(mujoco.mj_isSparse(model)),is_dual=bool(mujoco.mj_isDual(model)))
  meta=dict(state_id=stateid,time=float(e.tether.time),owner=owner,counters=counters,config=conf,null_arrays=nulls,c_pointer_null={k:slot(d,k).value is None for k in NAMES},force=np.asarray(e.tether.force).tolist(),array_shapes={k:list(getattr(d,k).shape) for k in NAMES})
  metas.append(meta);base.write(out/'state-config.json',metas);torch.save(initial,out/(stateid+'-initial.pt'))
  def execute():
   p=len(t.physics);q=len(t.controls);at=e.tether.time;inputs=[];raws=[];counter=[]
   for j in range(30):
    assert not t.done,('cannot complete3s',stateid,j,t.boundary)
    x=torch.cat([t.student_features(),t.features()],-1);raw=nets[owner].distribution(x).mean if owner!='native' else e.standing/16
    inputs.append(x.cpu().clone());raws.append(raw.cpu().clone());t.step(raw)
    counter.append({k:int(getattr(d,k)) for k in counters})
   ph=copy.deepcopy(t.physics[p:]);ctrl=copy.deepcopy(t.controls[q:]);return dict(physics=ph,controls=ctrl,inputs=inputs,raws=raws,scores=score(ph,ctrl,at),counters=counter,actual_s=len(ph)*.005)
  snap.restore();st=time.monotonic();ref=execute();wall=time.monotonic()-st;assert ref['actual_s']>=3.-1e-8
  torch.save(ref,out/(stateid+'-reference.pt'));base.write(out/(stateid+'-reference-cost.json'),dict(wall_s=wall,actual_s=ref['actual_s'],digest=digest(ref)))
  for ki,names in enumerate([[k] for k in NAMES]+[NAMES]):
   for mode in ('zero','archive_random'):
    snap.restore();before=portable(t);nulls=null_arrays(d);struct_before=struct_without(d,names);changed=[];gen=np.random.default_rng(601100+ki);buffers=[]
    assert all(slot(d,k).value is None for k in NAMES)
    for k in names:
     view=getattr(d,k);old=view.copy();buffer=np.empty(max(view.size,65536),dtype=view.dtype);buffers.append(buffer);arr=buffer[:view.size].reshape(view.shape)
     if mode=='zero':arr[...] = 0
     else:
      pool=np.concatenate([v[k].ravel() for v in archives]+[initial['mjdata_arrays'][k].ravel()]);pool=pool[np.isfinite(pool)]
      assert pool.size
      arr[...] = gen.choice(pool,size=arr.shape)
     slot(d,k).value=buffer.ctypes.data
     assert np.shares_memory(arr,getattr(d,k)),('injected field not backed',k)
     changed.append(dict(array=k,injected_pointer=True,count=int(np.count_nonzero(arr!=old)),capacity=int(arr.size),finite=bool(np.all(np.isfinite(arr)))))
    after=portable(t)
    assert struct_before==struct_without(d,names),('C struct outside pointer fields changed',stateid,names)
    assert digest(canonical_before(before,nulls))==digest(canonical_before(after,nulls)),('valid non-target state changed',stateid,names)
    different=names;aliases={k:[] for k in names}
    st=time.monotonic();got=execute();wall=time.monotonic()-st;eq=tensorsame(ref,got)
    dest=out/(stateid+'-'+str(ki)+'-'+mode+'.pt');torch.save(got,dest)
    assert all(slot(d,k).value is None for k in NAMES),('engine did not clear inactive pointers',stateid,names)
    row=dict(state_id=stateid,arrays=names,mode=mode,changes=changed,other_state_byte_equal=True,c_struct_non_target_byte_equal=True,all_25_pointers_null_after_rollout=True,intervention='NULL pointer replaced with owned zero/random scratch; not Python phantom array',actual_changed_arrays=different,memory_aliases=aliases,physics_equal=tensorsame(ref['physics'],got['physics']),controls_equal=tensorsame(ref['controls'],got['controls']),score_equal=tensorsame(ref['scores'],got['scores']),all_output_equal=eq,actual_s=got['actual_s'],wall_s=wall,path=dest.name,sha256=base.sha(dest))
    rows.append(row);base.write(out/'interventions.json',rows)
    print('ARRAY',stateid,ki,mode,eq,round(wall,3),flush=True)
    if not eq:
     base.write(out/'effective-state-detected.json',row);raise AssertionError(('array intervention changed trajectory; stop item',stateid,names,mode))
  snap.restore()
 base.write(out/'checks.json',dict(status='passed',states=3,individual_arms=150,joint_arms=6,seconds_per_arm=3,physics_ticks=sum(round(r['actual_s']/.005) for r in rows)+1800,training=0,scope='same current model config only; not blanket MuJoCo field exclusion',runtime=base.ARGS.runtime))
 return base.read(out/'checks.json')
if __name__=='__main__':base.study=study;base.main()
