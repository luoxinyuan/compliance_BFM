"""H0 native sustained commands, full physical state branches; no actor optimization."""
import argparse,copy,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
from research.interaction_fb.h0_common import *

def study(e,m,c,out):
 import torch,mujoco
 from native_env import install
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 from dual_track_task import Task,global_rng
 from dual_track_runtime import TrainingBranch
 from diagnostic_frozen import portable
 from diagnostic_fb_resume import tree_digest
 from f1_mjdata_contract import canonical,metadata,exact,install_restore_guard
 from research.interaction_fb.journal import dump,read
 from research.interaction_fb.gate_v11 import metrics
 from locomotion_rollout import spherical
 from m9_cyclic import sample
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 install();speed(e);(out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix');install_restore_guard()
 assert c['model_sha256']==BFM
 lock=json.loads((S/'prep/method-lock.json').read_text());task=Task(e,m,c,660201)
 base_digest=tree_digest(m._actor.state_dict());obs_digest=tree_digest(m._obs_normalizer.state_dict()) if hasattr(m,'_obs_normalizer') else None
 def seed(v):
  import random
  random.seed(v);np.random.seed(v);torch.manual_seed(v);torch.cuda.manual_seed_all(v)
 def source(pr):
  seed(pr['seed']);task.reset(pr['spec'],record=True,label=pr['parent'][:12])
  while float(e.tether.time)<pr['boundary_s']-1e-7:task.step(e.standing/16)
  assert abs(float(e.tether.time)-pr['boundary_s'])<1e-6 and not task.boundary
  task.prepare();h=task.student_features()[0].cpu().clone();assert h.shape==(1639,)
  # Hidden target/phase invariance is tested on the already prepared public observation.
  from research.interaction_fb.schema import deploy
  ob=task.extended_observation();ob.update(_previous_z=task.last_command,_target_z=task.last_command,_slot=torch.zeros(1,1,device='cuda'))
  hidden={k:v.clone() for k,v in ob.items()};hidden['privileged_state']+=100.;hidden['contact_privileged_v1'][:,9:15]+=100.;hidden['contact_privileged_v1'][:,18:22]+=100.
  assert torch.equal(deploy(ob),deploy(hidden))
  last=task.last_command.clone();ref=task.refs['straight'];phase=min([j/8 for j in range(8)],key=lambda p:float((sample(ref,p)-last).square().sum()))
  plans=[]
  for name in ('standing','goal_arm5','goal_arm15','motion'):
   zs=[]
   for j in range(20):
    if name=='standing':z=e.standing
    elif name!='motion':z=task.native_z[name]
    else:
     a=float(np.interp(j/29,[0,.5,1],[.25,.5,.75]));z=spherical(e.standing,sample(ref,phase+j*.1*.5/ref['period']),a)
    assert abs(float(z.norm())-16)<1e-4 and torch.isfinite(z).all();zs.append(z.detach().cpu())
   plans.append(torch.cat(zs))
  snap=TrainingBranch(task);pp=portable(task);md=metadata(e.sim.model,e.sim.data);before=canonical(pp,md)
  source_meta=dump(out/(pr['parent'][:12]+'-source.ptz'),dict(physical=pp,mj_metadata=md,prefix_data=task.prefix_data,actual_prefix=[d['raw'] for d in task.decision_rows],spec=task.spec,rng=global_rng(),h=h,plans=plans,previous_z=last.cpu(),runtime=RUNTIME,bfm=BFM))
  return h,plans,snap,before,source_meta
 def execute(plan):
  pi=len(task.physics);ci=len(task.controls);inputs=[];previous=[];raws=[]
  for z in plan:
   inputs.append(task.student_features().cpu());previous.append(task.last_command.cpu());raws.append(z[None].cpu()/16)
   task.step(z[None].to(e.standing.device)/16)
   if task.boundary:break
  ph=copy.deepcopy(task.physics[pi:]);co=copy.deepcopy(task.controls[ci:]);af=error_audit(ph,task.spec['h0_stiffness'])
  return dict(physics=ph,controls=co,inputs=inputs,previous_z=previous,actual_raws=raws,actual_target_plan=plan,formula=af,boundary=task.boundary,complete=len(ph)==400,quality=metrics(ph),y=result(ph).tolist() if len(ph)==400 else None,identity='actual_frozen_command_execution_not_PPO')
 results=[]
 if MODE in ('qualify','qualify-restore'):
  pr=lock['fit_new'][0];h,plans,snap,before,sm=source(pr)
  if MODE=='qualify':
   a=execute(plans[0]);assert a['complete'] and not a['boundary'];snap.restore();assert exact(canonical(portable(task),metadata(e.sim.model,e.sim.data)),before)
   b=execute(plans[0]);assert exact(a,b),'same-state deterministic command not exact'
   # Real K80 preservation: default source retained, compare exact original K80 expression/function on every audited row.
   meta=dump(out/'qualification.ptz',dict(source=sm,source_state=before,branch=a,model_digest=base_digest,plans=plans,h=h,rng=global_rng()))
   write(out/'qualification.json',dict(status='SAME_PROCESS_RESTORE_EXACT',archive=meta,source_archive=sm,formula=a['formula'],quality=a['quality']))
  else:
   q=S/'runs/qualification-01';aa=json.loads((q/'qualification.json').read_text());old=read(q/'qualification.ptz',aa['archive'])
   assert exact(before,old['source_state']),'new-process full source restore differs'
   a=execute(plans[0]);assert exact(a,old['branch']),'new-process actual control/physics differs'
   write(out/'qualification.json',dict(status='NEW_PROCESS_RESTORE_EXACT',full_source_exact=True,full_control_physics_score_exact=True,formula=a['formula'],quality=a['quality']))
  return dict(status='H0_QUALIFICATION_COMPLETE',mode=MODE)
 if MODE=='fit':
  assert json.loads((S/'runs/qualification-restore-02/qualification.json').read_text())['status']=='NEW_PROCESS_RESTORE_EXACT'
  for pr in lock['fit_new']:
   h,plans,snap,before,sm=source(pr)
   for i,plan in enumerate(plans):
    snap.restore();assert exact(canonical(portable(task),metadata(e.sim.model,e.sim.data)),before)
    r=execute(plan);am=dump(out/f"{pr['parent'][:12]}-command{i}.ptz",r)
    rec=dict(parent=pr['parent'],source_parent=pr['source_parent'],partner_parent=pr['partner_parent'],K=pr['K'],command=i,h=h,plan=plan,y=torch.tensor(r['y']) if r['y'] is not None else None,quality=r['quality'],boundary=r['boundary'],source_archive=sm,archive=am,phase=pr['phase'],identity='actual_fit_branch')
    results.append(rec);meta=dump(out/'fit-records.ptz',results);write(out/'progress.json',dict(completed=len(results),archive=meta))
    print('H0_FIT_COMMITTED',pr['parent'][:12],i,r['complete'],r['boundary'],flush=True)
    if r['boundary']:return dict(status='HARD_PHYSICS_STOP',completed=len(results))
  write(out/'fit.json',dict(status='FIT_COMPLETE',branches=len(results),archive=meta));return dict(status='FIT_COMPLETE')
 if MODE=='evaluation':
  from research.interaction_fb.h0_learn import load_models,predict,select
  models,packet=load_models();mdig=tree_digest(models.state_dict())
  for pr in lock['evaluation']:
   h,plans,snap,before,sm=source(pr)
   force=np.array(e.tether.force);direction=force/np.linalg.norm(force) if np.linalg.norm(force)>1e-6 else np.array([1.,0.,0.]);local=np.array(e.sim.data.xmat[1]).reshape(3,3).T@direction
   Y,preds=predict(models,packet,h,plans);choices={};targets={}
   for method in ('E','I'):
    for goal in ('HOLD','FOLLOW'):
     g=target(goal=='FOLLOW',local);choices[method+'-'+goal]=select(models,packet,preds,g,method);targets[goal]=g.tolist()
   # Immutable prediction/choice before actual future, and shuffle uses no new simulator branch.
   pm=dump(out/(pr['parent'][:12]+'-predictions.ptz'),dict(predictions=Y,preds=preds,choices=choices,public_goals=targets,h=h,plans=plans,source=sm,models=packet['manifest']))
   rows=[]
   for method in ('E','I'):
    for goal in ('HOLD','FOLLOW'):
     snap.restore();assert exact(canonical(portable(task),metadata(e.sim.model,e.sim.data)),before)
     i=choices[method+'-'+goal];r=execute(plans[i]);am=dump(out/f"{pr['parent'][:12]}-{method}-{goal}.ptz",r)
     ph=r['physics'];progress=float((np.array(ph[-1]['wrist'])-ph[0]['wrist'])@direction)
     ac=[p for p in ph if p['active']];err=float(np.sqrt(np.mean([p['error']**2 for p in ac]))) if ac else None;force_rms=float(np.sqrt(np.mean([np.dot(p['force'],p['force']) for p in ac]))) if ac else 0.
     rr=dict(method=method,goal=goal,command=i,progress_m=progress,error_rms_m=err,force_rms_N=force_rms,quality=r['quality'],boundary=r['boundary'],y=r['y'],archive=am);rows.append(rr)
     print('H0_EVAL_COMMITTED',pr['parent'][:12],method,goal,i,progress,flush=True)
     if r['boundary']:
      results.append(dict(parent=pr['parent'],K=pr['K'],rows=rows,predictions=pm));write(out/'evaluation.json',dict(status='HARD_PHYSICS_STOP',results=results));return dict(status='HARD_PHYSICS_STOP')
   effects=[]
   for method in ('E','I'):
    a=next(r for r in rows if r['method']==method and r['goal']=='FOLLOW');b=next(r for r in rows if r['method']==method and r['goal']=='HOLD');q=a['quality']
    safety=all(r['quality']['max_joint_excess_rad']<=.02 and r['quality']['unsupported_ticks']==0 and r['quality']['slip_peak']<=.45 and r['quality']['yaw_excursion_deg']<=lock['quality']['yaw_excursion_deg'] for r in (a,b))
    delta=a['progress_m']-b['progress_m'];error_ok=a['error_rms_m']<=b['error_rms_m']+.005
    effects.append(dict(method=method,follow_minus_hold_m=delta,error_worsening_m=a['error_rms_m']-b['error_rms_m'],quality_pass=safety,effect_pass=delta>=.020 and error_ok and safety,goal_shuffle_effect_m=-delta,goal_changes_command=a['command']!=b['command']))
   results.append(dict(parent=pr['parent'],K=pr['K'],phase=pr['phase'],rows=rows,effects=effects,predictions=pm));write(out/'evaluation.json',dict(status='EVALUATION_RUNNING',results=results))
  assert tree_digest(models.state_dict())==mdig and tree_digest(m._actor.state_dict())==base_digest
  write(out/'evaluation.json',dict(status='EVALUATION_COMPLETE',results=results));return dict(status='EVALUATION_COMPLETE')

def main():
 global MODE,RUNTIME
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['qualify','qualify-restore','fit','evaluation'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();MODE=a.mode
 import subprocess
 RUNTIME=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 from research.interaction_fb import benchmark
 import mujoco
 original=mujoco.mj_step
 prior=sum(json.loads(f.read_text())['reserved'] for f in (S/'runs').glob('*/physics-counter.json'));calls=[0]
 def guard(*args,**kwargs):
  calls[0]+=int(kwargs.get('nstep',1));assert prior+calls[0]<=45000,'H0_TOTAL_PHYSICS_BUDGET'
  return original(*args,**kwargs)
 mujoco.mj_step=guard;benchmark.study=study;sys.argv=[__file__,'--output',str(a.output),'--arm','A'];benchmark.main()
if __name__=='__main__':main()
