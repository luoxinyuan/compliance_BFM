"""Read-only archival identity and immutable H0 scope prelock. No CUDA or physics."""
import sys,copy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import torch,numpy as np,json
from research.interaction_fb.h0_common import *
from research.interaction_fb.journal import dump,read
torch.set_num_threads(1)
def main():
 t=time.monotonic();out=S/'prep';records=[]
 f1=P/'experiments/f1-l1-l2-20261111/runs/l1-merged';rows=json.loads((f1/'candidate-scores.json').read_text())
 parents=[]
 for r in rows:
  if r['parent_id'] not in parents:parents.append(r['parent_id'])
 for parent in parents[:3]:
  rr=next(r for r in rows if r['parent_id']==parent);sid=rr['state_id']
  ancestry=json.loads((P/'experiments/f1-l1-l2-20261111/prep/state-plan.json').read_text())
  ancestor=next(a for a in ancestry['states'] if a['state_id']==sid)
  if 'source_snapshot' in ancestor:
   original=torch.load(P/ancestor['source_snapshot'],map_location='cpu',weights_only=False);sp=original['spec']
  else:sp=ancestor['spec']
  for ci in (6,7,8,9):
   q=next(r for r in rows if r['state_id']==sid and r['candidate']==ci);p=f1/q['artifact'];assert sha(p)==q['artifact_sha256']
   a=torch.load(p,map_location='cpu',weights_only=False);assert len(a['physics'])==600 and len(a['inputs'])==30
   ph=a['physics'][:400];co=a['controls'][:100];assert len(co)==100
   assert all(abs(np.linalg.norm(z)-16)<1e-4 for z in a['target_z'][:20])
   af=error_audit(ph,80)
   records.append(dict(parent=parent,source_parent=sp['source_parent_id'],partner_parent=sp['partner_parent_id'],state=sid,K=80,command=ci-6,h=a['inputs'][0][0,:1639].float(),plan=torch.cat(a['target_z'][:20]).reshape(20,256).float(),y=torch.tensor(result(ph)),original_artifact=str(p),original_sha=sha(p),quality=q['metrics'],force_formula=af,identity='archived_actual_candidate_execution_fit_only',phase=q['phase']))
 # New source + partner ancestry is taken only from already train-designated groups.
 data=json.loads((ROOT/'research_notes/r2_splits/train.json').read_text());new=[];evals=[]
 for split,indices in [('fit',[13,14,15]),('evaluation',[16,17,18,19])]:
  for k,index in enumerate(indices):
   g=copy.deepcopy(data['parents'][index]);sp=g['specs'][0];K=40 if split=='fit' else [80,40,80,40][k]
   sp.update(h0_stiffness=K,split='train',demand_family=g['family']);sp['end']=max(sp['end'],sp['onset']+3.)
   item=dict(parent=g['parent_id'],source_parent=g['source_parent_id'],partner_parent=g['partner_parent_id'],spec=sp,original_train_index=index,K=K,boundary_s=round((sp['onset']+.5)*10)/10,seed=660100+index,phase='entry' if g['family']=='near' else 'moving')
   (new if split=='fit' else evals).append(item)
 assert len(set([r['parent'] for r in records]+[p['parent'] for p in new]))==6
 for key in ('parent','source_parent','partner_parent'):
  assert len(set(p[key] for p in new+evals))==len(new+evals)
  assert len(set(r[key] for r in records))==3
  assert set(p[key] for p in new+evals).isdisjoint({r[key] for r in records})
 # Commands identical in meaning to archived F1 indices6..9; motion is its actual first2s of3s plan.
 lock=dict(version='H0_METHOD_LOCK_V1',approved_issue=66,seed=660201,groups=1024,batch=24,lr=.001,hidden=64,latent=16,variance_weight=.1,covariance_weight=.1,fit_new=new,evaluation=evals,fit_archived_parents=parents[:3],
  commands=['standing','goal_arm5','goal_arm15','F1_motion9_first2s_of3s'],h_dim=1639,command_dim=5120,outcome_dim=24,outcome_scales=SCALES.tolist(),outcome_fields=FIELDS,
  predictor='current restricted h + full proposed20x256 legal target plan; no hidden partner/phase/clock',
  goals='FOLLOW desired wrist+10cm along currently measured force (else world+X); HOLD zero wrist displacement; public target, unchanged partner',
  execution='deterministic100ms commands; original five spherical interpolation substeps;2s real window; no random actions; no unexecuted labels',
  mapping='P-E finite argmin normalized raw predicted-result distance; P-I finite argmin learned predicted successor embedding vs B(public target); identical four proposals; T has0 fitted parameters',
  losses='E raw future MSE; I decoded-F future MSE + F/B latent MSE + B autoencoding MSE + .1 variance + .1 offdiagonal covariance; no scalar-Q/GAN/CPR/Actor update',
  candidate_plan='motion phase nearest8-grid to own actual last-z; alpha interpolation [.25,.5,.75] on original3s timeline, rate.5; cannot read another trajectory phase',
  quality=dict(joint_excess_max=.02,unsupported_ticks=0,slip_peak=.45,yaw_excursion_deg=20.,goal_effect_m=.020,follow_error_worsening_max_m=.005),
  budget=dict(job_s=5400,output_bytes=2*2**30,physics_ticks=45000,phases={'prep':450,'fit':1500,'learn':600,'eval':1800,'closeout':450,'reserve':600}),
  data_permission='new sources are train-only; four eval groups never used for fit/checkpoint selection; no Dev/Sealed reads',goal_shuffle='frozen offline and paired deterministic replay equivalence where same selected command; no added physical branches',
  withholding='leave-one-fit-parent-out metrics via raw nearest predictor baseline, not refitting1024 heads; all formal weights fixed endpoint; four independent eval parents are primary prediction holdout')
 write(out/'method-lock.json',lock)
 meta=dump(out/'archived-fit.ptz',records);assert len(read(out/'archived-fit.ptz',meta))==12
 support=dict(status='ARCHIVED12_VERIFIED_NEW_K40_REQUIRED',archived_parent_count=3,archived_branches=12,archived_used_seconds=24,source_BFM=BFM,E3500_support='single standing zeroU-neighborhood; not multicommand causal labels; reused QA2 coverage audit only',F1_identity='same BFM/PD, actual executed complete plans and physics; only native/static/motion9, no learned old actor labels',K40_support=0,need_new_fit_branches=12,new_eval_branches=16,archive=meta,wall_s=time.monotonic()-t)
 write(out/'data-support.json',support);print(json.dumps(support),flush=True)
if __name__=='__main__':main()
