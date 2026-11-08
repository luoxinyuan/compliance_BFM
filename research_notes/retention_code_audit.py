"""Read-only targeted source/tensor audit. Does not modify implementation or artifacts."""
from pathlib import Path
import os,sys,json,time,hashlib,copy
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/ppo-retention-audit-20261109';R=S/'repo';O=P/'experiments/interactive-increment-20261108';Q=P/'experiments/interactive-contact-pilot-20261108';sys.path[:0]=[str(R),str(R/'research_notes')]
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import torch,numpy as np
torch.set_num_threads(1);out=S/'runs/C-readonly';out.mkdir(exist_ok=False);beg=time.monotonic();res={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
from retention_eval import MODELS
from clean_policy import Expert
ident=[]
for role,(path,digest) in MODELS.items():
 assert sha(path)==digest;ck=torch.load(path,map_location='cpu',weights_only=True);n=Expert(1639,'direct');n.load_state_dict(ck['state_dict'],strict=True)
 ident.append(dict(role=role,path=str(path),sha256=digest,normalizer_key_exact=all(torch.equal(getattr(n,k),ck['state_dict'][k]) for k in ['center','scale']),input_dim=ck['input_dim'],runtime=ck.get('runtime'),BFM_actor_not_in_checkpoint=not any('bfm' in k.lower() for k in ck['state_dict'])))
res['model_identity']=ident
# Verify actual index artifacts, rather than infer the mixing form from comments.
a=np.load(O/'runs/fit-01/sample-orders.npz');old=a['old_reference'];mix=a['interactive'];original=np.load(Q/'runs/bc-fit-01/matched-orders.npz')['old_data_only'];data=Q/'runs/final-analysis-01/corrected-new-labels.pt';new=torch.load(data,map_location='cpu',weights_only=True)['rows'];assert len(new)==34
assert np.array_equal(old,original) and np.array_equal(mix[:,45:],old[:,45:]);assert np.all(mix[:,:45]>=311) and np.all(mix[:,:45]<345)
groups=[np.array([j for j,r in enumerate(new) if r['time']<5.3 or 'normal72'in r['episode']]),np.array([j for j,r in enumerate(new) if 'normal72'not in r['episode']])]
group_draw=[int(np.isin(mix[:,:45]-311,g).sum()) for g in groups]
res['mixing']=dict(verified=True,shape=list(mix.shape),old_order_exact=True,unchanged_old_slots=83,replaced_slots=45,actual_fraction=45/128,group_sizes=list(map(len,groups)),group_draws=group_draw,unique_fingerprints=len({r['state_fingerprint'] for r in new}),valid=all(r['valid'] for r in new),data_sha256=sha(data),independent_new_sources=2,correlated_descendants=32,first500_not_full1000_counts=[int(np.isin(mix[:500,:45]-311,g).sum()) for g in groups])
# All current entrypoint import sites; old same-reset helper is intentionally retained.
uses=[]
for p in (R/'research_notes').glob('*.py'):
 if not any(x in p.name for x in ['interactive','rolling','retention']):continue
 for i,l in enumerate(p.read_text().splitlines(),1):
  if 'import' in l and ('BranchState'in l or 'CrossResetState'in l):uses.append(dict(file=p.name,line=i,source=l.strip()))
res['restore_entrypoints']=uses
checks=[]
for p in (Q/'runs').glob('*/validation.json'):
 try:
  d=json.loads(p.read_text())
  if isinstance(d,list) and d and 'fixed_cross_reset'in d[0]:checks.append(dict(path=str(p),sha256=sha(p),result=d))
 except Exception:pass
res['saved_cross_reset_checks']=checks
from rolling_snapshot import state_of,put
class Dummy:pass
d=Dummy();d.rng=np.random.default_rng(773);d.value=1;snap=state_of(d);before=copy.deepcopy(d.rng.bit_generator.state);d.rng.uniform();put(d,snap);res['snapshot_rng_minimal_repro']=dict(rng_in_snapshot='rng'in snap,rng_restored=d.rng.bit_generator.state==before,scope='clone deliberately skips numpy Generator; complete task sampler RNG is not part of snapshot. Existing fixed-demand branches do not call sample_spec; not proof of invalid physical branches.',fix_applied=False)
# Old eval shares task then resets it, independent action-noise Generator is unaffected directly.
src=R/'research_notes/interactive_increment_ppo.py';lines=src.read_text().splitlines();res['eval_isolation_source']=[dict(line=i,text=l.strip()) for i,l in enumerate(lines,1) if 'def evaluate'in l or 'task.reset'in l or 'evaluate(iteration)'in l or 'Generator'in l]
res['evaluation_classification']=dict(status='confirmed limited isolation',fact='old evaluate operates on same live task; after dev resets sample_spec and discards unfinished rollout continuation. Separate action-noise torch.Generator remains explicit. No claim that physical evaluation metrics or oldlogprob were false.',impact='training trajectory/curriculum may depend on evaluation frequency; per-batch dev must not reuse this wrapper unchanged',old_code_changed=False,repair='new A1 evaluator required independent process; archived runtime remains unchanged')
# Recheck F source contracts and exact line evidence.
for key,name,needles in [('F','humanoidverse/agents/fb/model.py',['def forward_map','def backward_map','def reward_inference','wr =','return self.project_z','def project_z','self._normalize']),('shadow','research_notes/interactive_pilot.py',['emb=','def forward_scores','z=spherical(task.last_command','action=m.act','ff=m.forward_map','vv=','fs,f_detail','all_F_scores'])]:
 pp=R/name;res[key+'_source']=dict(path=name,sha256=sha(pp),lines=[dict(line=i,text=l.strip()) for i,l in enumerate(pp.read_text().splitlines(),1) if any(n in l for n in needles)])
res['F_contract_verdict']=dict(status='source contract consistent for proxy',obs='current e.obs normalized once by public F/actor wrappers',z='first20% interpolation from actual previous z toward first proposed command',action='same-state m.act mean for the same interpolated z, not q_PD or raw highlevel',embedding='unprojected forward015 reward sum divided150000, F dot embedding then conservative min heads',limitation='fixed-z robot-state SF score does not model full external contact reward or future time-varying sequence; action/z tensors per candidate were not stored explicitly',no_recomputation=True)
res['elapsed_s']=time.monotonic()-beg;res['code_mutations']=0;res['new_physics']=0;res['new_training']=0
(out/'audit.json').write_text(json.dumps(res,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in res.items() if k not in ['F_source','shadow_source','saved_cross_reset_checks','restore_entrypoints']},indent=2))
