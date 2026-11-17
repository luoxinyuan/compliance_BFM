"""Frozen causal online-context probe of actual, ancestor-separated outcomes.
Never encodes an actual future outcome as a current-state feature.
This distinguishes I's online F_I context from its future-target B_I encoder.
"""
import os,sys,time,json,hashlib,traceback,collections
from pathlib import Path
START=time.monotonic();S=Path(__file__).resolve().parents[1];R=S/'eval-runtime'
sys.path[:0]=[str(R),str(R/'research_notes')]
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import torch,numpy as np
torch.set_num_threads(1);torch.set_grad_enabled(False)
from torch import nn
from safetensors import safe_open
from research.interaction_fb.ei_v2_dev import load_core
from research.interaction_fb.journal import read
from research.interaction_fb.schema import robot,deploy
from research.interaction_fb.ei_v2_agent import mlp,SCALES
OUT=S/'runs/final-probe-01';OUT.mkdir(exist_ok=False)
def jw(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def frozen_modules(cp):
 # Selective frozen-module read, never claims Task/Adam/RNG restoration.
 import zlib,pickle
 from research.interaction_fb import delta_tree
 cp=Path(cp);chain=[];seen=set();cursor=cp
 while True:
  assert cursor.name not in seen;seen.add(cursor.name);m=json.loads((cursor/'manifest.json').read_text());assert m['complete'];chain.append((cursor,m))
  if m['previous']is None:break
  cursor=cp.parent/m['previous']
 modules=None
 for path,m in reversed(chain):
  b=(path/'core.delta').read_bytes();meta=m['files']['core.delta'];assert len(b)==meta['bytes']and hashlib.sha256(b).hexdigest()==meta['sha256'];e=pickle.loads(zlib.decompress(b));assert e[0]=='dict';agent=dict(e[2])['agent'];assert agent[0]=='dict';module=dict(agent[2])['modules'];modules=delta_tree.decode(module,modules)
 ident=dict(checkpoint=str(cp),manifest_sha256=hashlib.sha256((cp/'manifest.json').read_bytes()).hexdigest(),module_only_not_training_restore=True,core_sha256=chain[0][1]['files']['core.delta']['sha256'])
 return modules,ident
result={}
try:
 root=S/'runs/final-stage-quality-03';support=json.loads((root/'probe-label-support.json').read_text());labels=json.loads((root/'probe-label-records.json').read_text());eligible=[float(h)for h,v in support.items()if v['status']=='ELIGIBLE_FOR_GROUPED_PROBE']
 if not eligible:
  result=dict(status='UNDERPOWERED',support=support,feature_computation_skipped=True)
 else:
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import roc_auc_score
  from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModel
  from humanoidverse.agents.base_model import migrate_legacy_aux_critic_config
  from humanoidverse.agents.envs.utils.gym_spaces import json_to_space
  p=S.parents[1]/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view/checkpoint/model'
  cfg=json.loads((p/'config.json').read_text());cfg['device']='cpu';cfg=FBcprAuxModel.config_class(**migrate_legacy_aux_critic_config(cfg));kw=json.loads((p/'init_kwargs.json').read_text());space=json_to_space(kw['obs_space'])
  # Build only the unchanged B_R and normalizer. No forward map/actor/optimizer or simulation is instantiated.
  br=cfg.archi.b.build(space,256).eval().requires_grad_(False);norm=cfg.obs_normalizer.build(space).eval().requires_grad_(False)
  with safe_open(p/'model.safetensors',framework='pt',device='cpu')as f:
   keys=f.keys()
   for module,prefix in ((br,'_backward_map.'),(norm,'_obs_normalizer.')):
    sd={k[len(prefix):]:f.get_tensor(k)for k in keys if k.startswith(prefix)};module.load_state_dict(sd,strict=True)
  from diagnostic_fb_resume import tree_digest
  fixture=S/'runs/formal-E-03/checkpoints/block-initial';full,_=load_core(fixture);partial,_=frozen_modules(fixture);assert tree_digest(partial)==tree_digest(full['agent']['modules']);del full,partial
  jw('partial-reader-fixture.json',dict(original_full_reader_equal=True,fixture=str(fixture),scope='only frozen inference modules; original delta checksums retained'))
  extract={};identities={}
  for arm in('E','I'):
   sd,ident=frozen_modules(S/f'runs/formal-{arm}-06/checkpoints/block-node-32000');predkeys={k[len('predictor.'):]:v for k,v in sd.items()if k.startswith('predictor.')};inp=predkeys['0.weight'].shape[1];pred=mlp(inp,128,72 if arm=='E' else 48);pred.load_state_dict(predkeys,strict=True);ge=nn.Sequential(nn.Linear(24,32),nn.SiLU(),nn.Linear(32,16));ge.load_state_dict({k[len('goal_encoder.'):]:v for k,v in sd.items()if k.startswith('goal_encoder.')},strict=True)
   for module in(pred,ge):module.eval().requires_grad_(False)
   selector_keys={k[len('selector.'):]:v for k,v in sd.items()if k.startswith('selector.')};selector=mlp(selector_keys['0.weight'].shape[1],128,4);selector.load_state_dict(selector_keys,strict=True);selector.eval().requires_grad_(False);extract[arm]=(pred,ge,selector);identities[arm]=ident;del sd
  d=S/'runs/formal-dev-02';rows=json.loads((d/'evaluation.json').read_text());bycase=collections.defaultdict(list)
  for lb in labels:
   if lb['horizon']in eligible:bycase[lb['case']].append(lb)
  data=[];features=[];selector_diagnostics=[];scales=torch.tensor(SCALES)[None]
  for ci,ll in sorted(bycase.items()):
   rr=rows[ci];a=read(d/f'case-{ci:03d}-actual.ptz',rr['archive']);unique=sorted({lb['row_index']for lb in ll});fi={}
   for b in range(0,len(unique),64):
    inds=unique[b:b+64];actual=[a['rows'][j]for j in inds];o={k:torch.cat([v['observation'][k]for v in actual])for k in actual[0]['observation']};z=torch.cat([v['z']for v in actual]);zfirst=o.get('_first_interpolated_z',z);opts=torch.tensor([v['option']for v in actual]);public=torch.cat((o['_goal_y']/scales,o['_goal_mask'],o['_goal_mode']),-1);dep=deploy(o);x=torch.cat((dep,public,torch.nn.functional.one_hot(opts,4).float(),zfirst),-1)
    eb=extract['E'][0](x).reshape(len(actual),3,24);ec=extract['E'][1](eb.reshape(-1,24)).reshape(len(actual),3,16).mean(1);ic=extract['I'][0](x).reshape(len(actual),3,16).mean(1);brv=br(norm(robot(o)))
    ff={'B_R':brv.numpy(),'I_online_FI_context':ic.numpy(),'B_R_plus_I_context':torch.cat((brv,ic),-1).numpy(),'current_deploy_public_baseline':torch.cat((dep,public),-1).numpy(),'E_online_outcome_context':ec.numpy()}
    for j,idx in enumerate(inds):fi[idx]={k:v[j]for k,v in ff.items()}
   for lb in ll:data.append(lb);features.append(fi[lb['row_index']])
   for row in a['rows']:
    if not row['cat_boundary']:continue
    o=row['selector_obs'];zs=row['candidate_first_z'];pub=torch.cat((o['_goal_y']/scales,o['_goal_mask'],o['_goal_mode']),-1);dep=deploy(o)
    for arm in('E','I'):
     contexts=[]
     for j in range(4):
      opt=torch.nn.functional.one_hot(torch.tensor([j]),4).float();x=torch.cat((dep,pub,opt,zs[:,j]),-1);pr=extract[arm][0](x).reshape(1,3,-1);context=extract[arm][1](pr.reshape(-1,24)).reshape(1,3,16).mean(1)if arm=='E'else pr.mean(1);contexts.append(context)
     raw=extract[arm][2](torch.cat((dep,pub,*contexts,o['_option_memory']),-1));follow=o['_goal_mode'][:,0:1];prior=torch.log(follow*torch.tensor([.10,.10,.70,.10])+(1-follow)*torch.tensor([.97,.01,.01,.01]));logits=raw+prior;sort=logits.sort(-1,descending=True).values
     selector_diagnostics.append(dict(case=ci,arm=arm,time=float(row['provenance']['time']),max_abs_raw=float(raw.abs().max()),raw_logits=raw[0].tolist(),argmax=int(logits.argmax(-1)),prior_argmax=int(prior.argmax(-1)),winning_margin=float(sort[0,0]-sort[0,1])))
   jw('progress.json',dict(complete_cases=len({r['case']for r in data}),planned_cases=len(bycase)));print('PROBE_FEATURES',ci,len(unique),flush=True)
  jw('feature-contract.json',dict(features={k:len(features[0][k])for k in features[0]},B_I_direct_not_a_causal_state_feature='The frozen B_I Goal encoder accepts a 24D physical outcome. Feeding the actual labeled future would leak the label. We report the deployed 16D F_I predicted context instead; no standalone B_I(state) result or full registered B_I probe PASS is claimed.',source_only_actual_I_primary=True,all_features_before_labeled_future=True,identities=identities))
  jw('selector-diagnostics.json',selector_diagnostics)
  results={};rng=np.random.default_rng(660304)
  for h in eligible:
   ix=[i for i,v in enumerate(data)if v['horizon']==h];y=np.array([data[i]['label']for i in ix]);groups=np.array([data[i]['ancestor']for i in ix]);unique=np.unique(groups);predictions={};scores={}
   for name in features[0]:
    x=np.stack([features[i][name]for i in ix]);pred=np.empty(len(y));folds=[]
    for g in unique:
     test=groups==g;train=~test;assert len(np.unique(y[train]))==2
     scaler=StandardScaler().fit(x[train]);clf=LogisticRegression(C=1,penalty='l2',solver='liblinear',max_iter=1000,tol=1e-4,random_state=660304);clf.fit(scaler.transform(x[train]),y[train]);pred[test]=clf.predict_proba(scaler.transform(x[test]))[:,1];folds.append(dict(ancestor=g,train_rows=int(train.sum()),test_rows=int(test.sum()),iterations=int(clf.n_iter_[0])))
    scores[name]=dict(AUROC=float(roc_auc_score(y,pred)),folds=folds);predictions[name]=pred;print('PROBE',h,name,scores[name]['AUROC'],flush=True)
   delta=[];auc_dist={k:[]for k in predictions}
   for _ in range(1000):
    draw=rng.choice(unique,len(unique),replace=True);ii=np.concatenate([np.flatnonzero(groups==g)for g in draw])
    if len(np.unique(y[ii]))<2:continue
    aa={k:float(roc_auc_score(y[ii],p[ii]))for k,p in predictions.items()}
    for k,v in aa.items():auc_dist[k].append(v)
    delta.append(aa['I_online_FI_context']-max(aa[k]for k in('B_R','current_deploy_public_baseline','E_online_outcome_context')))
   best=max(scores[k]['AUROC']for k in('B_R','current_deploy_public_baseline','E_online_outcome_context'));gap=scores['I_online_FI_context']['AUROC']-best
   for k in scores:scores[k]['ancestor_bootstrap95_AUROC']=np.quantile(auc_dist[k],[.025,.975]).tolist()
   results[str(h)]=dict(rows=len(y),positive_rows=int(y.sum()),negative_rows=int((1-y).sum()),independent_ancestors=len(unique),scores=scores,I_context_increment=gap,ancestor_bootstrap95_increment=np.quantile(delta,[.025,.975]).tolist(),registered_full_BI_probe_status='UNMEASURED_no_causal_BI_state_input',causal_context_increment_supported=bool(gap>=.05 and np.quantile(delta,.025)>0))
   jw('scores.json',results)
  result=dict(status='CAUSAL_PREDICTED_CONTEXT_PROBE_COMPLETE_NOT_STANDALONE_BI',support=support,results=results,model_identities=identities,physical_steps=0,policy_updates=0,representation_updates=0,probe_classifier_only=True,sealed_read=False)
except BaseException as ex:result=dict(status='PROBE_FAILED',error=repr(ex),traceback=traceback.format_exc(),physical_steps=0,policy_updates=0)
finally:
 result['wall_s']=time.monotonic()-START;result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();jw('result.json',result)
 lp=S/'ledger.json';ledger=json.loads(lp.read_text());ledger['charged_s']+=result['wall_s'];ledger['jobs'].append(dict(name=OUT.name,status=result['status'],wall_s=result['wall_s'],physics=dict(actual=0,reserved=0),source_sha256=result['source_sha256']));tmp=lp.with_suffix('.pending');tmp.write_text(json.dumps(ledger,indent=2)+'\n');os.replace(tmp,lp)
 print(json.dumps(result),flush=True)
if result['status']=='PROBE_FAILED':raise RuntimeError(result['error'])
