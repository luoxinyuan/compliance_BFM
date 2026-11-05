import json,hashlib,sys,time
from pathlib import Path
import torch
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/feedback-coexistence-20261105'
sys.path.insert(0,str(P/'bridge-v2-owned/trees/feedback-coexistence/repo/research_notes'))
from m39a_metrics import cost
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(d,it):return {x['label'].split('__')[0]:x for x in read(d/f'dev-{it:04d}/summary.json')}
checks=[];ids={}
for s in [4601,4602]:
 d=R/f'ppo{s}';result=read(d/'result.json');assert result['status']=='completed' and result['iterations']==32
 sel=read(d/'selection.json');a=rows(d,0);b=rows(d,sel['iteration'])
 v=dict(seed=s,selection=sel,score_improvement=1-sel['score']/1.5,local_error_improvement=1-sum(b[k]['error_cm'] for k in ['local7','local9'])/sum(a[k]['error_cm'] for k in ['local7','local9']),long_worst_cost=max(cost(b[k])/cost(a[k]) for k in ['long65','long80']),boundary=any(x['boundary'] for x in b.values()))
 v['qualifies_extension']=v['score_improvement']>=.05 and v['local_error_improvement']>=.1 and v['long_worst_cost']<=1.15 and not v['boundary'];checks.append(v)
 for role in ['best','last']:
  p=d/(role+'.pt');ids[f'seed{s}_{role}']=dict(path=str(p),sha256=sha(p),role=role)
extend=all(x['qualifies_extension'] for x in checks);decision=dict(checks=checks,extend_both=extend,final_unopened=True,time=time.time(),rl_decisions=sum(read(R/f'ppo{s}'/'result.json')['decisions'] for s in [4601,4602]))
(R/'joint-extension-decision.json').write_text(json.dumps(decision,indent=2));assert not extend,'Qualifies extension: stop final setup, check remaining budget and proper resume'
plan=read(P/'bridge-v2-owned/trees/feedback-coexistence/repo/research_notes/coexist_protocol.json');ids={'I':plan['initial'],**ids}
refs={'C0':P/'bridge-v2-owned/runs/prior-recovery-20261105/correct-C0/C0.pt','F4501':P/'bridge-v2-owned/runs/conditional-adapter-20261105/F4501/F4501-2000.pt','R4501':P/'bridge-v2-owned/runs/conditional-adapter-20261105/R4501/R4501-2000.pt'}
for k,p in refs.items():ids[k]=dict(path=str(p),sha256=sha(p))
# Mean function identity: actor plus input normalization. Critic does not drive evaluation.
def signature(p):
 c=torch.load(p,map_location='cpu',weights_only=True);sd=c['state_dict'];h=hashlib.sha256()
 for k in sorted(sd):
  if k.startswith('actor.') or k in ['center','scale']:
   h.update(k.encode());h.update(sd[k].numpy().tobytes())
 return h.hexdigest()
seen={};unique={};aliases={}
for k,v in ids.items():
 sig=signature(v['path']);v['mean_function_sha256']=sig
 if sig in seen:aliases[k]=seen[sig]
 else:seen[sig]=k;unique[k]=v
(R/'final-frozen-models.json').write_text(json.dumps(dict(all=ids,unique=unique,aliases=aliases,selection='development only; last included as prespecified negative comparison'),indent=2))
groups={'final-learned':{},'final-reference':{},'final-FR':{}}
for k,v in unique.items():groups['final-learned' if k.startswith('seed') else 'final-FR' if k.startswith(('F','R')) else 'final-reference'][k]=v
for k,v in groups.items():(R/(k+'-models.json')).write_text(json.dumps(v,indent=2))
print(json.dumps(decision,indent=2));print('unique',list(unique),'aliases',aliases)
