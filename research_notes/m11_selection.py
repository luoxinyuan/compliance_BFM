"""Bind selection score to evaluated weights and protocol, never an inherited scalar."""
import hashlib,json,os,shutil
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,obj):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));os.replace(tmp,p)
class Selection:
 def __init__(self,folder,scope):
  self.folder=Path(folder);self.scope=scope;self.record=None
 def consider(self,candidate,rank,evaluation,metadata=None):
  candidate=Path(candidate);evaluation=Path(evaluation);assert candidate.is_file() and evaluation.is_file()
  rank=list(rank)
  if self.record is not None and tuple(rank)>=tuple(self.record['rank']):return False
  target=self.folder/'best.pt';tmp=target.with_suffix('.tmp');shutil.copyfile(candidate,tmp);os.replace(tmp,target)
  self.record=dict(scope=self.scope,rank=rank,checkpoint_sha256=sha(target),candidate_sha256=sha(candidate),candidate_path=str(candidate),evaluation_path=str(evaluation),evaluation_sha256=sha(evaluation),metadata=metadata or {})
  assert self.record['checkpoint_sha256']==self.record['candidate_sha256'];atomic(self.folder/'selection.json',self.record);return True

def continuation_rank(mean_rows,baseline):
 """Failure count first, then family-balanced relative cost + worst-two tail.
 No final result participates. All per-family costs remain in the record.
 """
 def cost(r):
  s=r['condition'];c=r['force_rms']+3*r['error_cm']/100+r['cap_fraction']
  if not s['enabled']:c+=100*r.get('root_max_displacement',0)
  elif s['family']=='old':c+=2*r.get('root_max_displacement',0)
  return c
 ratios={r['case']:cost(r)/max(cost(baseline[r['case']]),.5) for r in mean_rows}
 values=sorted(ratios.values(),reverse=True);tail=sum(values[:2])/min(2,len(values));score=sum(values)/len(values)+.5*tail
 return [sum(r['boundary'] is not None for r in mean_rows),score],dict(relative_costs=ratios,worst_two_mean=tail,mean=sum(values)/len(values))

def exploration_scale(arm,batch,total=160):
 assert arm in ['fixed','annealed'] and 0<=batch<total
 if arm=='fixed':return 1.
 # Hold first 20, exponential decay at batch boundaries to 1/4 by batch120,
 # then retain nonzero .02-scale sampling for the final 40 batches.
 import math
 progress=min(max((batch-20)/100,0.),1.)
 return math.exp(math.log(.25)*progress)
