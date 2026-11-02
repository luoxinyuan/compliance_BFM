"""Summarize immutable frozen-policy diagnostics and verify paired physics inputs.
No model inference, policy updates, new rollout or parameter selection.
"""
import argparse,json,gzip,statistics
from pathlib import Path

def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_text())
def aggregate(rows):
 return {k:dict(mean=statistics.mean(r[k] for r in rows),sample_sd=statistics.stdev(r[k] for r in rows) if len(rows)>1 else None,values=[r[k] for r in sorted(rows,key=lambda r:r['condition']['diagnostic_seed'])]) for k in ['error_cm','force_rms','cap_fraction']}
def difference(a,b):
 if isinstance(a,list):
  assert len(a)==len(b)
  return max([difference(x,y) for x,y in zip(a,b)]+[0.])
 return abs(a-b)
def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=a.run
 noise=read(r/'diagnostic-attempt-01/outputs/job/diagnostic-summary.json');memory=read(r/'memorycheck-attempt-01/outputs/job/diagnostic-summary.json');checks=read(r/'readcheck-attempt-02/outputs/job/diagnostic-summary.json')
 out=dict(noise={},memory={},pair_audit=[],readcheck=[{k:x[k] for k in ['label','error_cm','force_rms','point_read_audit']} for x in checks])
 for task in ['dev_straight','exposed_new_straight']:
  for s in [.5,1.]:out['noise'][f'{task}:std{s}']=aggregate([x for x in noise if x['label'].startswith(task) and x['condition']['diagnostic_std_scale']==s])
 baseline=read(r/'final-attempt-01/outputs/job/evaluation/new_straight__learned.json.gz')
 for e,m in [(1.,0.),(0.,1.)]:out['memory'][f'execution{e}:memory{m}']=aggregate([x for x in memory if x['condition']['diagnostic_std_scale']==e and x['condition']['diagnostic_memory_std_scale']==m])
 for x in memory:
  ep=read(r/'memorycheck-attempt-01/outputs/job/evaluation'/(x['label']+'.json.gz'))
  initial=max(difference(ep['initial'][k],baseline['initial'][k]) for k in ['qpos','qvel'])
  partner=max(difference([v[k] for v in ep['physics']],[v[k] for v in baseline['physics']]) for k in ['anchor','anchor_velocity','time','active'])
  assert initial==0 and partner==0 and ep['boundary'] is None
  out['pair_audit'].append(dict(label=x['label'],initial_difference=initial,partner_difference=partner,completed=True))
 out['interpretation']='Memory-only noise does not recover failed mean execution; executed noise with mean-command memory retains large tracking improvement. Fictitious memory is diagnostic only, not deployable performance. Final combo is now exposed; no updates or checkpoint reselection.'
 (r/'analysis/frozen-diagnostic-summary.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['readcheck','pair_audit']},indent=2))
if __name__=='__main__':main()
