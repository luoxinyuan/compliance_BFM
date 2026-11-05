"""Saved-trajectory independent geometry reward reconstruction, no new physics."""
import json,gzip,math,time
from pathlib import Path
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs')
def wrap(x):return math.atan2(math.sin(x),math.cos(x))
def run(out):
 started=time.monotonic();results=[]
 for model,folder in [('I','final-reference'),('C0','final-reference'),('C1','final-reference'),('F4501','final-F'),('R4501','final-R')]:
  for cond in ['new9','new65','stop','continue']:
   path=P/'conditional-adapter-20261105'/folder/(model+'__'+cond+'.json.gz');e=json.load(gzip.open(path,'rt'));s=e['condition'];ci=pi=0;phase={};errors=[];sums={};discounted=0.;discount=1.
   for st in e['rl_stats']:
    n=st['control_steps'];nn=st['physics_steps'];cs=e['control'][ci:ci+n];rs=e['physics'][pi:pi+nn];ci+=n;pi+=nn
    terms={k:float(np.mean([c['legacy_reward_terms_not_used'][k] for c in cs])) for k in st['terms']}
    terms['force']=1.5*np.mean([len(r['contacts'])*sum(np.dot(c['force'],c['force']) for c in r['contacts'])/25 for r in rs])
    ec=np.mean([np.mean([np.sum((np.array(c['anchor'])-c['point'])**2) for c in r['contacts']]) if r['active'] else 0 for r in rs]);terms['tracking']=.15*min(ec/.04,16)
    orient=[]
    for r in rs:
     pp=np.array(r['pelvis_matrix']);tt=np.array(r['torso_matrix']);pe=wrap(r['target_yaw']-math.atan2(pp[1,0],pp[0,0]));te=wrap(r['target_yaw']-math.atan2(tt[1,0],tt[0,0]));pair=s['geometry']=='torso_pair'
     orient.append(min((.8*te*te+.2*pe*pe if pair else pe*pe)/math.radians(20)**2,9)*(.5 if pair else .35*min(s['distance']/.8,1)) if r['active'] else 0.)
    terms['heading']=float(np.mean(orient));terms['uncapped']=.15*np.mean([np.mean([min(np.log1p(np.dot(c['raw_force'],c['raw_force'])/25),8) for c in r['contacts']]) for r in rs]);terms['relative_velocity']=.1*np.mean([np.mean([min(np.sum((np.array(c['anchor_velocity'])-c['velocity'])**2)/.25,16) for c in r['contacts']]) if r['active'] else 0 for r in rs])
    reward=-.02*n*sum(terms.values());errors.append(abs(reward-st['reward']));discounted+=discount*reward;discount*=.99**(n/5)
    t=rs[0]['force_time'];ph='pre' if t<s['onset'] else 'release' if t>=s['release'] else 'decelerate' if 'stop_time' in s and s['stop_time']<=t<s['stop_time']+s['deceleration'] else 'hold' if ('stop_time' in s and t>=s['stop_time']+s['deceleration']) or ('stop_time' not in s and t>=s['onset']+s['ramp']) else 'move'
    row=phase.setdefault(ph,dict(seconds=0,return_sum=0,terms={}));row['seconds']+=.02*n;row['return_sum']+=reward
    for k,v in terms.items():row['terms'][k]=row['terms'].get(k,0)+.02*n*v;sums[k]=sums.get(k,0)+.02*n*v
   assert max(errors)<1e-5,(path,max(errors));results.append(dict(model=model,condition=cond,source=str(path),max_reward_error=max(errors),return_sum=-sum(sums.values()),discounted_return=discounted,integrated_costs=sums,phases=phase))
 Path(out).write_text(json.dumps(dict(rows=results,seconds=time.monotonic()-started,physics_steps=0,training_decisions=0,scope='same condition comparisons only; unchanged components checked against recorded per-control costs'),indent=2));print([(r['model'],r['condition'],round(r['return_sum'],3)) for r in results])
if __name__=='__main__':
 import sys;run(sys.argv[1])
