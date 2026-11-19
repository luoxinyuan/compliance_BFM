import copy,gzip,json,time,shutil,os
from pathlib import Path
import numpy as np,torch
from rolling_snapshot import BranchState
def conduct(task,env,condition,search,propose,rng,save_episode,counts,old,prefix,spherical,out):
 ep=prefix(task,condition(),old,out)
 history=json.loads((old/'runs/rolling-02/searches.json').read_text())[:10]
 for row in history:
  assert row['candidates'][row['selected']]['kind']=='standing'
  pool=propose(row['phase'],None);assert len(pool)==7
 from measure import monitor,metrics
 with gzip.open(old.parent/'ctft_positive50_01/low01/rolling_continue.json.gz','rt') as f:materials=json.load(f)
 for i,r in enumerate(task.physics):r['ctft_material_contacts']=materials[i]
 active=monitor(env)
 state=BranchState(task);rng0=copy.deepcopy(rng.bit_generator.state)
 (out/'source.ok').write_text(state.fingerprint)
 if os.environ.get('GATE_ONLY')=='1':return {'status':'prefix_verified'}
 for label in ('baseline','soft'):
  state.restore();rng.bit_generator.state=copy.deepcopy(rng0);phase=0.;previous={'kind':'standing'};i=0;log=[]
  while not task.done:
   now=float(env.tether.time);z,previous,res=search(label,phase,previous)
   phase+=(.1*previous.get('rate',0.))/task.refs['straight']['period'] if previous['kind']=='sequence' else 0.
   actual=spherical(env.standing,z,.5) if label=='soft' and i<5 else z
   active[0]=True
   _,_,_,st=task.step(actual/16);active[0]=False;counts['actual_physics_steps']+=st['physics_steps']
   log.append(dict(t=now,selected=res['selected'],applied=actual.flatten().cpu().tolist(),planned=z.flatten().cpu().tolist(),wall=res['wall_s']))
   (out/(label+'-commands.json')).write_text(json.dumps(log))
   assert shutil.disk_usage(out).free>100*2**30
   i+=1
   if label=='baseline' and i==5:
    remaining=211*float(np.percentile([v['wall'] for v in log],90))*1.15
    assert remaining+time.time()-json.loads((out/'launch.json').read_text())['start']+float(os.environ.get('PRIOR_SEC',0))<10800,('budget projection',remaining)
   if i%10==0:print(label,i,now,flush=True)
  save_episode(label)
  q=metrics(task,env);(out/(label+'-measured.json')).write_text(json.dumps(q,indent=1))
  if label=='baseline':assert q['move']['root_cm']>=50 and q['contact']['err']<=10 and q['contact']['force']<=6.5 and q['slip_pass'] and q['hard'] is None,('reference quality',q['move'],q['contact'],q['slip_pass'])
 return dict(status='completed',counts=counts)
