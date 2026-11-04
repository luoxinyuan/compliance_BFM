"""Read-only saved-trajectory audit; no model, simulation or GPU execution."""
import json, gzip, pathlib, math, sys
p=pathlib.Path(sys.argv[1])
def load(f):
 with gzip.open(f,'rt') as h:return json.load(h)
def flat(x):return [y for v in x for y in flat(v)] if isinstance(x,list) else [x]
def norm(x):return math.sqrt(sum(v*v for v in x))
def diff(a,b):return [x-y for x,y in zip(flat(a),flat(b))]
base=load(p/'B-ppo-04/job/dev-0000/stand_walk__mean.json.gz')
out={'scope':'saved development traces, not causal counterfactual or final reselection','arms':{}}
for arm in ['A-ppo-03','B-ppo-04']:
 rows=[]
 for it in [0,16,32,48,64]:
  folder=p/arm/'job'/f'dev-{it:04d}'
  summaries=json.loads((folder/'summary.json').read_text())
  d=load(folder/'stand_walk__mean.json.gz');c=d['control'][0];b=base['control'][0]
  z=flat(c['executed_z']);zz=flat(b['executed_z'])
  angle=math.degrees(math.acos(max(-1,min(1,sum(x*y for x,y in zip(z,zz))/(norm(z)*norm(zz))))))
  rows.append({'iteration':it,'first_control_time':c['time'],'initial_qpos_max_diff':max(map(abs,diff(c['pre_qpos'],b['pre_qpos']))),'first_z_angle_deg':angle,'first_qpd_rms_difference_deg':math.degrees(norm(diff(c['q_PD'],b['q_PD']))/math.sqrt(29)),'first_root_separation_over_2cm_s':next((x['time'] for x,y in zip(d['physics'],base['physics']) if norm(diff(x['root'],y['root']))>.02),None),'root_x_displacement_cm':{str(t):(min(d['physics'],key=lambda x:abs(x['time']-t))['root'][0]-d['physics'][0]['root'][0])*100 for t in [6,8,10,12,14]},'conditions':[{k:x[k] for k in ['label','error_cm','force_rms','boundary','episode_return']} for x in summaries]})
 out['arms'][arm]=rows
ledger=json.loads((p/'compute-ledger.json').read_text())['attempts']
out['ledger_wall_seconds']=sum(x.get('wall_seconds',0) for x in ledger)
out['all_ledger_jobs_finished']=all(x['status']=='finished' for x in ledger)
print(json.dumps(out,indent=2))
