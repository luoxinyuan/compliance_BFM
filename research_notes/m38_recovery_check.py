from pathlib import Path
import gzip,json,numpy as np
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m38-rotation-20261103')
def read(p):return json.load(gzip.open(p,'rt'))
out=[]
for name,t in [('pair_left',7.),('pair_right',7.),('straight',5.)]:
 a=read(R/'A-goal-03/job'/('retained_'+name+'.json.gz'));b=read(R/'A-goal-07/job'/(('T30_takeover_' if name=='straight' else 'source_takeover_')+name+'.json.gz'));aa=a['physics'];bb=b['physics'];before=[i for i,r in enumerate(aa) if r['force_time']<t-1e-7];diff={k:float(np.max(np.abs(np.array([aa[i][k] for i in before])-np.array([bb[i][k] for i in before])))) for k in ['qpos','qvel','force']};assert max(diff.values())<1e-9
 def tail(rows):
  rr=[r for r in rows if r['active'] and r['force_time']>=t];e=[np.mean([np.sum((np.array(c['anchor'])-c['point'])**2) for c in r['contacts']]) for r in rr];return dict(error_cm=100*float(np.sqrt(np.mean(e))),sum_force_rms=float(np.sqrt(np.mean([r['sum_force_norm']**2 for r in rr]))))
 out.append(dict(condition=name,takeover=t,prefix_max_abs=diff,continued=tail(aa),takeover_result=tail(bb),scope='Only this exact real prefix/state; no claim all later states recoverable.'))
(R/'recovery-exact.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
