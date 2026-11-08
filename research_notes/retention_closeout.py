"""Read-only aggregate/selection after authorized stage completion, no new evaluation."""
from pathlib import Path
import json,gzip,hashlib,math,time,os
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/ppo-retention-audit-20261109';O=P/'experiments/interactive-increment-20261108';out=S/'runs/final-analysis';out.mkdir(exist_ok=False)
def read(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else Path(p).open() as f:return json.load(f)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rs=sum([read(p) for p in (S/'runs').glob('A0-eval-*/summaries.json')],[]);base={r['case']:r for r in rs if r['role']=='BC500'}
def cost(r):a=r['all'];return a['force_rms']+.03*(a['error_cm']or 0)+a['cap_fraction']
def severe(r,b):
 a,c=r['all'],b['all'];bad=[]
 if r['boundary']:bad.append('boundary')
 if a['error_cm'] is not None and a['error_cm']>c['error_cm']+max(5,.5*c['error_cm']):bad.append('error')
 if a['force_rms']>c['force_rms']+2:bad.append('force')
 if r['case'] in ['short08','none'] and r['root_max_displacement']>b['root_max_displacement']+.1:bad.append('root')
 peak=lambda a:max(q['slip_peak_mps'] for q in a['quality'].values())
 if peak(r)>max(peak(b)+.15,2*peak(b)):bad.append('slip')
 yaw=lambda a:max(abs(q['yaw_deg_final']-a['quality']['precontact']['yaw_deg_initial']) for q in a['quality'].values())
 if yaw(r)>yaw(b)+15:bad.append('yaw')
 return bad
run=S/'runs/A1-train-5403';new=[];select=[]
for i in range(8):
 p=run/f'eval-{i}/summaries.json'
 if not p.exists():continue
 rr=read(p)
 if len(rr)!=4:continue
 new+=rr;dd={r['case']:r for r in rr};bad={k:severe(dd[k],base[k]) for k in dd};a,b=dd['normal72']['all'],base['normal72']['all'];main=a['error_cm']<=.95*b['error_cm'] and a['force_rms']<=.95*b['force_rms'];side=all(dd[k]['all']['error_cm']<=base[k]['all']['error_cm']+.5 and dd[k]['all']['force_rms']<=base[k]['all']['force_rms']+.25 for k in ['short08','stop66']);score=cost(dd['normal72'])+.1*np.mean([cost(dd[k]) for k in ['short08','stop66']]);select.append(dict(iteration=i,eligible=not any(bad.values()) and main and side,severe=bad,main_gain=main,side_tolerance=side,score=float(score),model=rr[0]['model_path'],sha256=rr[0]['model_sha256'],metrics={k:dict(error_cm=v['all']['error_cm'],force_rms=v['all']['force_rms'],cap=v['all']['cap_fraction']) for k,v in dd.items()}))
elig=[r for r in select if r['iteration']>0 and r['eligible']];chosen=min(elig,key=lambda r:(r['score'],r['iteration'])) if elig else next((r for r in select if r['iteration']==0),None)
learn=read(run/'learning.json') if (run/'learning.json').exists() else []
oldlearn={str(seed):read(O/f'runs/ppo-clean-{seed}/learning.json') for seed in [5401,5402]}
iso=[read(p) for p in sorted(run.glob('eval-*-isolation.json'))]
counts={}
for p in run.glob('rollout-*-stats.json'):
 rows=read(p)
 for r in rows:
  k=(r['demand_family'],r['phase']);key='/'.join(k);v=counts.setdefault(key,dict(decisions=0,physics=0,wall_s=0));v['decisions']+=1;v['physics']+=r['physics_steps'];v['wall_s']+=r['decision_wall_s']
jobs=[read(p) for p in (S/'jobs').glob('*.json')];costs={w:sum(j.get('elapsed_s',0) for j in jobs if j['workflow']==w) for w in ['A0','A1','B','C']}
result=dict(A0=read(S/'runs/A0-decision.json'),retrospective_rows=rs,same_state=read(S/'runs/A0-same-state/same-state.json'),support_events=read(S/'runs/A0-support-events.json'),gradient_conflict=read(S/'runs/A0-same-state/gradient-conflict.json'),old_learning=oldlearn,new_learning=learn,new_development=new,selection=dict(rule=read(S/'repo/research_notes/retention_protocol.json')['selection'],all_candidates=select,selected=chosen,best_selection_preserves_benefit=bool(elig),final_iteration=max([x['iteration'] for x in select],default=None),final_eligible=select[-1]['eligible'] if select else False,stability_not_implied_by_best=True,no_new_blind_test=True),evaluation_isolation=iso,coverage=counts,B=read(S/'runs/B-eligibility/eligibility.json'),C=read(S/'runs/C-readonly/audit.json'),jobs=jobs,costs=costs,RL_decisions=sum(v['decisions'] for v in counts.values()),Teacher_queries=0,B_fit=False,checkpoint_identities=[dict(path=str(p),sha256=sha(p)) for p in sorted(run.glob('candidate-*.pt'))])
(out/'evidence.json').write_text(json.dumps(result,indent=2,allow_nan=False))
# Static scientific artifact from saved metrics only.
os.environ['MPLCONFIGDIR']=str(S/'.matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
for j,k in enumerate(['short08','normal72','stop66']):
 for ax,metric,unit in [(axes[0,j],'error_cm','error (cm)'),(axes[1,j],'force_rms','force RMS (N)')]:
  ax.plot([r['iteration'] for r in select],[r['metrics'][k][metric] for r in select],marker='o',label='seed5403');ax.axhline(base[k]['all'][metric],color='gray',linestyle='--',label='BC500');ax.set(title=k,xlabel='PPO batch',ylabel=unit);ax.grid(alpha=.25)
axes[0,0].legend();fig.savefig(out/'retention_learning.png',dpi=150);plt.close(fig)
print(json.dumps(dict(costs=costs,decisions=result['RL_decisions'],selected=chosen,final=select[-1] if select else None),indent=2))
