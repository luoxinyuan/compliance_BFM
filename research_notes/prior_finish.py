import json,pathlib,hashlib,gzip,time,collections,subprocess,os
import numpy as np,torch,mujoco
P=pathlib.Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');W=P/'bridge-v2-owned/trees/latent-prior-audit/repo';R=P/'bridge-v2-owned/runs/latent-prior-audit-20261105';O=W/'research_notes/prior_artifacts';t=time.monotonic()
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2))
models=read(R/'frozen-final-manifest.json');rows=read(R/'final-metrics.json');m=mujoco.MjModel.from_xml_path(str(W/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'));assert m.nq==36 and m.nv==35
j=[i for i in range(m.njnt) if m.jnt_qposadr[i]>=7];assert len(j)==29;names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in j];ranges=m.jnt_range[j];margins=[]
for row in rows:
 ep=json.load(gzip.open(row['path'],'rt'));rr=[r for r in ep['physics'] if r['active']];rr=rr or ep['physics'];q=np.array([r['qpos'][7:] for r in rr]);v=np.minimum(q-ranges[:,0],ranges[:,1]-q);mins=np.rad2deg(v.min(0));margins.append(dict(label=row['label'],right_arm_min_remaining_deg={n:float(d) for n,d in zip(names,mins) if 'right_' in n and any(a in n for a in ['shoulder','elbow','wrist'])},all_joint_min_remaining_deg=float(mins.min()),minimum_joint=names[int(mins.argmin())],scope='hard joint limits; not comfort margin or full Cartesian reachability'))
train={}
for a in 'AB':
 d=R/('ppo-'+a);log=[json.loads(s) for s in (d/'training.jsonl').read_text().splitlines()];counts={k:collections.Counter() for k in ['phase','family','completed']}
 for row in log:
  for k in counts:counts[k].update(row['coverage'][k])
 initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=True);selected=torch.load(d/'best.pt',map_location='cpu',weights_only=True)
 delta=max(float((v-selected['state_dict'][k]).abs().max()) for k,v in initial['state_dict'].items() if k.startswith('actor.'))
 dev=[]
 for f in sorted(d.glob('dev-*/summary.json')):
  rr=read(f)
  for row in rr:
   row.pop('handoff',None)
  save(O/(a+'-'+f.parent.name+'.json'),rr);dev.append(dict(iteration=int(f.parent.name.split('-')[-1]),score=read(f.parent/'score.json') if (f.parent/'score.json').exists() else read(d/'selection.json')['score']))
 train[a]=dict(result=read(d/'result.json'),selection=read(d/'selection.json'),coverage={k:dict(v) for k,v in counts.items()},development_scores=dev,selected_actor_max_delta_from_prePPO=delta,rollbacks=[read(f) for f in sorted(d.glob('rollback*.json'))],physical_steps=sum(x['physical_steps'] for x in log),control_steps=sum(x['control_steps'] for x in log),source_physics_steps=sum(x['source_physics_steps'] for x in log),prefill_physics_steps=sum(x['prefill_physics_steps'] for x in log))
 slim=[{k:v for k,v in x.items() if k!='episodes'} for x in log];save(O/(a+'-training.json'),slim)
ledgers={f.name:read(f) for f in R.glob('*-ledger.json')};save(O/'resource-ledger.json',dict(jobs=ledgers,total_recorded_job_wall=sum(x.get('job_wall',0) for x in ledgers.values()),extra_analysis_scope='Short CPU parsing, Git, transfer and video decoding separate; no RL samples added'))
save(O/'training-final.json',train);save(O/'final-metrics.json',rows);save(O/'joint-margins.json',margins);save(O/'selected-models.json',models);save(O/'videos.json',read(R/'videos/manifest.json'));save(O/'final-evaluation.json',read(R/'final-pipeline.json'));save(R/'closeout-analysis.json',dict(elapsed_s=time.monotonic()-t,rl_decisions=0,physics_steps=0))
print(json.dumps(dict(training={a:v['result'] for a,v in train.items()},actor_delta={a:v['selected_actor_max_delta_from_prePPO'] for a,v in train.items()},margins_min=min(x['all_joint_min_remaining_deg'] for x in margins),episodes=len(rows),analysis_s=time.monotonic()-t),indent=2))
