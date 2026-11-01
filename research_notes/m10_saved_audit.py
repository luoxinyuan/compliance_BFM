"""Saved executed qpos + source reference FK audit. No dynamics or model inference."""
from pathlib import Path
import json,gzip,hashlib,os
import numpy as np,mujoco
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs';ROOT=Path(__file__).resolve().parents[1]
OUT=R/'m10-task-appropriate-20261101';OUT.mkdir(exist_ok=True)
m=mujoco.MjModel.from_xml_path(str(ROOT/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'));d=mujoco.MjData(m)
names=['pelvis','torso_link','left_wrist_roll_link','right_wrist_roll_link','left_shoulder_pitch_link','right_shoulder_pitch_link'];ids={n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in names};assert min(ids.values())>=0
read=lambda p:json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
def fk(q):
 d.qpos[:]=q;mujoco.mj_forward(m,d);p={n:d.xpos[i].copy() for n,i in ids.items()};a=d.xmat[ids['pelvis']].reshape(3,3);b=d.xmat[ids['torso_link']].reshape(3,3)
 # Positive pitch: top of body's vertical axis leans toward its forward axis.
 pitch=lambda x:float(np.rad2deg(np.arctan2(-x[2,0],x[2,2])))
 return dict(left_rel=(a.T@(p[names[2]]-p['pelvis'])).tolist(),right_rel=(a.T@(p[names[3]]-p['pelvis'])).tolist(),left_world=p[names[2]].tolist(),right_world=p[names[3]].tolist(),pelvis_pitch=pitch(a),torso_pitch=pitch(b),waist_pitch=pitch(a.T@b),root=p['pelvis'].tolist())
def aggregate(v):
 return {k:{'mean':np.mean([r[k] for r in v],axis=0).tolist(),'min':np.min([r[k] for r in v],axis=0).tolist(),'max':np.max([r[k] for r in v],axis=0).tolist()} for k in v[0]}
items=[]
def add(run,stage,labels):
 for label in labels:items.append((run,stage,label))
add('m8-contact-body-20261101A','qualification-attempt-02',['candidate-1','standing'])
add('m8-contact-body-20261101A','final-attempt-01',['down_test__teacher','down_test__distill'])
add('m9-feedback-continuation-20261101B','final-attempt-01',['held_straight__expert','held_straight__student','held_left__student','held_right__student','down_test__student'])
add('m9c-expert-20261101','final-attempt-01',['new_straight__expert','new_straight__learned_expert','new_straight__transfer','new_straight__ablation','wrist_early__m8','wrist_early__ablation','wrist_early__transfer','down_test__transfer'])
result={'scope':'Actual saved qpos forward kinematics and controller logs; no physics/model execution','episodes':{},'references':{}}
for run,stage,label in items:
 path=R/run/stage/'outputs'/(label+'.json.gz');e=read(path);rs=e['physics'];ts=np.array([r['time'] for r in rs]);spec=e['condition'];on=spec.get('onset',3.8);rel=spec.get('release',9.);ramp=spec.get('ramp',3.);v=[fk(r['qpos']) for r in rs[::4]];tt=ts[::4];initial=fk(e['initial']['qpos']);phases={}
 for name,mask in {'pre':tt<on,'entry':(tt>=on)&(tt<min(on+2,rel)),'contact':(tt>=on)&(tt<rel),'late_contact':(tt>=min(on+ramp,rel-1))&(tt<rel),'recover':tt>=rel}.items():
  rows=[r for r,a in zip(v,mask) if a]
  if rows:phases[name]=aggregate(rows)
 active=[r for r in rs if r.get('active')];F=np.array([r['force'] for r in active]);E=np.array([np.array(r['anchor'])-r['wrist'] for r in active]);root=np.array([r['root'] for r in rs]);cs=e['control'];q=np.array([r['q_PD'] for r in cs]);dq=np.diff(q,axis=0)
 c=dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),condition=spec,boundary=e.get('boundary'),initial=initial,phases=phases,root_displacement=(root[-1]-root[0]).tolist(),qpd_step_max_deg=float(np.rad2deg(abs(dq).max())),qpd_group_max_deg={n:float(np.rad2deg(abs(dq[:,sl]).max())) for n,sl in [('legs_waist',slice(0,15)),('left_arm',slice(15,22)),('right_arm',slice(22,29))]})
 if len(F):c.update(force_xyz_rms=np.sqrt((F*F).mean(0)).tolist(),force_rms=float(np.sqrt((F*F).sum(1).mean())),cap_fraction=float((np.linalg.norm(F,axis=1)>4.999).mean()),error_xyz_rms_cm=(100*np.sqrt((E*E).mean(0))).tolist(),error_rms_cm=float(100*np.sqrt((E*E).sum(1).mean())))
 if 'wrist_early__' in label:
  c['decision_samples']=[dict(time=r['time'],force=r.get('measured_force_xyz'),qpd=r['q_PD'],root=r.get('pre_qpos',[])[:3],z=r['executed_z']) for r in cs if on<=r['time']<=on+2][::5]
 result['episodes'][run+'/'+label]=c;print(label,'done',flush=True);(OUT/'saved-audit.json').write_text(json.dumps(result,indent=2))
cycles=read(R/'m9-contact-locomotion-20261101A/native-cycles.json')['cycles']
for name,ref in cycles.items():result['references'][name]=dict(source=ref.get('source'),period=ref['period'],phase_offset=ref['phase_offset'],kinematics=aggregate([fk(q) for q in ref['q']]))
ref=read(R/'m8-contact-body-20261101A/qualification-attempt-02/outputs/candidate-1.json');result['references']['crouch_goal']=dict(motion=ref['motion'],frame=ref['frame'],fk=fk(ref['qpos']))
(OUT/'saved-audit.json').write_text(json.dumps(result,indent=2));print('AUDIT_COMPLETE',flush=True)
