"""Only complete, matched two-second windows enter mechanism denominators."""
import gzip,pickle,sys,os,json,math,hashlib
from pathlib import Path
import numpy as np,mujoco
root=Path(os.environ['CTFT_PINNED_RUNTIME']);from score_endpoint import material,stance_segments
s=Path(sys.argv[1]);m=mujoco.MjModel.from_xml_path(str(root/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'))
weight=m.body_mass.sum()*np.linalg.norm(m.opt.gravity)
def read(p):
    with gzip.open(p,'rb') as f:return pickle.load(f)
def metric(x):
    ph=x['physics'][:400];cs=x['controls'][:100];q0=x['initial_qpos']
    if len(ph)!=400 or len(cs)!=100 or not all('native_terms' in v for v in cs):return None
    loads=np.array([r['foot_normal_force_n'] for r in ph])
    segments=material(m,ph,q0,stance_segments(loads,weight))
    if not segments or not all(v['measured'] for v in segments):return None
    ang=math.radians(x['spec']['partner']['direction_deg']);h=np.array([math.cos(ang),math.sin(ang)])
    kin=mujoco.MjData(m);kin.qpos[:]=q0;mujoco.mj_kinematics(m,kin)
    wrist0=kin.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"right_wrist_yaw_link")]
    return dict(root=float((ph[-1]['qpos'][:2]-q0[:2])@h),wrist=float((np.array(ph[-1]['wrist'])[:2]-wrist0[:2])@h),
      p95=max(v['p95'] for v in segments),path=max(v['path'] for v in segments),slip_cost=float(np.mean([2*v['native_terms']['penalty_slippage'] for v in cs])),
      aux=float(np.mean([v['reward_components']['native_aux'] for v in cs])),contact=float(np.mean([v['reward_components']['contact_aux'] for v in cs])),force=float(np.sqrt(np.mean([np.dot(r['force'],r['force']) for r in ph]))))
def rank(a,b):
    if a is None or b is None:return dict(status='INCOMPLETE_OR_UNMEASURED')
    if min(a['root'],b['root'])<.05 or any(abs(a[k]-b[k])>max(.03,.1*max(abs(a[k]),abs(b[k]))) for k in ['root','wrist']):return dict(status='UNMATCHED_PROGRESS')
    clean,dirty=sorted([a,b],key=lambda v:(not(v['p95']<=.05 and v['path']<=.02),v['p95'])); comparable=(dirty['p95']>0 and (dirty['p95']-clean['p95'])>=.3*dirty['p95']) or (clean['p95']<=.05 and clean['path']<=.02 and (dirty['p95']>.05 or dirty['path']>.02))
    return dict(status='MATCHED' if comparable else 'EFFECT_UNRESOLVED',clean='baseline' if clean is a else 'alternative',proxy_weak_or_reversed=dirty['slip_cost']-clean['slip_cost']<=.1*abs(a['slip_cost']),aux_prefers_clean=clean['aux']>dirty['aux'])
rows=[]
for p in sorted((s/'d1').glob('*-U.gz')):
    a=metric(read(p))
    for name in ['B0','T']:
        other=p.with_name(p.name[:-4]+name+'.gz')
        if other.exists():
            b=metric(read(other));rows.append(dict(test='D1',case=p.name[:2],candidate=name,a=a,b=b,**rank(a,b)))
for p in sorted((s/'d2').glob('*-continue-walk.gz')):
    q=s/'reference'/p.name.replace('-walk','');a=metric(read(q));b=metric(read(p));rows.append(dict(test='D2',case=p.name[:2],candidate='walk',a=a,b=b,**rank(a,b)))
(s/'comparisons.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
