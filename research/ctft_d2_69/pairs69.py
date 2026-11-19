"""Only complete, matched two-second windows enter mechanism denominators."""
import gzip,pickle,sys,os,json,math,hashlib
from pathlib import Path
import numpy as np,mujoco
root=Path(os.environ['CTFT_PINNED_RUNTIME']);from score_endpoint import material,stance_segments
import csv
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
    force=np.array([r['force'] for r in ph]);w=np.array([r['wrist'] for r in ph]);anc=np.array([r['anchor'] for r in ph]); err=anc-w
    projected=np.c_[force[:,:2]@h,force[:,:2]@[-h[1],h[0]],force[:,2]]
    loaded=np.linalg.norm(force,axis=1)>=1;v=projected[loaded]
    return dict(segments=segments,forward_N=projected[:,0].mean(),lateral_N=projected[:,1].mean(),vertical_N=projected[:,2].mean(),forward_frac=float(np.mean((v[:,0]>0)&(v[:,0]>=abs(v[:,1]))&(v[:,0]>=abs(v[:,2])))) if len(v) else None,height_m=-err[:,2].mean(),lag_forward_m=(err[:,:2]@h).mean(),lag_rms_m=np.sqrt(np.mean((err**2).sum(axis=1))),root=float((ph[-1]['qpos'][:2]-q0[:2])@h),wrist=float((np.array(ph[-1]['wrist'])[:2]-wrist0[:2])@h),
      p95=max(v['p95'] for v in segments),path=max(v['path'] for v in segments),slip_cost=float(np.mean([2*v['native_terms']['penalty_slippage'] for v in cs])),
      aux=float(np.mean([v['reward_components']['native_aux'] for v in cs])),contact=float(np.mean([v['reward_components']['contact_aux'] for v in cs])),force=float(np.sqrt(np.mean([np.dot(r['force'],r['force']) for r in ph]))))
rows=[];spans={};matches=[]
def stoprelease(j):return bool(j['complete_release'] and not j['hard'] and j['release_root_path']<=.15 and j['release_yaw']<=15)
for src in sorted((s/'reference').glob('[0-9][0-9]-*.json')):
    series=src.stem; arm=[src,s/'d2'/(series+'-walk.json')]
    result={};win=[];meta=[]
    for label,p in zip('AB',arm):
        x=read(p.with_suffix('.gz'));j=json.loads(p.read_text());meta.append(j);v=metric(x);win.append(v)
        spans[series+label]={'first2s':v['segments'] if v else [],'full':j['stances'],'phases':j['stages']}
        result.update({label+'_'+k:j[k] for k in ['hard','slip_result','root_forward','alternating_support_changes','root_path','yaw_abs_deg','force_rms','lag_rms','stopped','release_root_path','release_yaw','complete_release','joint_limit_excess','qpd_step_peak_rad','actual_q_step_peak_rad']})
        result[label+'_release_pass']=stoprelease(j)
        result.update({label+'_two_s_'+k:vv for k,vv in (v or {}).items() if k!='segments'})
    a,b=win
    ok=bool(a and b and not any(j['hard'] for j in meta) and min(a['root'],b['root'])>=.05 and all(abs(a[k]-b[k])<=max(.03,.1*max(abs(a[k]),abs(b[k]))) for k in ['root','wrist']))
    entry=dict(series=series,matched=ok,**result);rows.append(entry)
    if ok and series.endswith('-continue'):matches.append(series)
keys=list(dict.fromkeys(k for row in rows for k in row))
with (s/'pair24.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
(s/'stance-phase.json').write_text(json.dumps(spans,allow_nan=False))
(s/'denominator.json').write_text(json.dumps(matches))
