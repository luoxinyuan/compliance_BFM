from pathlib import Path
import csv,os,gzip,pickle,json,hashlib
import numpy as np
import mujoco as mj
from score_endpoint import material,stance_segments
s=Path.cwd()
model=mj.MjModel.from_xml_path(str(Path(os.environ['CTFT_PINNED_RUNTIME'])/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'))
data=mj.MjData(model);weight=model.body_mass.sum()*np.linalg.norm(model.opt.gravity)
body=[model.body(k).id for k in ['torso_link','right_wrist_yaw_link','left_wrist_yaw_link','right_shoulder_pitch_link']]
joints=[model.jnt_qposadr[model.joint(k).id] for k in ['right_elbow_joint','right_shoulder_pitch_joint']]
def hand(q):
    data.qpos[:]=q;mj.mj_kinematics(model,data)
    torso,wr,wl,sh=body;v=data.xmat[torso].reshape(3,3).T@(data.xpos[wr]-data.xpos[sh])
    return [*v,data.xpos[wr,2]-data.xpos[wl,2],*np.rad2deg(q[joints])]
rows=[];curves=[];prefix={};hashes={}
for group in ['reference','soft']:
    for p in sorted((s/group).glob('[0-9][0-9]-*.gz')):
        if '.state.' in p.name:continue
        with gzip.open(p,'rb') as f:x=pickle.load(f)
        hashes[group+'/'+p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        met=json.loads(p.with_suffix('.json').read_text())
        ph=x['physics'];pp=x['spec']['partner'];ang=np.deg2rad(pp['direction_deg']);h=np.array([np.cos(ang),np.sin(ang)]);l=np.array([-h[1],h[0]])
        get=lambda k:np.asarray([r[k] for r in ph])
        t=get('force_time')-x['start'];act=get('active').astype(bool);force=get('force');mag=np.linalg.norm(force,axis=1)
        fw=force[:,:2]@h;lat=force[:,:2]@l;up=force[:,2];dom=(fw>0)&(fw>=np.abs(lat))&(fw>=np.abs(up))
        q=get('qpos');root=q[:,:2];w=get('wrist');a=get('anchor');q0=x['initial_qpos'];init=hand(q0)
        wr0=data.xpos[body[1]].copy();snap=np.array([hand(v) for v in q[3::4]])
        dec=pp['distance_m']/pp['plateau_speed_mps'];rel=dec+pp['accel_s']+pp['hold_s']
        row=dict(group=group,case=x['spec']['source']['case'],branch=x['spec']['branch'],**{k:met[k] for k in ['hard','slip_result','force_rms','lag_rms','root_path','yaw_abs_deg','stopped','release_root_path','release_yaw','complete_release','alternating_support_changes','root_forward','wrist_progress_ratio']})
        for win,mask in [('early',np.arange(len(t))<400),('moving',t<dec)]:
            loaded=mask&act&(mag>=1)
            row.update({win+'_'+k:v for k,v in dict(fwd=fw[mask].mean(),lat=lat[mask].mean(),up=up[mask].mean(),frac=dom[loaded].mean() if loaded.any() else None,nticks=int(loaded.sum()),sat=(mag[mask]>=10-1e-8).mean(),height=(w-a)[mask,2].mean()).items()})
        for label,i in [('init',-1),('half',24),('two',99),('release_end',len(snap)-1)]:
            vals=init if i<0 else snap[i] if i<len(snap) else [None]*6
            for k,v in zip(['reach','lateral','vertical','heightdiff','elbow','shoulder'],vals):row[label+'_'+k]=v
        k=min(399,len(ph)-1);row['two_root']=(root[k]-q0[:2])@h;row['two_wrist']=(w[k,:2]-wr0[:2])@h
        loads=get('foot_normal_force_n')[:400];segments=material(model,ph[:400],q0,stance_segments(loads,weight))
        good=segments and all(v['measured'] for v in segments);row['two_p95']=max(v['p95'] for v in segments) if good else None;row['two_path']=max(v['path'] for v in segments) if good else None
        rows.append(row)
        for i,v in enumerate(snap):
            curves.append([group,p.stem,(i+1)*.02,int(act[4*i+3]),*v])
        if group=='soft' and row['branch'] in ['continue','stop']:
            z=prefix.setdefault(row['case'],{});z[row['branch']]=(dec,[get(k) for k in ['qpos','qvel','force','anchor','foot_normal_force_n']])
with open('paired.csv','w') as f:
    out=csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
with gzip.open('arm_curves.csv.gz','wt') as f:
    out=csv.writer(f);out.writerow(['arm','series','t_s','active','wrist_shoulder_forward_m','lateral_m','vertical_m','RminusL_height_m','elbow_deg','shoulder_pitch_deg']);out.writerows(curves)
checked={}
for case,p in prefix.items():
    n=round(p['stop'][0]/.005)
    assert all(np.array_equal(a[:n],b[:n]) for a,b in zip(p['stop'][1],p['continue'][1]))
    checked[case]=n
assert len(rows)==48 and len(checked)==8
Path('checks.json').write_text(json.dumps(dict(source_sha=hashes,prefix_ticks=checked,rows=len(rows)),indent=2))
