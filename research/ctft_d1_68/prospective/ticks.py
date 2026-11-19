import mujoco,numpy as np,sys,json,gzip,pickle,os
from pathlib import Path
def material(model, ph, q0, segments):
    data=mujoco.MjData(model)
    result=[]
    for seg in segments:
        if not seg["qualified"]:continue
        # A material point born in this stance is tracked only through its
        # original sphere-ground contact and the ground intersection cap.
        tracks={};paths=[];speeds=[];missing=0;first_v=None;first_path=None
        for i in range(seg["start"],seg["end_exclusive"]):
            row=ph[i]
            data.qpos[:]=q0 if i==0 else ph[i-1]["qpos"]
            mujoco.mj_kinematics(model,data)
            geometry={}
            for c in row["ctft_material_contacts"]:
                if c["foot"]!=seg["foot"] or c["normal_n"]<=0:continue
                gid,ground=sorted(c["geom_pair"],key=lambda g:int(model.geom_bodyid[g])==0)
                body=c["body"]
                assert model.geom_bodyid[gid]==body and model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_SPHERE
                assert model.geom_bodyid[ground]==0 and model.geom_type[ground]==mujoco.mjtGeom.mjGEOM_PLANE and model.geom_pos[ground,2]==0
                pos,R=data.xpos[body].copy(),data.xmat[body].reshape(3,3).copy()
                seed=model.geom_pos[gid]-R.T@np.array([0.,0.,model.geom_size[gid,0]])
                surf=pos+R@seed
                if surf[2]>64*np.finfo(float).eps*max(1.,np.abs(surf).max()):
                    missing+=1
                    continue
                geometry[gid]=(body,pos,R,seed)
            for gid in list(tracks):
                if gid not in geometry:
                    paths.extend(tracks.pop(gid)[1])
            data.qpos[:]=row["qpos"];data.qvel[:]=row["qvel"]
            mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
            current=[]
            for gid,(body,pos,R,seed) in geometry.items():
                seeds,sums=tracks.get(gid,(np.empty((0,3)),np.empty(0)))
                seeds=np.vstack((seeds,seed));sums=np.r_[sums,0.]
                pre=seeds@R.T+pos
                valid=pre[:,2]<=64*np.finfo(float).eps*np.maximum(1.,np.abs(pre).max(axis=1))
                paths.extend(sums[~valid])
                newR=data.xmat[body].reshape(3,3);newpos=data.xpos[body]
                post=seeds[valid]@newR.T+newpos
                tracks[gid]=(seeds[valid],sums[valid]+np.linalg.norm((post-pre[valid])[:,:2],axis=1))
                jac=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jac,None,newpos+newR@seed,body)
                current.append(np.linalg.norm((jac@data.qvel)[:2]))
            if current and max(current)>.05 and first_v is None:first_v=i
            if first_path is None and any(np.any(t[1]>.02) for t in tracks.values()):first_path=i
            if current:speeds.append(max(current))
            else:missing+=1
        for _,sums in tracks.values():paths.extend(sums)
        measured=bool(speeds and paths and not missing)
        result.append(dict(**seg,first_instant_v_gt_005=first_v,first_accum_path_gt_002=first_path,measured=measured,missing=missing,p95=float(np.percentile(speeds,95)) if speeds else None,
            path=float(max(paths)) if paths else None))
    return result

s=Path(sys.argv[1]);root=Path(os.environ['CTFT_PINNED_RUNTIME']);m=mujoco.MjModel.from_xml_path(str(root/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'));res=[]
for met in sorted((s/'reference').glob('[0-9]*.json')):
    mm=json.loads(met.read_bytes())
    with gzip.open(met.with_suffix('.gz'),'rb') as f:x=pickle.load(f)
    for ok in (False,True):
        opts=[v for v in mm['stances'] if v['measured'] and (v['p95']<=.05 and v['path']<=.02)==ok]
        if not opts:continue
        v=min(opts,key=lambda q:(q['start'],q['foot']))
        raw={k:v[k] for k in ['start','end_exclusive','qualified','foot']};w=material(m,x['physics'],x['initial_qpos'],[raw])[0]
        assert w['p95']==v['p95'] and w['path']==v['path']
        cs=x['controls'][v['start']//4:(v['end_exclusive']+3)//4]
        vals=[c['native_terms']['penalty_slippage'] for c in cs if 'native_terms' in c]
        res.append(dict(series=met.stem,qualified_clean=ok,raw_slip_mean=float(np.mean(vals)) if vals else None,**w))
(s/'stance-ticks.json').write_text(json.dumps(res,indent=2))
