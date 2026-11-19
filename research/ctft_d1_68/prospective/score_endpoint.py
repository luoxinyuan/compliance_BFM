"""Physical score from actual 5ms rows; no model updates or episode selection."""
import gzip
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import mujoco
import os,time
ROOT=Path(os.environ["CTFT_PINNED_RUNTIME"])
sys.path.insert(0,str(ROOT/"research/bfm_ctft_v1"))
def write(path,obj):
    tmp=path.with_suffix(path.suffix+".partial")
    tmp.write_text(json.dumps(obj,allow_nan=False,indent=2)+"\n");tmp.rename(path)
from quality import stance_segments

def material(model, ph, q0, segments):
    data=mujoco.MjData(model)
    result=[]
    for seg in segments:
        if not seg["qualified"]:continue
        # A material point born in this stance is tracked only through its
        # original sphere-ground contact and the ground intersection cap.
        tracks={};paths=[];speeds=[];missing=0
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
            if current:speeds.append(max(current))
            else:missing+=1
        for _,sums in tracks.values():paths.extend(sums)
        measured=bool(speeds and paths and not missing)
        result.append(dict(**seg,measured=measured,missing=missing,p95=float(np.percentile(speeds,95)) if speeds else None,
            path=float(max(paths)) if paths else None))
    return result

def score(x,m):
    ph=x["physics"];spec=x["spec"];q0=x["initial_qpos"];start=x["start"];dt=.005
    def array(key):return np.asarray([r[key] for r in ph])
    q=array("qpos");root=q[:,:2];v=array("qvel")[:,:2];f=array("force")
    yaw=np.unwrap(np.arctan2(array("pelvis_matrix")[:,1,0],array("pelvis_matrix")[:,0,0]))*180/np.pi
    load=array("foot_normal_force_n");weight=np.sum(m.body_mass)*np.linalg.norm(m.opt.gravity)
    seg=stance_segments(load,weight);stances=material(m,ph,q0,seg)
    qual=bool(stances and all(s["measured"] and s["p95"]<=.05 and s["path"]<=.02 for s in stances))
    if not all(s["measured"] for s in stances) or not stances: slip_result="UNMEASURED"
    else:slip_result="PASS" if qual else "FAIL"
    joints=np.asarray([c["action"].reshape(-1) for c in x["controls"]])
    pd=x["contract"];scale=np.array(pd["q_PD_scale"]).reshape(-1);target=np.array(pd["q_PD_offset"]).reshape(-1)+np.clip(5*joints,-pd["clip"],pd["clip"])*scale/5
    t=array("force_time")-start
    res=dict(model=x["model"],parent=spec["parent"],branch=spec["branch"],hard=x["hard"] is not None,
        terminal_time=float(t[-1]+dt),slip_result=slip_result,stances=stances,old_slip_peak=float(array("contact_tangent_speed_max_mps").max()),
        old_slip_p95=float(np.percentile(array("contact_tangent_speed_max_mps"),95)),old_slip_over_s=float(dt*np.sum(array("contact_tangent_speed_max_mps")>.45)),
        yaw_abs_deg=float(np.max(abs(yaw-yaw[0]))),root_path=float(np.linalg.norm(np.diff(np.vstack((q0[:2],root)),axis=0),axis=1).sum()),
        qpd_step_peak_rad=float(abs(np.diff(target,axis=0)).max()) if len(target)>1 else None,
        actual_q_step_peak_rad=float(abs(np.diff(np.vstack((q0,q)),axis=0)[:,7:]).max()),
        joint_limit_excess=float(array("joint_limit_excess_max_rad").max()),hard_reason=None if x["hard"] is None else {k:x["hard"].get(k) for k in ["ctft_stop_reason","tilt_deg","height","joint_limit_excess_max_rad","undesired_ground_contacts","failed"]})
    if not spec["mode"]:return res
    p=spec["partner"];ang=np.deg2rad(p["direction_deg"]);direc=np.array([np.cos(ang),np.sin(ang)])
    dec=p["distance_m"]/p["plateau_speed_mps"];hold=dec+p["accel_s"];release=hold+p["hold_s"]
    wrist=array("wrist");anchor=array("anchor");lag=anchor-wrist;fmag=np.linalg.norm(f,axis=1)
    stages={}
    for key,sel in [("moving",t<dec),("decelerating",(t>=dec)&(t<hold)),("holding",(t>=hold)&(t<release)),("released",t>=release)]:
        if np.any(sel):
            stages[key]=dict(ticks=int(sel.sum()),root_speed_mean=float(np.linalg.norm(v[sel],axis=1).mean()),
                force_xyz_rms=np.sqrt((f[sel]**2).mean(0)).tolist(),force_rms=float(np.sqrt((fmag[sel]**2).mean())),
                lag_xyz_rms=np.sqrt((lag[sel]**2).mean(0)).tolist() if key!="released" else None)
    active=(t<release);steps=[]
    left,right=load.T>=.05*weight
    for tick in np.where(active)[0]:
        if left[tick]!=right[tick]:
            foot=int(right[tick])
            if not steps or steps[-1]!=foot:steps.append(foot)
    ir=np.where(t>=release)[0]
    recovered=len(ir)>=798
    res.update(stages=stages,complete_release=recovered,alternating_support_changes=max(0,len(steps)-1),
        root_forward=float((root[active][-1]-q0[:2])@direc),wrist_progress_ratio=float((wrist[active][-1,:2]-anchor[0,:2])@direc/p["distance_m"]),
        lag_rms=float(np.sqrt((lag[active]**2).sum(1).mean())),force_rms=float(np.sqrt((fmag[active]**2).mean())),
        saturation=float((fmag[active]>=10-1e-8).mean()),peak_force=float(fmag.max()),p95_force=float(np.percentile(fmag[active],95)),
        stopped=False,release_root_path=None,release_yaw=None)
    speed=np.linalg.norm(v,axis=1)
    for i in np.where((t>=dec)&(t<=dec+2))[0]:
        if i+200<=len(t) and np.all(speed[i:i+200]<.05):res["stopped"]=True;break
    if len(ir):
        k=max(0,ir[0]-1);res["release_root_path"]=float(np.linalg.norm(np.diff(root[k:],axis=0),axis=1).sum());res["release_yaw"]=float(abs(yaw[k:]-yaw[k]).max())
    for label, stage in stages.items():
        bounds={"moving":(0,dec),"decelerating":(dec,hold),"holding":(hold,release),"released":(release,float("inf"))}[label]
        cs=[c for c in x["controls"] if bounds[0]-1e-9<=c["time"]-start<bounds[1]-1e-9 and "reward_components" in c]
        if cs:
            stage["mean_contact_reward"]=float(np.mean([c["reward_components"]["contact_aux"] for c in cs]))
    return res

if __name__=="__main__":
    folder=Path(sys.argv[1])
    model=mujoco.MjModel.from_xml_path(str(ROOT/"humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml"))
    begun=time.monotonic()
    for p in sorted(folder.glob("ctft-*.pkl.gz")):
        dest=folder/(p.name+".metrics.json")
        if dest.exists():continue
        with gzip.open(p,"rb") as stream:x=pickle.load(stream)
        write(dest,score(x,model))
    print("wall_s",time.monotonic()-begun,flush=True)
