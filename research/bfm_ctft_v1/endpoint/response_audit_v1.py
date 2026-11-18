"""Reconstruct each recorded auxiliary contact reward before interpreting its wrist term."""
import os,sys,time,gzip,pickle
from pathlib import Path
import numpy as np
import mujoco
from score_endpoint_v1 import ROOT,write
from reward import ForceEMA,contact_cost

def audit(x,model):
    spec=x["spec"]; d=mujoco.MjData(model)
    bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"right_wrist_yaw_link")
    ema=ForceEMA();rows=[];p=spec["partner"]
    dec=p["distance_m"]/p["plateau_speed_mps"]; hold=dec+.5;rel=hold+p["hold_s"]
    for j,c in enumerate(x["controls"]):
        if "reward_components" not in c:continue
        r=x["physics"][4*j+3]
        d.qpos[:]=r["qpos"];d.qvel[:]=r["qvel"];mujoco.mj_kinematics(model,d);mujoco.mj_comPos(model,d)
        jac=np.zeros((3,model.nv));mujoco.mj_jac(model,d,jac,None,d.xpos[bid],bid)
        force=np.asarray(r["force"]); vw=jac@d.qvel
        cost=contact_cost(filtered_force_n=ema.update(force),applied_force_n=force,wrist_velocity_mps=vw,root_velocity_mps=d.qvel[:3],guidance_mode=1)
        error=abs(cost["contact_native_reward"]-c["reward_components"]["contact_aux"])
        t=c["time"]-x["start"]
        ph="moving" if t<dec else "decelerating" if t<hold else "holding" if t<rel else "released"
        rows.append((ph,error,cost["parts"]["response"],c["reward_components"]["contact_aux"],c["reward_components"]["native_aux"]))
    phases={}
    for phase in ["moving","decelerating","holding","released"]:
        a=np.array([r[1:] for r in rows if r[0]==phase]); 
        if len(a):phases[phase]=dict(n=len(a),max_reward_reconstruction_difference=float(a[:,0].max()),mean_response_cost=float(a[:,1].mean()),mean_contact_reward=float(a[:,2].mean()),mean_native_aux=float(a[:,3].mean()))
    return phases
if __name__=="__main__":
    folder=Path(sys.argv[1]);start=time.monotonic()
    model=mujoco.MjModel.from_xml_path(str(ROOT/"humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml"))
    for p in sorted(folder.glob("ctft-eval-*.pkl.gz")):
        dest=folder/(p.name+".reward.json")
        if dest.exists():continue
        with gzip.open(p,"rb") as f:x=pickle.load(f)
        write(dest,audit(x,model))
    print("wall_s",time.monotonic()-start)
