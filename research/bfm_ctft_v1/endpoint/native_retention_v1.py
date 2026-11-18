"""Native g0 executed tracking segment, using actual prelocked 50 Hz motion references."""
from pathlib import Path
import gzip,hashlib,json,os,pickle,sys,time
import numpy as np
import torch
import mujoco
import joblib
from score_endpoint_v1 import ROOT,write
folder=Path(sys.argv[1]);stage=ROOT.parent;started=time.monotonic()
plan=json.loads((stage/"locks/source-qualification-v1.json").read_bytes())
refdir=stage/"runs/s0-cas-reference-02/reference"; refmanifest=json.loads((refdir/"manifest.json").read_bytes())
assert hashlib.sha256((refdir/"episodes.pt").read_bytes()).hexdigest()==refmanifest["episode_sha256"]
ref=torch.load(refdir/"episodes.pt",map_location="cpu",weights_only=False)
raw=joblib.load(plan["asset"])
model=mujoco.MjModel.from_xml_path(str(ROOT/"humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml"))
rows=[]
for j,item in enumerate(plan["cases"]):
    assert item["ancestor"]==refmanifest["cases"][j]["ancestor"]
    reference=ref[j]["observation"]["state"][:,:29].numpy()
    parent=f"ctft-native-regression-{j:02d}"
    for name in ("B0","U","T"):
        with gzip.open(folder/f"{parent}-{name}-native_continuation.pkl.gz","rb") as f:x=pickle.load(f)
        contract=x["contract"];offset=np.array(contract["q_PD_offset"])[0];joints=contract["dof_names"]
        addresses=[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,k)] for k in joints]
        # Verify reference coordinate offset/scale using actual archived raw
        # frames at identical 30/50Hz times; interpolation-free each 100ms.
        actualref=reference+offset
        rawq=np.asarray(raw[item["motion"]]["dof"])
        dif=float(np.max(abs(rawq[:181:3]-actualref[:301:5])))
        q=[];target=[]
        for k,c in enumerate(x["controls"]):
            source_control=k+25
            if source_control<75 or source_control>=375 or len(x["physics"])<4*(k+1):continue
            q.append(np.array(x["physics"][4*k+3]["qpos"])[addresses]);target.append(actualref[source_control-75+1])
        e=np.array(q)-target if q else None
        rows.append(dict(parent=parent,model=name,reference_validation_max_rad=dif,n=len(q),complete=len(q)==300,reference_q_rmse_rad=float(np.sqrt((e*e).mean())) if len(q) else None,hard=x["hard"] is not None))
write(folder/"native-motion-retention.json",dict(rows=rows,seconds=time.monotonic()-started,window="actual original six-second motion execution segment; incomplete tails flagged and not treated as equal exposure"))
