"""Immutable endpoint rollout collector. Uses pinned simulation and records failed tails."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(os.environ["CTFT_PINNED_RUNTIME"])
STAGE = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/"research_notes"), str(ROOT/"research/bfm_ctft_v1")]
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import m11_ipc
from cas_checkpoint import durable

def write(path, value):
    durable(path, (json.dumps(value, indent=2, allow_nan=False)+"\n").encode())

def load(path):
    return json.loads(path.read_text())

def study(e, original, contract, out):
    import numpy as np
    import torch
    import mujoco
    from agent import build
    from learning_checkpoint import digest, set_rng
    from m8_stage import reset_common
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from training_live import TrainingLive
    from packed_checkpoint import PackedTensorStore
    from locomotion_rollout import UnsafePhysics, spherical
    speed(e)
    (out/"backend-prefix").mkdir()
    backend(e, original, out/"backend-prefix")
    reset_common(e, original, original.act)
    m,d=e.sim.model,e.sim.data
    native_physics=e._original_step
    def record_material_points():
        pre_velocity=d.qvel.copy();native_physics()
        material=[]
        for i,c in enumerate(d.contact[:d.ncon]):
            b=[int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])]
            if 0 not in b:continue
            body=max(b);name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,body) or ''
            if 'ankle' not in name and 'foot' not in name:continue
            f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
            jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,c.pos,body)
            R=d.xmat[body].reshape(3,3)
            material.append(dict(foot=0 if 'left' in name else 1,body=body,
                geom_pair=[int(c.geom1),int(c.geom2)],normal_n=float(abs(f[0])),
                material_local=(R.T@(c.pos-d.xpos[body])).tolist(),contact_world=c.pos.tolist(),
                body_world=d.xpos[body].tolist(),body_rotation=R.tolist(),
                tangent_velocity=(c.frame.reshape(3,3)[1:]@(jac@pre_velocity)).tolist()))
        e.contact_sample['ctft_material_contacts']=material
    e._original_step=record_material_points
    manifest_path=STAGE/"locks/final-evaluation-manifest-v2.json"
    manifest=load(manifest_path)
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()=="8c70be20c4ec08414985f1567beb6968d03a105cabcebb899ed7212fb3b80575"
    coeff=load(STAGE/"runs/s0-online-probe-01/agent-identity.json")["original_training_config"]["aux_rewards_scaling"]
    with torch.enable_grad():
        agent,_=build(aux_scale=load(STAGE/"locks/common-aux-scale-v1.json")["value"],compile_execution=False)
    states={"B0":{k:v.cpu().clone() for k,v in agent._model._actor.state_dict().items()}}
    ids={"B0":digest(states["B0"])}
    assert ids["B0"]==load(STAGE/"locks/bfm-ctft-v1a-signed.json")["frozen_actor_sha256"]
    for name in ("U","T"):
        store=PackedTensorStore(STAGE/f"runs/formal-{name}-v1a-01/store")
        ck="round-131072"
        receipt=load(store.root/ck/"readback.json")
        assert receipt["complete"] and receipt["all_tensor_raw_hashes_verified"]
        assert receipt["state_sha256"]==load(store.root/ck/"manifest.json")["state_sha256"]
        model=store.metadata(ck)["learner"]["model"]
        states[name]={k[7:]:store.unpack(v) for k,v in model.items() if k.startswith("_actor.")}
        for k,v in agent._model.state_dict().items():
            if k.startswith("_obs_normalizer."):
                assert torch.equal(v.cpu(),store.unpack(model[k]).cpu()),k
        ids[name]=digest(states[name])
    write(out/"model-identities.json",ids)
    live=TrainingLive(e,STAGE,coeff,[])
    progress=[]
    for parent in [*manifest["native_g0_evaluation"],*manifest["parents"]]:
        native="source_control" in parent
        src=parent["source"]
        directory="s0-native-sources-03" if native else "s0-holdout-sources-02"
        idx=parent["source_control"] if native else src["selected_control"]
        path=STAGE/"runs"/directory/src["evidence_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==src.get("source_sha256",src.get("evidence_sha256"))
        live.cache.clear()
        with gzip.open(path,"rb") as stream: x=pickle.load(stream)
        case=src["case"]; live.sources={case:src};live.cache["formal_native_source",case]=x
        saved=x["states"][idx]
        branches=parent.get("branches",[dict(branch="native_continuation",mode=0,apply_force=False)])
        for name,state in states.items():
            agent._model._actor.load_state_dict(state,strict=True)
            prefixes={}
            for b in branches:
                assert sum(r["bytes"] for r in progress)<7*2**30
                assert shutil.disk_usage(STAGE).free>=100*2**30
                spec=dict(parent=parent["parent_id"],source=dict(case=case,control=idx),mode=b["mode"],force=b["apply_force"],branch=b["branch"])
                if not native:spec["partner"]=b["partner"]
                import random
                random.setstate(saved["python_rng"]);np.random.set_state(saved["numpy_rng"]);torch.set_rng_state(saved["torch_rng"]);torch.cuda.set_rng_state_all(saved["cuda_rng"])
                live.start(spec)
                q0=d.qpos.copy(); start=float(e.tether.time)
                if native:
                    # Preserve the original exogenous schedule beyond a baseline hard stop.
                    cmds=[r["z"] for r in x["controls"][idx:]]
                    zz=x["native_z"].to(e.standing.device)
                    for j in range(idx+len(cmds),650):
                        t=j*.02
                        if t<1.5:
                            u=t/1.5;z=spherical(e.standing,zz[:1],u*u*(3-2*u))
                        elif t<7.5:z=zz[min(round((t-1.5)/.02),len(zz)-1):min(round((t-1.5)/.02),len(zz)-1)+1]
                        elif t<9:
                            u=(t-7.5)/1.5;z=spherical(zz[-1:],e.standing,u*u*(3-2*u))
                        else:z=e.standing
                        cmds.append(z.cpu())
                    live.native_commands=cmds;live.controls=len(cmds)
                controls=[];hard=None
                while not live.done:
                    obs=live.observation();z=live.command()
                    with torch.no_grad():action=agent.act(obs,z,mean=True)
                    if native and name=="B0" and live.i<12:
                        assert torch.equal(action.cpu(),x["controls"][idx+live.i]["action"]),("native canary",case,live.i)
                    controls.append(dict(time=float(e.tether.time),action=action.cpu().numpy(),z=z.cpu().numpy()))
                    try:row=live.step(action,obs,z)
                    except UnsafePhysics as exc:
                        hard=exc.row;break
                    controls[-1]["reward_components"]={k:v.item() for k,v in row["ctft_reward_components"].items()}
                assert e.trace and len(e.trace)<=4*len(controls)
                if not native and spec["force"]:
                    prefix=[np.asarray(r["qpos"]).tobytes()+np.asarray(r["qvel"]).tobytes()+np.asarray(r["force"]).tobytes() for r in e.trace if r["time"]-start<parent["common_prefix_s"]-1e-9]
                    if prefixes:
                        assert prefix==prefixes["continue"],("actual sibling prefix",parent["parent_id"],name)
                    prefixes[b["branch"]]=prefix
                dest=out/(parent["parent_id"]+"-"+name+"-"+b["branch"]+".pkl.gz")
                tmp=dest.with_suffix(".partial")
                with gzip.open(tmp,"wb",compresslevel=1) as f:
                    pickle.dump(dict(spec=spec,model=name,model_sha=ids[name],initial_qpos=q0,start=start,physics=e.trace,controls=controls,hard=hard,contract=contract),f,protocol=5)
                tmp.rename(dest)
                progress.append(dict(parent=parent["parent_id"],model=name,branch=b["branch"],ticks=len(e.trace),controls=len(controls),hard=hard is not None,bytes=dest.stat().st_size))
                write(out/"progress.json",progress)
        del x
    assert len(progress)==252
    write(out/"rollouts-complete.json",dict(episodes=252,updates=0))
    return dict(episodes=252)

def main():
    import mujoco
    from resource_guard import check
    out=Path(sys.argv[1]).resolve()
    out.mkdir(exist_ok=False)
    start=time.monotonic();guard=check(STAGE);counts=dict(actual=0,reserved=0)
    os.environ.update(BFM_RECOVERY_OUTPUT=str(out),CCL_DEADLINE_MONOTONIC=str(start+14400))
    write(out/"launch.json",dict(pid=os.getpid(),resource=guard,source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),started=time.time()))
    raw=mujoco.mj_step
    def step(*a,**kw):
        n=int(kw.get("nstep",1));counts["reserved"]+=n
        assert counts["reserved"]<=950000 and time.monotonic()-start<14400
        result=raw(*a,**kw);counts["actual"]+=n;return result
    mujoco.mj_step=step
    import m10_physics,m10_stage
    m10_stage.run_study=study;m10_physics.OUT=out
    try:m10_physics.main()
    finally:write(out/"cost.json",dict(wall_s=time.monotonic()-start,physics=counts,updates=0))
if __name__=="__main__":main()
