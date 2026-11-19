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
    coeff=load(STAGE/"runs/s0-online-probe-01/agent-identity.json")["original_training_config"]["aux_rewards_scaling"]
    with torch.enable_grad():
        agent,_=build(aux_scale=load(STAGE/"locks/common-aux-scale-v1.json")["value"],compile_execution=False)
    states={"B0":{k:v.cpu().clone() for k,v in agent._model._actor.state_dict().items()}}
    ids={"B0":digest(states["B0"])}
    assert ids["B0"]==load(STAGE/"locks/bfm-ctft-v1a-signed.json")["frozen_actor_sha256"]
    for name in ("U",):
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
    import random
    from f1_mjdata_contract import metadata,inactive,exact
    from score_endpoint import score
    scope=Path(__file__).resolve().parent
    ref=scope.parent/'ctft_d2_69/lock'
    lock=load(ref/'lock.json')
    assert hashlib.sha256((ref/'walking.npy').read_bytes()).hexdigest()==lock['walking']['z_file_sha256']
    z_walk=torch.from_numpy(np.load(ref/'walking.npy')).reshape(1,-1).to(e.standing.device)
    zs=e.standing.reshape(1,-1).double();zw=z_walk.double()
    assert zs.shape==zw.shape==(1,256)
    assert max(abs(v.norm().item()-16) for v in [zs,zw])<1e-4,'LATENT_INVALID'
    u=zs/zs.norm();v=zw/zw.norm();angle=torch.acos((u*v).sum().clamp(-1,1))
    assert torch.isfinite(angle) and angle.sin().abs().item()>1e-6,'LATENT_INVALID'
    z_alpha=(16*(torch.sin(.85*angle)*u+torch.sin(.15*angle)*v)/angle.sin()).to(z_walk.dtype)
    assert abs(z_alpha.norm().item()-16)<1e-4,'LATENT_INVALID'
    np.save(out/'command.npy',z_alpha.cpu().numpy())
    write(out/'command.json',dict(alpha=.15,norm=z_alpha.norm().item(),full_angle_rad=angle.item(),identity=hashlib.sha256((out/'command.npy').read_bytes()).hexdigest()))
    def dump(path,x):
        with gzip.open(path.with_suffix('.tmp'),'wb',compresslevel=1) as f:pickle.dump(x,f,protocol=5)
        path.with_suffix('.tmp').rename(path)
    def read(path):
        with gzip.open(path,'rb') as f:return pickle.load(f)
    def phys_restore(saved):
        mask=mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_setState(m,d,saved['integration'],mask); skip=inactive(metadata(m,d))
        for k,v in saved['mjdata_arrays'].items():
            if k not in skip:
                dest=getattr(d,k)
                if dest.flags.writeable and dest.shape==v.shape:dest[...]=v
        mujoco.mj_setState(m,d,saved['integration'],mask)
        assert all(exact(v,getattr(d,k)) for k,v in saved['mjdata_arrays'].items() if k not in skip)
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    def eq(a,b):
        if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(eq(i,j) for i,j in zip(a,b))
        if isinstance(a,np.ndarray):return np.array_equal(a,b,equal_nan=True)
        return a==b
    log=[]
    picks=[v for v in load(scope/'origins.json') if v['first'] is not None]
    assert len(set(v['ancestor'] for v in picks))==len(picks) and len(picks)>=2
    srcdir=STAGE/'runs/s0-native-sources-03'
    records={v['case']:v for v in load(srcdir/'source-summary.json')['sources']}
    agent._model._actor.load_state_dict(states['U'],strict=True)
    for v in picks[:3]:
        case=v['case'];i0=v['first'];src=records[case];f=srcdir/src['evidence_file']
        assert hashlib.sha256(f.read_bytes()).hexdigest()==src['evidence_sha256']
        live.cache.clear();x=read(f);live.cache['formal_native_source',case]=x;live.sources={case:src}
        before=x['states'][i0];assert torch.equal(before['actual_last_z'].reshape_as(e.standing).to(e.standing.device),e.standing)
        for branch in ['continue','stop','no_force']:
            partner=dict(distance_m=.12 if branch=='stop' else .24,plateau_speed_mps=.08,accel_s=.5,hold_s=2.,stiffness=80.,direction_deg=0.)
            spec=dict(parent=v['ancestor'],source=dict(case=case,control=i0),mode=1,force=branch!='no_force',branch=branch,partner=partner)
            live.start(spec);phys_restore(before);snap=live.snapshot()
            refph=refcs=None
            for cand,begin in [('unchanged',99999),('early',25),('late',75)]:
                live.restore(snap);phys_restore(snap);e.trace=[];cs=[];hard=None;start=float(e.tether.time);q0=d.qpos.copy()
                assert np.array_equal(q0,before['mjdata_arrays']['qpos'])
                charge('branches',1)
                while not live.done:
                    i=live.i;obs=live.observation();z=z_alpha if begin<=i<begin+50 else live.command()
                    with torch.no_grad():a=agent.act(obs,z,mean=True)
                    c=dict(time=float(e.tether.time),action=a.cpu().numpy(),z=z.cpu().numpy());cs.append(c)
                    try:r=live.step(a,obs,z)
                    except UnsafePhysics as exc:hard=exc.row;break
                    c['reward_components']={k:v.item() for k,v in r['ctft_reward_components'].items()}
                    if i%100==0:assert shutil.disk_usage(scope).free>=100*2**30
                if cand=='unchanged':refph=e.trace.copy();refcs=cs
                else:
                    k=min(4*begin,len(refph),len(e.trace));assert eq(e.trace[:k],refph[:k]) and eq(cs[:k//4],refcs[:k//4]),(case,branch,cand,'prefix')
                packet=dict(spec=spec,initial_qpos=q0,start=start,model=cand,model_sha=ids['U'],physics=e.trace,controls=cs,contract=contract,hard=hard)
                name=f'{case:02d}-{branch}-{cand}'
                dump(out/(name+'.gz'),packet);write(out/(name+'.json'),score(packet,m))
                log.append(dict(name=name,hard=bool(hard),ticks=len(e.trace)));write(out/'progress.json',log)
    write(out/'complete.json',log)
    return dict(status='COMPLETE',branches=len(log),updates=0)

def charge(key,n):
    ctr[key]+=n
    assert ctr[key]+previous[key]<=dict(ticks=65000,branches=27)[key]
    if key=='branches' or ctr[key]%200==1:
        # Persist the pending reservation, including abnormal process termination.
        write(OUT/'cost.json',dict(ticks=ctr['ticks']+199,branches=ctr['branches'],seconds=time.monotonic()-zero))
    assert time.monotonic()-zero<=limit
if __name__=='__main__':
    import mujoco
    from resource_guard import check
    mode=os.environ['DIAG_MODE'];OUT=Path(__file__).resolve().parent/mode
    OUT.mkdir(exist_ok=False); zero=time.monotonic();ctr=dict(ticks=0,branches=0);previous=dict(ticks=0,branches=0)
    for f in OUT.parent.glob('*/cost.json'):
        if f.parent!=OUT and not f.parent.is_symlink():
            for k in previous:previous[k]+=load(f).get(k,0)
    limit=float(os.environ['DIAG_TIMEOUT'])
    guard=check(OUT);write(OUT/'launch.json',dict(pid=os.getpid(),guard=guard,src=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),wall=time.time(),budget_s=limit))
    os.environ.update(BFM_RECOVERY_OUTPUT=str(OUT),CCL_DEADLINE_MONOTONIC=str(zero+limit))
    raw=mujoco.mj_step
    def step(*a,**kw):
        charge('ticks',int(kw.get('nstep',1)));return raw(*a,**kw)
    mujoco.mj_step=step
    import m10_physics,m10_stage
    m10_stage.run_study=study;m10_physics.OUT=OUT
    try:m10_physics.main()
    finally:write(OUT/'cost.json',dict(**ctr,seconds=time.monotonic()-zero))
