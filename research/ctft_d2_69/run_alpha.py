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
    import random
    from f1_mjdata_contract import metadata,inactive,exact
    from score_endpoint import score
    scope=Path(__file__).resolve().parent
    mode=os.environ['DIAG_MODE'];lock=load(scope/'lock/lock.json')
    assert hashlib.sha256((scope/'lock/lock.json').read_bytes()).hexdigest()==os.environ['LOCK_SHA']
    assert len(lock['picks'])>=6
    z_walk=torch.from_numpy(np.load(scope/'lock/walking.npy')).reshape(1,-1).to(e.standing.device)
    assert mode=='d2' and len(lock['picks'])==8
    assert hashlib.sha256((scope/'lock/walking.npy').read_bytes()).hexdigest()==lock['walking']['z_file_sha256']
    assert lock['walking']['case']==0 and lock['walking']['control']==273
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
    for parent in lock['picks']:
        src=parent['source']; case=src['case']; idx=parent['control']
        path=STAGE/'runs/s0-native-sources-03'/src['evidence_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==src['source_sha256']
        live.cache.clear(); x=read(path);live.sources={case:src};live.cache['formal_native_source',case]=x
        for branch,partner in parent['branches'].items():
            series=f'{case:02d}-{branch}'; ref=scope/'reference'/series
            spec=dict(parent=f'd68-{case:02d}',source=dict(case=case,control=idx),mode=1,force=branch!='no_force',branch=branch,partner=partner)
            if mode=='d1' and branch!='continue':continue
            if mode=='d2' and not (scope/'replay/complete.json').exists():raise ValueError('New-process replay not complete')
            for candidate in (['U','B0','T'] if mode=='d1' else ['walk'] if mode=='d2' else ['U']):
                agent._model._actor.load_state_dict(states['U'],strict=True)
                before=x['states'][idx]
                live.start(spec);phys_restore(before)
                if mode!='reference':
                    baseline=read(ref.with_suffix('.gz')); met=load(ref.with_suffix('.json'))
                    bad=[v for v in met['stances'] if v['measured'] and (v['p95']>.05 or v['path']>.02)]
                    bad.sort(key=lambda v:(v['start'],v['foot']))
                    which=max(0,(bad[0]['start']-1)//4) if bad else None
                    if mode=='d1' and (which is None or which+100>len(baseline['controls'])):continue
                    snap=read((scope/'replay'/f'{series}-event.gz') if mode=='d1' else ref.with_suffix('.state.gz'))
                    live.restore(snap);phys_restore(snap);e.trace=[]
                else:
                    which=None;dump(ref.with_suffix('.state.gz'),live.snapshot())
                start_i=live.i; q0=d.qpos.copy(); start=float(e.tether.time); cs=[];hard=None
                if mode=='d1' and candidate!='U':agent._model._actor.load_state_dict(states[candidate],strict=True)
                # Increment before the first command so failed attempts are counted.
                charge('branches',1)
                while not live.done and (mode!='d1' or len(cs)<100):
                    i=live.i
                    if mode=='replay' and which==i and branch=='continue':dump(out/f'{series}-event.gz',live.snapshot())
                    assert torch.equal(live.command().reshape_as(e.standing),e.standing)
                    obs=live.observation();z=z_alpha if mode=='d2' and i<50 else live.command()
                    with torch.no_grad():a=agent.act(obs,z,mean=True)
                    c=dict(time=float(e.tether.time),action=a.cpu().numpy(),z=z.cpu().numpy())
                    cs.append(c)
                    try:r=live.step(a,obs,z)
                    except UnsafePhysics as exc:hard=exc.row;break
                    c['reward_components']={k:v.item() for k,v in r['ctft_reward_components'].items()}
                    c['native_terms']={k:float(e.base.extras['aux_rewards'][k].reshape(-1)[0]) for k in coeff}
                    assert float(sum(c['native_terms'][k]*v for k,v in coeff.items()))==c['reward_components']['native_aux']
                    if mode=='d1' and len(cs)==1 and candidate!='U':agent._model._actor.load_state_dict(states['U'],strict=True)
                    if i%100==0:assert shutil.disk_usage(scope).free>=100*2**30
                    if i==0:assert sum(f.stat().st_size for f in scope.rglob('*') if f.is_file())+1024**3<32*2**30
                packet=dict(spec=spec,initial_qpos=q0,start=start,model=candidate,model_sha=ids[candidate] if candidate in ids else ids['U'],physics=e.trace,controls=cs,contract=contract,hard=hard)
                if mode=='replay' or (mode=='d1' and candidate=='U'):
                    st=start_i*4; old=baseline['physics'][st:st+len(e.trace)]; assert len(old)==len(e.trace)
                    for j,(a0,b0) in enumerate(zip(old,e.trace)):
                        if not eq(a0,b0):raise RuntimeError(('replay',series,j,[k for k in a0 if not eq(a0[k],b0[k])]))
                    assert eq(baseline['controls'][start_i:start_i+len(cs)],cs)
                    if mode=='replay':assert len(e.trace)==len(baseline['physics'])
                if mode!='replay':
                    name=series if mode=='reference' else f'{series}-{candidate}'
                    dump(out/f'{name}.gz',packet)
                    if mode!='d1':write(out/f'{name}.json',score(packet,m))
                log.append(dict(series=series,candidate=candidate,controls=len(cs),ticks=len(e.trace),hard=hard is not None,start_control=start_i))
                write(out/'progress.json',log)
                
        del x
    write(out/'complete.json',log)
    return dict(status='COMPLETE',branches=len(log),updates=0)

def charge(key,n):
    ctr[key]+=n
    assert ctr[key]+previous[key]<=dict(ticks=400000,branches=72)[key]
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
