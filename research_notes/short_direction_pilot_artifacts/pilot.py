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
    from short_live import ShortGuidanceLive as TrainingLive
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
    write(out/"model-identities.json",ids)
    from pathlib import Path
    import hashlib, random, gzip, pickle
    from learning_checkpoint import rng,set_rng
    from native_reference import NativeReference
    from training_data import PhaseReplay
    from contact_replay import cat
    from prefix_check import check_fork
    import training_live as tl,contact_interface_probe as ci
    from f1_mjdata_contract import metadata,inactive,exact
    from collections import Counter
    from locomotion_rollout import UnsafePhysics
    scope=Path(__file__).resolve().parent
    zeros=load(scope.parent/'ctft_start_pair_01/origins.json')
    train_ids=[1,2,3,4,5,6,7,11];dev_ids=[9,10]
    origins={v['case']:v['first'] for v in zeros}
    records={v['case']:v for v in load(STAGE/'runs/s0-native-sources-03/source-summary.json')['sources']}
    live=TrainingLive(e,STAGE,coeff,[])
    original_partner=ci.partner
    def fork_partner(p,t):
        pos,vel=original_partner(p,t)
        if 'fork_deg' not in p or pos[0]<=.025:return pos,vel
        angle=np.deg2rad(p['fork_deg']);direction=np.array([np.cos(angle),np.sin(angle),0.])
        u=pos[0]-.025;r=.015;b=min(u,r)
        gain=(b*b/(2*r)+max(0.,u-r))*(direction-np.array([1.,0.,0.]))
        return pos+gain,vel+vel[0]*min(u/r,1.)*(direction-np.array([1.,0.,0.]))
    ci.partner=tl.partner=fork_partner
    model=agent._model
    import copy
    b0=copy.deepcopy(model._actor).requires_grad_(False)
    reference=STAGE/'runs/s0-cas-reference-02/reference'
    rm=load(reference/'manifest.json')
    from learning_checkpoint import file_sha
    assert file_sha(reference/'episodes.pt')==rm['episode_sha256']
    episodes=torch.load(reference/'episodes.pt',map_location='cpu',weights_only=False)
    allowed={records[i]['ancestor'] for i in train_ids}
    keep=[i for i,r in enumerate(rm['cases']) if r['ancestor'] in allowed]
    assert len(keep)==len(train_ids)
    expert=NativeReference([episodes[i] for i in keep]);del episodes
    replay=PhaseReplay(capacity=16400,quotas={0:.25,1:.75*1.8/7.8,2:.75*2/7.8,3:.75*4/7.8})
    original_sample=replay.sample
    def audited_sample(size):
        data=original_sample(size)
        labels=data['short_source_branch_stage_loaded'].tolist()
        mix=Counter(tuple(x) for x in labels)
        with (out/'sampled.jsonl').open('a') as stream:
            print(json.dumps(dict(update=updates,counts={str(k):v for k,v in mix.items()})),file=stream)
        return data
    replay.sample=audited_sample
    branches=('native','left','front','right','zero','none','intermediate','fork_left','fork_right')
    branch_names=branches
    buffers=dict(train=replay,expert_slicer=expert)
    seed=27301;plan_rng=np.random.default_rng(seed);plan=[]
    for cycle in range(30):
        double=['left','front','right'][cycle%3]
        branches=['left','front','right','zero','none',double,'native','native']
        for branch in plan_rng.permutation(branches):
            plan.append((int(plan_rng.choice(train_ids)),str(branch),float(plan_rng.uniform(.05,.10))))
    write(out/'plan.json',plan)
    def start(case,branch,distance):
        rec=records[case];key=('formal_native_source',case)
        if key not in live.cache:
            live.cache.clear();f=STAGE/'runs/s0-native-sources-03'/rec['evidence_file']
            assert file_sha(f)==rec['evidence_sha256']
            with gzip.open(f,'rb') as stream:live.cache[key]=pickle.load(stream)
        live.sources={case:rec}
        ix=50 if branch=='native' else origins[case]
        before=live.cache[key]['states'][ix]
        angle={'left':30,'front':0,'right':-30,'intermediate':15}.get(branch,0)
        stationary=branch=='zero'
        p=dict(distance_m=0. if stationary else distance,plateau_speed_mps=.05,accel_s=0. if stationary else .5,hold_s=2.5 if stationary else 2.,direction_deg=angle,stiffness=80.)
        if branch in ('fork_left','fork_right'):p['fork_deg']=30 if branch=='fork_left' else -30
        spec=dict(source=dict(case=case,control=ix),parent=rec['ancestor'],branch=branch,mode=int(branch!='native'),force=branch not in ('native','none'),partner=p)
        live.start(spec)
        mask=mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_setState(m,d,before['integration'],mask);skip=inactive(metadata(m,d))
        for k,val in before['mjdata_arrays'].items():
            if k not in skip and getattr(d,k).flags.writeable:getattr(d,k)[...]=val
        mujoco.mj_setState(m,d,before['integration'],mask)
        assert all(exact(val,getattr(d,k)) for k,val in before['mjdata_arrays'].items() if k not in skip)
        return spec
    def actual(spec,training):
        obs=live.observation();z=live.command()
        with torch.no_grad():
            if training and spec['mode']==0:
                with torch.autocast(device_type=model.device,dtype=model.amp_dtype,enabled=model.cfg.amp):
                    a=b0(model._normalize(obs),z,model.cfg.actor_std).mean.float()
            else:a=agent.act(obs,z,mean=not training)
        row=live.step(a,obs,z)
        ph=e.trace[-4:];assert len(ph)==4
        parts=live.last_reward_parts
        new=sum(parts.values())
        assert torch.equal(torch.tensor([[new]],dtype=torch.float32),row['aux_rewards']['ctft_total_native'])
        loaded=any(r['active'] and np.linalg.norm(r['force'])>0 for r in ph)
        if spec['branch']=='none':assert all(np.linalg.norm(r['force'])==0 for r in ph)
        row['short_source_branch_stage_loaded']=torch.tensor([[spec['source']['case'],branch_names.index(spec['branch']),row['ctft_replay_phase'].item(),int(loaded)]])
        if spec['branch'].startswith('fork'):
            inp=digest(([obs[k] for k in model._actor.input_filter.cfg.key],z))
            action=digest(a)
            for r in ph:r.update(actor_input_or_hash=inp,executed_action_or_hash=action)
        return row,new
    def eval_node(n):
        saved=rng();res=[]
        for case in dev_ids:
            prefix=None
            for branch in ('left','front','right','fork_left','fork_right','intermediate','none','zero','native'):
                set_rng(saved)
                spec=start(case,branch,.08);rewards=[];hard=None
                sid=digest((records[case]['evidence_sha256'],spec['source'],d.qpos.copy(),d.qvel.copy(), live.history.state(),live.last_z))
                rid=digest(rng())
                try:
                    while not live.done:
                        _,v=actual(spec,False);rewards.append(v)
                except UnsafePhysics as exc:hard=str(exc)
                ph=e.trace;force=np.array([r['force'] for r in ph]);active=np.array([r['active'] for r in ph]);w=np.array([r['wrist'] for r in ph]);an=np.array([r['anchor'] for r in ph]);sh=np.array([r['shoulder_xyz'] for r in ph]);roots=np.array([r['root'] for r in ph])
                disp=w-sh;last=np.flatnonzero(active)[-1] if active.any() else -1
                if branch.startswith('fork') and hard is None:
                    if prefix is None:prefix=(ph,sid,rid)
                    else:
                        fork=check_fork(prefix[0],ph,source_sha_a=prefix[1],source_sha_b=sid,rng_sha_a=prefix[2],rng_sha_b=rid)
                        write(out/f'fork-{n}-{case}.json',dict(ticks=fork[0],time=fork[1],source=sid,rng=rid))
                item=dict(case=case,branch=branch,hard=hard,steps=len(rewards),reward_per_step=float(np.mean(rewards)) if rewards else None,
                    error_cm=100*float(np.mean(np.sum((w[active]-an[active])**2,axis=1))**.5) if active.any() else None,
                    force_n=float(np.mean(np.sum(force**2,axis=1)[active])**.5) if active.any() else 0.,
                    arm_delta_cm=((disp[last]-disp[0])*100).tolist(),root_path_cm=100*np.linalg.norm(np.diff(roots[:,:2],axis=0),axis=1).sum())
                res.append(item);write(out/f'dev-{n}.json',res)
                with gzip.open(out/f'dev-{n}-{case}-{branch}.gz','wb',compresslevel=1) as f:pickle.dump(ph,f,protocol=5)
                if hard:raise RuntimeError(('dev hard boundary',item))
        set_rng(saved);return res
    def checkpoint(n):
        path=out/f'node-{n}.pt'
        state=dict(n=n,model=model.state_dict(),optimizers=agent.optimizer_dict,rng=rng(),z_buffer=dict(storage=agent.z_buffer._storage,idx=agent.z_buffer._idx,full=agent.z_buffer._is_full),actor_enabled=agent.actor_enabled,update_count=updates,contracts=['LOCK.md','PILOT_LOCK.md'])
        temp=path.with_suffix('.partial');torch.save(state,temp);temp.rename(path);write(out/f'node-{n}-identity.json',dict(sha=file_sha(path)))
    updates=0;n=0;pending=[];timeline=[];hist=Counter()
    checkpoint(0);base=eval_node(0);next_node=12288
    for ordinal,(case,branch,distance) in enumerate(plan):
        spec=start(case,branch,distance);rs=[];hard=None
        while not live.done and n<16384:
            try:row,value=actual(spec,True)
            except UnsafePhysics as exc:
                with gzip.open(out/'physical-stop.gz','wb') as f:pickle.dump(dict(spec=spec,physics=e.trace,reason=exc.row),f)
                write(out/'stopped.json',dict(n=n,case=case,branch=branch,updates=updates,reason='train hard physics'))
                return dict(status='HARD_STOP',n=n,updates=updates)
            rs.append(value);hist[(branch,int(row['ctft_replay_phase'].item()))]+=1
            pending.append(row);n+=1
            if n%1024==0:
                data=cat(pending);pending.clear();replay.extend(data)
                torch.save(data,out/f'rows-{n}.pt')
                agent.actor_enabled=n>8192
                with torch.enable_grad():
                    for _ in range(8):
                        metrics=agent.update(buffers,updates);updates+=1
                        with (out/'updates.jsonl').open('a') as stream:
                            print(json.dumps(dict(n=n,update=updates,metrics={k:float(v) for k,v in metrics.items() if torch.is_tensor(v) and v.numel()==1}),allow_nan=False),file=stream)
                        assert all(torch.isfinite(v).all() for v in metrics.values() if torch.is_tensor(v))
                assert shutil.disk_usage(out).free>100*2**30
                assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<25*2**30
                print('BLOCK',n,updates,flush=True);write(out/'coverage.json',dict(n=n,updates=updates,distribution={str(k):v for k,v in hist.items()}))
        timeline.append(dict(n=n,branch=branch,case=case,complete=live.done,decisions=len(rs),reward_per_step=float(np.mean(rs))));write(out/'training.json',timeline)
        if n>=next_node:
            checkpoint(n);dev=eval_node(n)
            if n==16384:break
            next_node=16384
    return dict(status='COMPLETE',n=n,updates=updates)

def charge(key,n):
    ctr[key]+=n
    assert ctr[key]+previous[key]<=dict(ticks=250000,branches=400)[key]
    if ctr[key]%2000==1:
        # Persist the pending reservation, including abnormal process termination.
        write(OUT/'cost.json',dict(ticks=ctr['ticks'],branches=ctr['branches'],seconds=time.monotonic()-zero))
    assert time.monotonic()-zero<=limit
if __name__=='__main__':
    import mujoco
    from resource_guard import check
    mode=os.environ['DIAG_MODE'];OUT=Path(__file__).resolve().parent/mode
    OUT.mkdir(exist_ok=False); zero=time.monotonic();ctr=dict(ticks=0,branches=0);previous=dict(ticks=0,branches=0)
    for f in OUT.parent.glob('cue*/cost.json'):
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
