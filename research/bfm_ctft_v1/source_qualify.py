"""CTFT S0 finite original-motion source pool qualification.

Uses the verified original physics/bootstrap loader only; no old learner/weights
or controller is instantiated. Original B maps real motion observations to native
tracking z. Every body/history state comes from original Actor/PD execution.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes'),str(Path(__file__).parent)]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def prepare():
    import joblib
    asset=Path('/mnt/data-oss/users/yi.gao/compliance_BFM/data/lafan_29dof_10s-clipped.pkl')
    x=joblib.load(asset);groups={}
    for key in x:
        ancestor=key.rsplit('_clip',1)[0]
        if ancestor.startswith('walk'):
            groups.setdefault(ancestor,[]).append(key)
    cases=[]
    for ancestor,keys in sorted(groups.items()):
        key=min(keys,key=lambda k:int(k.rsplit('_clip',1)[1]))
        cases.append(dict(ancestor=ancestor,motion=key,start=0,fps=int(x[key]['fps']),
                          role='source_qualification_not_formal_training_or_evaluation'))
    assert len(cases)==12
    plan=dict(version='CTFT_SOURCE_QUALIFICATION_12_ORIGINAL_WALK_LINEAGES_V1',
        status='prelocked_before_physics',asset=str(asset),asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        ancestors=12,cases=cases,selection='all12 walk ancestors, first chronological10s clip only; no physical-result selection',
        native_codec='original get_backward_observation + original tracking_inference; first301 real reference states at50Hz',
        source_command=dict(settle_until_s=3.,entry_s=1.5,execute_s=6.,exit_s=1.5,recovery_s=4.),
        transition_rule=dict(standing_speed_mps=.05,walking_speed_mps=.15,
            walking_sustained_s=.2,requires_real_left_right_support_switch=True,
            before_window_s=[-1.,-.1],at_window_s=[-.1,.3],after_window_s=[.3,1.]),
        load_fraction=.05,no_support_limit_s=.05,offline_stance_min_s=.10,
        snapshot_every_control=True,new_physics_ceiling=50000,
        no_qpos_only_handoff=True,no_old_learned_weights=True,
        warning='12 ancestors remain12, not source-frame count. Final train/eval ancestry qualification is separate and NOT yet passed.')
    dest=STAGE/'locks/source-qualification-v1.json';dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists();write(dest,plan);print(json.dumps(dict(plan=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),ancestors=12)))


def capture_state(e,last_z,source_progress):
    import numpy as np,torch,mujoco
    from rolling_snapshot import state_of
    mask=mujoco.mjtState.mjSTATE_INTEGRATION
    integration=np.empty(mujoco.mj_stateSize(e.sim.model,mask))
    mujoco.mj_getState(e.sim.model,e.sim.data,integration,mask)
    d=e.sim.data
    # Data-only evidence; actual restore qualification remains a separate gate.
    arrays={k:getattr(d,k).copy() for k in dir(d) if not k.startswith('_') and isinstance(getattr(d,k),np.ndarray)}
    return dict(integration=integration,mjdata_arrays=arrays,
        env=state_of(e,exclude=('trace',)),base=state_of(e.base),sim=state_of(e.sim),
        wrapped=state_of(e.wrapped),tether=state_of(e.tether),
        bfm_history=copy.deepcopy(e.base.history_handler.history),
        actual_last_z=last_z.detach().cpu().clone(),native_source_progress=source_progress,
        python_rng=__import__('random').getstate(),numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())


def study(e,model,contract,out):
    import numpy as np,torch,mujoco
    from m8_stage import reset_common,advance
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from locomotion_rollout import spherical,UnsafePhysics
    from humanoidverse.utils.helpers import get_backward_observation
    from quality import OnlineSupport,stance_segments
    speed(e);(out/'backend-prefix').mkdir();backend(e,model,out/'backend-prefix')
    plan=json.loads((STAGE/'locks/source-qualification-v1.json').read_text())
    m=e.sim.model;d=e.sim.data
    weight=float(m.body_mass.sum()*np.linalg.norm(m.opt.gravity))
    assert abs(m.opt.timestep-.005)<1e-12 and abs(e.dt-.02)<1e-12
    monitor=OnlineSupport(weight);enabled=False;last_command=e.standing.clone()
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
    original_hook=e.sim.simulate_at_each_physics_step
    def guarded():
        original_hook()
        state=monitor.update(e.contact_sample['foot_normal_force_n'])
        if e.trace:e.trace[-1]['ctft_online_support']=state
        if enabled and state['stop']:
            row=copy.deepcopy(e.trace[-1]);row['ctft_stop_reason']='continuous_no_effective_support_gt_50ms'
            raise UnsafePhysics(row)
    e.sim.simulate_at_each_physics_step=guarded
    brief=[];lib=e.base._motion_lib
    scale=torch.as_tensor(contract['q_PD_scale'],device=e.standing.device)
    off=torch.as_tensor(contract['q_PD_offset'],device=e.standing.device)
    actor_sha_before=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model._actor.parameters())).hexdigest()
    for i,case in enumerate(plan['cases']):
        enabled=False
        reset_common(e,model,model.act)
        # ComplianceEnv.reset restores its own callback; rebind the explicit
        # causal guard after the real prefix, before any qualified source tick.
        e.sim.simulate_at_each_physics_step=guarded
        monitor=OnlineSupport(weight)
        for r in e.trace:monitor.update(r['foot_normal_force_n'])
        e.cfg.enabled=False;e.tether.cfg.enabled=False;e.qualification_active=True
        source_idx=lib._motion_data_keys.tolist().index(case['motion'])
        lib.load_motions(random_sample=False,start_idx=source_idx,num_motions_to_load=1)
        obs,_=get_backward_observation(e.base,0,use_root_height_obs=bool(e.base.config.obs.get('root_height_obs',True)))
        indices=torch.arange(1,302,device=e.standing.device)
        zz=model.tracking_inference({k:v[indices] for k,v in obs.items()})
        assert torch.isfinite(zz).all() and float((zz.norm(dim=-1)-16).abs().max())<1e-4
        cs=[];states=[];start=len(e.trace);boundary=None;enabled=True
        begin=float(e.tether.time);last_command=e.standing.clone()
        try:
            for step in range(round(13./.02)):
                t=float(e.tether.time)-begin
                if t<1.5:z=spherical(e.standing,zz[:1],min(1.,max(0.,t/1.5))**2*(3.-2.*min(1.,max(0.,t/1.5))))
                elif t<7.5:z=zz[min(round((t-1.5)/.02),len(zz)-1):min(round((t-1.5)/.02),len(zz)-1)+1]
                elif t<9.:
                    u=min(1.,max(0.,(t-7.5)/1.5));z=spherical(zz[-1:],e.standing,u*u*(3.-2.*u))
                else:z=e.standing
                saved=capture_state(e,last_command,dict(source_case=i,control=step))
                saved['online_support']=dict(vars(monitor))
                states.append(saved)
                a=model.act(e.obs,z,mean=True)
                assert torch.isfinite(a).all()
                qpd=off+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
                cs.append(dict(time=float(e.tether.time),z=z.detach().cpu(),previous_z=last_command.detach().cpu(),
                    action=a.detach().cpu(),q_PD=qpd.detach().cpu(),
                    bfm_observation={k:v.detach().cpu().clone() for k,v in e.obs.items()}))
                last_command=z.clone();e.previous_z=z.clone()
                if advance(e,a):boundary=dict(kind='native_termination',time=float(e.tether.time));break
        except UnsafePhysics as exc:
            boundary=dict(kind='native_source_hard_boundary',physics=exc.row)
        finally:enabled=False;e.qualification_active=False
        ph=copy.deepcopy(e.trace[start:]);assert all(np.linalg.norm(r['force'])==0 for r in ph)
        # Complete private evidence includes failed sources and every saved state.
        import gzip,pickle
        dest=out/f'source-{i:02d}.pkl.gz'
        with gzip.open(dest,'wb',compresslevel=1) as f:
            pickle.dump(dict(case=case,controls=cs,states=states,physics=ph,boundary=boundary,
                native_z=zz.detach().cpu(),identity='original_frozen_native_source_execution'),f,protocol=5)
        speeds=np.linalg.norm(np.array([r['qvel'][:2] for r in ph]),axis=1)
        valid_speed=speeds>.15
        long_speed=np.convolve(valid_speed.astype(np.int32),np.ones(40,dtype=np.int32),'valid') if len(valid_speed)>=40 else np.array([])
        support=np.array([r['foot_normal_force_n'] for r in ph])>=.05*weight
        labels=np.where(support[:,0]&~support[:,1],0,np.where(support[:,1]&~support[:,0],1,-1))
        seq=[int(v) for j,v in enumerate(labels) if v>=0 and (j==0 or v!=labels[j-1])]
        switches=sum(a!=b for a,b in zip(seq,seq[1:]))
        record=dict(case=i,ancestor=case['ancestor'],motion=case['motion'],boundary_kind=None if boundary is None else boundary['kind'],
            complete=boundary is None,physics_rows=len(ph),controls=len(cs),
            moving_sustained_200ms=bool(len(long_speed) and long_speed.max()>=40),
            root_progress_xy=(np.array(ph[-1]['root'][:2])-np.array(ph[0]['root'][:2])).tolist(),
            support_switches=switches,longest_no_support_s=monitor.maximum_no_support_ticks*.005,
            evidence_file=dest.name,evidence_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
            source_restoration_qualified=False)
        brief.append(record);write(out/'progress.json',brief);print('CTFT_NATIVE_SOURCE',json.dumps(record),flush=True)
        del states,cs,ph
    actor_after=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model._actor.parameters())).hexdigest()
    assert actor_after==actor_sha_before
    usable=[r for r in brief if r['complete'] and r['moving_sustained_200ms'] and r['support_switches']>=2]
    result=dict(status='SOURCE_CANDIDATES_RECORDED_RESTORE_PENDING' if len(usable)>=12 else 'DATA_SUPPORT_BLOCKED',
        sources=brief,usable_original_ancestors=len(usable),required_original_ancestors=12,
        body_states_from_real_native_execution=True,all_failed_sources_preserved=True,
        original_actor_unchanged=True,source_state_restore_qualification='NOT_RUN',
        new_training_samples=0,agent_updates=0)
    write(out/'source-summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.prepare_only:return prepare()
    assert a.output;a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'launch.json',dict(pid=os.getpid(),runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        argv=sys.argv,no_old_learned_weights=True,purpose='S0 frozen original native source qualification',started_unix=time.time()))
    os.environ.update(BFM_RECOVERY_OUTPUT=str(a.output),CCL_DEADLINE_MONOTONIC=str(time.monotonic()+5400))
    import mujoco
    raw_step=mujoco.mj_step;count={'actual':0,'reserved':0}
    def counted(*args,**kw):
        n=int(kw.get('nstep',1));count['reserved']+=n
        assert count['reserved']<=50000
        result=raw_step(*args,**kw);count['actual']+=n;return result
    mujoco.mj_step=counted
    import m10_stage,m10_physics
    m10_stage.run_study=study;m10_physics.OUT=a.output
    begun=time.monotonic()
    try:m10_physics.main()
    finally:
        write(a.output/'physics-counter.json',count)
        write(a.output/'cost.json',dict(wall_s=time.monotonic()-begun,physics=count,pid=os.getpid()))


if __name__=='__main__':main()
