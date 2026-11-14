"""Fresh zero-U v1.0 train-only deterministic and sampled physical qualification."""
import argparse,json,sys,gzip,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
P=ROOT.parents[2]

def quality(ph):
    import numpy as np
    if not ph:return None
    yaw=np.unwrap([math.atan2(p['pelvis_matrix'][1][0],p['pelvis_matrix'][0][0]) for p in ph])
    root=np.array([p['root'][:2] for p in ph])
    return dict(slip_peak=max(p['stage_loaded_slip_mps'] for p in ph),
        slip_p95=float(np.quantile([p['stage_loaded_slip_mps'] for p in ph],.95)),
        yaw_total_variation_deg=float(np.degrees(np.abs(np.diff(yaw)).sum())),
        yaw_net_deg=float(np.degrees(yaw[-1]-yaw[0])),root_path_m=float(np.linalg.norm(np.diff(root,axis=0),axis=1).sum()),
        unsupported_ticks=sum(not any(p['support']) for p in ph),
        max_joint_excess_rad=max(p['joint_limit_excess_max_rad'] for p in ph))

def study(e,m,c,out):
    import torch,numpy as np
    from native_env import install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from research.interaction_fb.env import Environment
    from research.interaction_fb.approved_agent import ApprovedAgent
    from research.interaction_fb.evaluation import summarize
    from research.interaction_fb.schema import deploy,privileged
    from research.interaction_fb.journal import dump
    from diagnostic_fb_resume import tree_digest
    from dual_track_task import global_rng,restore_global_rng
    from f1_mjdata_contract import install_restore_guard
    from frl_v3_counterfactual import install_contact_details
    from f1_sequence_search import install_post_velocity
    from diagnostic_frozen import portable
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    install();speed(e);(out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix');install_restore_guard()
    lock=json.loads((ROOT/'research_notes/r2_ei_approved_entry_lock.json').read_text())
    selected=[]
    for p in lock['qualification_parents']:
        if ARGS.mode=='sample' and p['parent_id'] not in lock['original_six_parent_ids']:continue
        for sp in p['specs']:
            sp=dict(sp,split='train')
            if p['family']=='stop':sp['pair_branch_at']=p['specs'][0]['stop_time']
            selected.append(sp)
    env=Environment(e,m,c,selected);env.reset(0);obs=env.proxy.enrich(env.task.extended_observation())
    scale=torch.tensor(c['q_PD_scale'],device='cuda');a=ApprovedAgent(m,obs,e.standing,scale,'E',lock)
    b=ApprovedAgent(m,obs,e.standing,scale,'I',lock)
    assert all(not opt.state for ag in (a,b) for opt in ag.opts.values())
    for k in ('u','goal_delta','q1','q2'):
        assert tree_digest(a.nets[k].state_dict())==tree_digest(b.nets[k].state_dict())
    assert tree_digest(a.nets['goal_encoder'].state_dict())==tree_digest(b.nets['bi'].state_dict())
    assert tree_digest(a.tail.state_dict())==tree_digest(b.tail.state_dict())
    align=dict(common_tensors_equal=True,optimizer_states_empty=True,no_historical_guidance_loaded=True,
        actor_digest=tree_digest(m._actor.state_dict()),tail_digest=tree_digest(a.tail.state_dict()),
        normalizer_digest=tree_digest(m._obs_normalizer.state_dict()),u_digest=tree_digest(a.nets['u'].state_dict()),
        goal_encoder_digest=tree_digest(a.nets['goal_encoder'].state_dict()),E_initial_digest=tree_digest(a.state()),I_initial_digest=tree_digest(b.state()),
        seed=lock['initialization_seed'],checked_control_steps=0,mode=ARGS.mode)
    (out/'initial-alignment.json').write_text(json.dumps(align,indent=2))
    execute=a.action
    @torch.no_grad()
    def checked_action(ob,target=False,sample=False,z_override=None):
        saved=global_rng();act,z,zi=execute(ob,target,sample,z_override);after=global_rng()
        restore_global_rng(saved);aa,zz,ii=b.action(ob,target,sample,z_override)
        assert torch.equal(act,aa) and torch.equal(z,zz) and torch.equal(zi,ii)
        assert tree_digest(global_rng())==tree_digest(after)
        assert torch.equal(a.inp(ob,act,z,zi),b.inp(ob,aa,zz,ii))
        for k in ('q1','q2'):
            assert torch.equal(a.nets[k](a.inp(ob,act,z,zi)),b.nets[k](b.inp(ob,aa,zz,ii)))
            assert (a.nets[k](a.inp(ob,act,z,zi))<=0).all()
        hidden={k:v.clone() for k,v in ob.items()};hidden['privileged_state']+=100
        for lo,hi in [(9,15),(18,22)]:hidden['contact_privileged_v1'][:,lo:hi]+=100
        assert torch.equal(deploy(ob),deploy(hidden))
        assert torch.equal(a.command(ob),a.command(hidden))
        mean=execute(ob,target,False,z_override)[0]
        if sample:
            restore_global_rng(saved)
            expected=(mean+(.01*torch.randn_like(mean)).clamp(-.06,.06)).clamp(-1,1)
            assert torch.equal(expected,act) and tree_digest(global_rng())==tree_digest(after)
        else:assert torch.equal(act,mean)
        assert torch.equal(act*scale,aa*scale),'Actual q_PD mapping differs'
        restore_global_rng(after);align['checked_control_steps']+=1
        return act,z,zi
    a.action=checked_action;env.proxy.agent=a;env.proxy.stochastic=ARGS.mode=='sample'
    install_contact_details(e);install_post_velocity(e)
    before_a=tree_digest(a.state());before_b=tree_digest(b.state());results=[];prefixes={};native_stats=[]
    for i,spec in enumerate(selected):
        torch.manual_seed(lock['physics_evaluation_seed']);np.random.seed(lock['physics_evaluation_seed'])
        env.reset(i);env.proxy.agent=a;row_start=len(env.rows);dest=out/f'case-{i}';dest.mkdir()
        # Complete start and RNG retained; this is a physical reference, never PPO data.
        dump(dest/'start-state.ptz',dict(physical=portable(env.task),rng=global_rng(),spec=spec))
        ns=env.proxy.source_pairs
        nq=np.array([(r['action'][0]*scale.cpu()).numpy() for r in ns]);steps=np.diff(nq,axis=0)
        native_stats.append(dict(parent_id=spec['parent_id'],source_sha=env.task.prefix_data['audit']['sha256'],
            p90_qpd_step_rms=float(np.quantile(np.sqrt(np.mean(steps**2,axis=1)),.9)),
            p95_single_joint_step=float(np.quantile(np.abs(steps).reshape(-1),.95))))
        while not env.task.done:
            try:
                with torch.no_grad():env.collect_call()
            except RuntimeError as exc:
                if not str(exc).startswith('HARD_PHYSICS_BOUNDARY:'):raise
                break
        t=env.task;r=summarize(t);r['spec']=spec;r['source_state_sha']=t.prefix_data['audit']['sha256']
        r['whole_quality']=quality(t.physics);r['recovery_quality']=quality([p for p in t.physics if p['time']>=spec['release']])
        whole=r['whole_quality'];rec=r['recovery_quality'];family=spec['demand_family']
        errors=[]
        if t.boundary is not None:errors.append('hard_physics_boundary')
        if family in ('none','zero'):
            for key,bound in [('slip_peak',.10),('yaw_total_variation_deg',15.),('root_path_m',.10),('unsupported_ticks',0)]:
                if whole[key]>bound:errors.append('none_zero_'+key)
        r['recovery_qualified']=bool(rec and not t.boundary and rec['slip_peak']<=.45 and rec['yaw_total_variation_deg']<=15 and rec['unsupported_ticks']==0)
        # Contract 3.3: an initially failing reference does not authorize moving its thresholds.
        if not r['recovery_qualified']:errors.append('frozen_recovery_reference_failed')
        r['protection_failures']=errors
        controls=[{k:x[k] for k in ('bfm_obs','executed_z','action','q_PD','pre_qpos','pre_qvel')} for x in t.controls if x['time']<spec.get('pair_branch_at',-1)-1e-8]
        if 'pair_branch_at' in spec:
            pid=spec['parent_id'];dh=tree_digest(controls)
            if pid in prefixes:assert prefixes[pid]==dh
            prefixes[pid]=dh
        with gzip.open(dest/'actual.json.gz','wt') as f:json.dump(dict(spec=spec,physics=t.physics,controls=t.controls),f)
        dump(dest/'reference-data.ptz',dict(rows=env.rows[row_start:],native=ns,parent_id=spec['parent_id'],
            recovery_qualified=r['recovery_qualified'],source_sha=r['source_state_sha']))
        (dest/'evaluation.json').write_text(json.dumps(r,indent=2,allow_nan=False));results.append(r)
        (out/'completed-cases.json').write_text(json.dumps(results,indent=2,allow_nan=False))
        (out/'initial-alignment.json').write_text(json.dumps(align,indent=2))
        print('QUALIFIED_CASE',i,family,json.dumps(whole),'FAILURES',errors,flush=True)
        if errors:break
        from research.interaction_fb import benchmark
        if benchmark.STOP_REQUESTED:break
    assert tree_digest(a.state())==before_a and tree_digest(b.state())==before_b
    unique={x['parent_id']:x for x in native_stats}
    eps=dict(eps_rms=float(np.clip(.25*np.median([x['p90_qpd_step_rms'] for x in unique.values()]),.015,.050)),
        eps_p95=float(np.clip(.5*np.median([x['p95_single_joint_step'] for x in unique.values()]),.030,.100)),
        parent_count=len(unique),source='actual native prefix qPD changes',native_statistics=list(unique.values()))
    (out/'native-retention-scale.json').write_text(json.dumps(eps,indent=2))
    result=dict(status='qualification_passed' if len(results)==len(selected) and all(not x['protection_failures'] for x in results) else 'qualification_stopped',
        mode=ARGS.mode,expected_episodes=len(selected),episodes=len(results),results=results,alignment=align,
        optimizer_updates=0,formal_training_transitions=0,sealed_executed=False,retention_scale=eps)
    (out/'qualification.json').write_text(json.dumps(result,indent=2,allow_nan=False));return result

def main():
    global ARGS
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['mean','sample'],required=True);ARGS=p.parse_args()
    from research.interaction_fb import benchmark
    benchmark.study=study;sys.argv=[__file__,'--output',str(ARGS.output),'--arm','A'];benchmark.main()
if __name__=='__main__':main()
