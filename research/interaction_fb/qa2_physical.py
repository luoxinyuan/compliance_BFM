"""QA2 prospective Q_mu ranking: frozen models, real common-state stochastic futures."""
import argparse,copy,itertools,json,os,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
S=ROOT.parent;P=ROOT.parents[2]

def write(p,x):
    tmp=p.with_suffix('.pending.json');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));os.replace(tmp,p)

def study(e,m,c,out):
    import numpy as np,torch
    from native_env import install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from research.interaction_fb.env import Environment
    from research.interaction_fb.approved_learning import ApprovedLearner
    from research.interaction_fb.approved_journal import load_state
    from research.interaction_fb.journal import dump,read
    from research.interaction_fb.schema import privileged
    from research.interaction_fb.qualification_v11 import install_gate
    from research.interaction_fb.gate_v11 import metrics
    from research.interaction_fb.qa2_value import setup
    from dual_track_runtime import TrainingBranch
    from dual_track_task import global_rng
    from rolling_snapshot import state_of,put
    from f1_mjdata_contract import canonical,metadata,exact,install_restore_guard
    from diagnostic_fb_resume import tree_digest
    from diagnostic_frozen import portable
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    import mujoco
    counted_step=mujoco.mj_step
    def bounded_step(*args,**kwargs):
        counter=out/'physics-counter.json'
        reserved=json.loads(counter.read_text())['reserved'] if counter.exists() else 0
        assert reserved+int(kwargs.get('nstep',1))<=26000,'P2_PHYSICS_BUDGET'
        return counted_step(*args,**kwargs)
    mujoco.mj_step=bounded_step
    lock=json.loads((S/'prep/p0-p1/p1.json').read_text());data=read(S/'prep/p0-p1/data.ptz',lock['data']);pros=data['prospective']
    assert len(pros)==4 and len({r['parent'] for r in pros})==4
    assert set(r['parent'] for r in pros).isdisjoint({m['parent'] for m in data['metadata']})
    install();speed(e);(out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix');install_restore_guard()
    specs=[dict(p['spec'],split='train') for p in pros];env=Environment(e,m,c,specs);env.reset(0)
    init=json.loads((ROOT/'research_notes/r2_ei_v11_entry_lock.json').read_text())
    obs=env.proxy.enrich(env.task.extended_observation());agent=ApprovedLearner(m,obs,e.standing,torch.tensor(c['q_PD_scale'],device='cuda'),'E',init)
    source=Path(data['source_reference']);src=load_state(source);agent.load(copy.deepcopy(src['agent']));frozen=tree_digest(agent.state())
    heads={};identities={}
    for name in ['V1-MC','V2-n1','V2-n5','V2-n25','V2-n50']:
        pp=S/'runs'/name/(name+'.ptz');mm=json.loads(pp.with_suffix('.manifest.json').read_text());ss=read(pp,mm)
        _,head=setup(data);head.load_state_dict(ss['model']);head.eval().requires_grad_(False);heads[name]=head;identities[name]=mm['sha256']
    assert tree_digest(heads['V1-MC'].state_dict())==tree_digest(heads['V2-n50'].state_dict())
    results=[]
    for case_index,pr in enumerate(pros):
        dest=out/f'parent-{case_index}';dest.mkdir();os.environ['BFM_RECOVERY_OUTPUT']=str(dest)
        seed=pr['prefix_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        env.proxy.agent=agent;env.proxy.stochastic=True;env.reset(case_index)
        install_gate(env,env.task.physics)
        with torch.no_grad():
            while float(e.tether.time)<pr['boundary_s']-1e-7:env.collect_call()
        assert abs(float(e.tether.time)-pr['boundary_s'])<1e-6
        assert env.task.spec['parent_id']==pr['parent'] and not env.task.done
        env.proxy.slot=0;env.proxy.start=env.task.last_command.clone();env.proxy.target=env.proxy.start.clone()
        ob=env.proxy.enrich(env.task.extended_observation())
        from clean_policy import direct_command
        with torch.no_grad():env.proxy.target=direct_command(agent.command(ob),env.task.e.standing)
        ob=env.proxy.enrich(env.task.extended_observation())
        with torch.no_grad():mean,z,zi=agent.action(ob,sample=False)
        gen=torch.Generator(device='cpu').manual_seed(pr['candidate_seed'])
        eps=[(.01*torch.randn(mean.shape,generator=gen)).clamp(-.06,.06).to(mean.device) for _ in range(2)]
        candidates=[('mean',mean),('draw0_plus',(mean+eps[0]).clamp(-1,1)),('draw0_minus',(mean-eps[0]).clamp(-1,1)),('draw1_plus',(mean+eps[1]).clamp(-1,1)),('draw1_minus',(mean-eps[1]).clamp(-1,1))]
        preds=[]
        for name,act in candidates:
            xx=torch.cat([privileged(ob),act,z,zi],-1)
            with torch.no_grad():scores={n:float(h(xx,torch.tensor([50],device='cuda'))) for n,h in heads.items()}
            preds.append(dict(candidate=name,scores=scores,action=act.cpu(),z=z.cpu(),zi=zi.cpu()))
        # All predictions are immutable before any candidate sees a future tick.
        prediction_meta=dump(dest/'predictions-before-physics.ptz',dict(predictions=preds,models=identities,observation=ob,phase_spec='current only; no future spec in head'))
        snapshot=TrainingBranch(env.task);extra=state_of(env,exclude=('task','proxy'));proxy=state_of(env.proxy)
        before=portable(env.task);bm=metadata(e.sim.model,e.sim.data);canon=canonical(before,bm);rng_digest=tree_digest(global_rng())
        snapshot_meta=dump(dest/'complete-start.ptz',dict(physical=before,mj_metadata=bm,rng=global_rng(),task_rng=copy.deepcopy(env.task.rng.bit_generator.state),physics_rng=copy.deepcopy(env.task.physics_rng),proxy=proxy,spec=env.task.spec,models=identities,source_manifest=data['source_manifest_sha256']))
        oldphys=env.task.physics;v=float(np.linalg.norm(oldphys[-1]['anchor_velocity']))
        pv=float(np.linalg.norm(oldphys[-2]['anchor_velocity'])) if len(oldphys)>1 else v
        phase='no_contact_hold' if not specs[case_index]['enabled'] else 'released_recovery' if not oldphys[-1]['active'] else 'holding' if v<=1e-8 else 'decelerating' if v<pv-1e-8 else 'moving'
        branches=[]
        for crn in pr['crn_seeds']:
            for name,act in candidates:
                snapshot.restore();put(env,extra);put(env.proxy,proxy)
                assert exact(canonical(portable(env.task),metadata(e.sim.model,e.sim.data)),canon)
                assert tree_digest(global_rng())==rng_digest
                # Exogenous action CRN is deliberately reset after matching the full
                # prefix. Physics RNG/history remain the real source state.
                random.seed(crn);np.random.seed(crn);torch.manual_seed(crn);torch.cuda.manual_seed_all(crn)
                crng=global_rng();env.proxy.agent=agent;env.proxy.stochastic=True;install_gate(env,env.task.physics)
                original=agent.action;calls=[0]
                def first_then_mu(o,target=False,sample=False,z_override=None):
                    if calls[0]==0:
                        aa,zz,ii=original(o,target=target,sample=False,z_override=z_override)
                        assert torch.equal(zz,z) and torch.equal(ii,zi)
                        assert torch.equal(privileged(o),privileged(ob))
                        calls[0]+=1;return act,zz,ii
                    calls[0]+=1
                    return original(o,target=target,sample=sample,z_override=z_override)
                agent.action=first_then_mu;rstart=len(env.rows);pstart=len(env.task.physics);cstart=len(env.task.controls);failure=None
                try:
                    with torch.no_grad():
                        for _ in range(10):env.collect_call()
                except RuntimeError as exc:
                    if not str(exc).startswith('HARD_PHYSICS_BOUNDARY:'):raise
                    failure=str(exc)
                finally:agent.action=original
                rows=env.rows[rstart:];ph=env.task.physics[pstart:];controls=env.task.controls[cstart:]
                assert len(rows)==calls[0] and torch.equal(rows[0]['action'],act.cpu())
                assert torch.equal(torch.tensor(controls[0]['q_PD']),env.task.qof(act)[0].cpu())
                w=1.;ret=0.
                for row in rows:ret+=w*float(row['aux_rewards']['contact_physical']);w*=.99**(float(row['duration'])/.1)
                tr=dict(identity='prospective_evaluation_not_training',rows=rows,physics=ph,controls=controls,first_action=act.cpu(),actual_z=z.cpu(),previous_z=ob['_previous_z'].cpu(),CRN=crn,crn_rng=crng,initial_effective_digest=tree_digest(canon),full_source=snapshot_meta,models=identities,followup_policy='frozen stochastic mu, std.01 clip.06; first20ms intervention consumes no random draw; 49 subsequent actual Gaussian calls')
                am=dump(dest/f'{name}-crn{crn}.ptz',tr)
                ac=[r for r in ph if r['active']]
                br=dict(candidate=name,crn=crn,parent=pr['parent'],return_1s=ret,complete_50_steps=len(rows)==50 and failure is None,prefix_restore_exact=True,quality=metrics(ph),force_rms=float(np.sqrt(np.mean([np.dot(r['force'],r['force']) for r in ac]))) if ac else 0.,error_rms=float(np.sqrt(np.mean([r['error']**2 for r in ac]))) if ac else None,archive=am,failure=failure)
                branches.append(br);write(dest/'branches.json',branches);print('QA2_P2_BRANCH',case_index,name,crn,ret,failure,flush=True)
                assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<=.70*1024**3,'P2_OUTPUT_BUDGET'
                assert tree_digest(agent.state())==frozen
                if failure:
                    dump(dest/'failed-physical.ptz',dict(physical=portable(env.task),rng=global_rng()));return dict(status='HARD_PHYSICS_STOP',results=results,failed_branch=br)
        result=dict(parent=pr['parent'],family=specs[case_index]['demand_family'],phase=phase,time=pr['boundary_s'],model_identities=identities,predictions=[dict(candidate=p['candidate'],scores=p['scores']) for p in preds],prediction_archive=prediction_meta,source_archive=snapshot_meta,branches=branches)
        results.append(result);write(out/'state-results.json',results)
    answer=dict(status='P2_COMPLETE',states=results,actor_updates=0,Guidance_updates=0,I_updates=0,Dev_Sealed_executed=False,formal_decision='separate immutable aggregation, no posthoc checkpoint selection',models=identities)
    write(out/'p2-result.json',answer);return answer

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from research.interaction_fb import benchmark
    import mujoco
    original=mujoco.mj_step
    def guard(*args,**kwargs):
        counter=a.output/'physics-counter.json'
        if counter.exists():assert json.loads(counter.read_text())['reserved']<=26000,'P2_PHYSICS_BUDGET'
        return original(*args,**kwargs)
    mujoco.mj_step=guard;benchmark.study=study;sys.argv=[__file__,'--output',str(a.output),'--arm','A'];benchmark.main()
if __name__=='__main__':main()
