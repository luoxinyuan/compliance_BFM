"""R2 preregistered finite-window Q qualification; no training and no sealed data."""
import argparse
import copy
import itertools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'research_notes')]
import m11_ipc
P = ROOT.parents[2]
STAGE = ROOT.parent


def adjudicate(results, lock):
    """All ten unordered comparisons per state; no selected best-action denominator."""
    answer = {}
    for arm in ('E', 'I'):
        comparisons = []
        for state in results:
            for a, b in itertools.combinations(state['branches'], 2):
                delta = a['discounted_return'] - b['discounted_return']
                qa = a['scores'][arm]['q_min']
                qb = b['scores'][arm]['q_min']
                comparisons.append(dict(parent=state['parent_id'], round=state['round'],
                    a=a['id'], b=b['id'], return_difference=delta, q_difference=qa-qb,
                    eligible=abs(delta) >= lock['return_difference_floor'],
                    correct=bool(delta * (qa-qb) > 0)))
        eligible = [r for r in comparisons if r['eligible']]
        groups = sorted({r['parent'] for r in eligible})
        accuracy = sum(r['correct'] for r in eligible) / len(eligible) if eligible else None
        # Four parents must contribute actual nontrivial comparisons, not just be present.
        enough = len(groups) >= 4 and bool(eligible)
        answer[arm] = dict(status=('PASS' if accuracy >= .70 else 'FAIL') if enough else 'UNDERPOWERED',
            correct=sum(r['correct'] for r in eligible), nontrivial=len(eligible),
            nontrivial_parents=len(groups), accuracy=accuracy,
            parent_accuracy={g:sum(r['correct'] for r in eligible if r['parent']==g) /
                              sum(r['parent']==g for r in eligible) for g in groups},
            comparisons=comparisons,
            sensitivity={str(t):dict(n=sum(abs(r['return_difference'])>=t for r in comparisons),
                correct=sum(r['correct'] and abs(r['return_difference'])>=t for r in comparisons))
                for t in (0., 1e-6, 1e-5, 1e-4, 1e-3)})
    return answer


def study(e, m, c, out):
    import numpy as np
    import torch
    from native_env import install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from research.interaction_fb.env import Environment
    from research.interaction_fb.approved_learning import ApprovedLearner
    from research.interaction_fb.approved_journal import load_state
    from research.interaction_fb.checkpoint import restore, sha
    from research.interaction_fb.journal import dump
    from research.interaction_fb.schema import to_device
    from research.interaction_fb.qualification_v11 import install_gate
    from research.interaction_fb.gate_v11 import metrics
    from diagnostic_frozen import portable
    from diagnostic_fb_resume import tree_digest
    from dual_track_runtime import TrainingBranch
    from dual_track_task import global_rng
    from rolling_snapshot import state_of, put
    from f1_mjdata_contract import install_restore_guard, canonical, metadata, exact

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    install(); speed(e); (out/'backend-prefix').mkdir(); backend(e,m,out/'backend-prefix')
    install_restore_guard()
    lock_path = ROOT/'research_notes/r2_ei_q_ranking_lock.json'
    lock = json.loads(lock_path.read_text())
    init = json.loads((ROOT/'research_notes/r2_ei_v11_entry_lock.json').read_text())
    data_path = ROOT/'research_notes/r2_splits/train.json'
    assert sha(data_path) == init['train_file_sha256'] == lock['train_sha256']
    specs = [dict(sp,split='train') for p in json.loads(data_path.read_text())['parents'] for sp in p['specs']]
    env = Environment(e,m,c,specs); env.reset(0)
    obs = env.proxy.enrich(env.task.extended_observation())
    agents = {arm:ApprovedLearner(m,obs,e.standing,torch.tensor(c['q_PD_scale'],device='cuda'),arm,init)
              for arm in ('E','I')}
    checkpoints = {arm:STAGE/'runs'/lock['warmup_runs'][arm]/'checkpoints/round-14' for arm in agents}
    for arm,path in checkpoints.items():
        assert sha(path/'manifest.json') == lock['warmup_manifest_sha256'][arm]
    states = {arm:load_state(path) for arm,path in checkpoints.items()}
    for arm,agent in agents.items():
        assert states[arm]['cursor'] == 3500 and states[arm]['agent']['actor_steps'] == 0
        agent.load(states[arm]['agent'])
    assert tree_digest(agents['E'].tail.state_dict()) == tree_digest(agents['I'].tail.state_dict())
    for key in ('u','goal_delta'):
        assert tree_digest(agents['E'].nets[key].state_dict()) == tree_digest(agents['I'].nets[key].state_dict())
    frozen = {arm:tree_digest(a.state()) for arm,a in agents.items()}
    results = []
    for round_id in lock['checkpoint_rounds']:
        dest = out/f'round-{round_id}'; dest.mkdir()
        os.environ['BFM_RECOVERY_OUTPUT'] = str(dest)
        incoming = checkpoints['E'].parent/f'round-{round_id}'
        # Restore full archived sampled physical prefix and optimizer state before
        # placing the fixed 3500-step models on that exact real visited state.
        restore(incoming,agents['E'],env)
        agents['E'].load(states['E']['agent'])
        env.proxy.agent=agents['E']; env.proxy.stochastic=False
        for agent in agents.values():assert agent.actor_steps == 0
        start_time=float(e.tether.time)
        assert start_time+lock['window_s'] <= env.task.spec['end']+1e-8
        env.proxy.slot=0; env.proxy.start=env.task.last_command.clone(); env.proxy.target=env.proxy.start.clone()
        current=env.proxy.enrich(env.task.extended_observation())
        with torch.no_grad():
            ae,z,zi=agents['E'].action(current)
            ai,iz,_=agents['I'].action(current)
        assert torch.equal(ae,ai) and torch.equal(z,iz)
        source_state=portable(env.task); source_meta=metadata(e.sim.model,e.sim.data)
        source_rng=tree_digest(global_rng()); source_digest=tree_digest(canonical(source_state,source_meta))
        generator=torch.Generator(device='cpu').manual_seed(lock['candidate_seed']+round_id)
        eps=[(.01*torch.randn(ae.shape,generator=generator)).clamp(-.06,.06).to(ae.device) for _ in range(2)]
        candidates=[('mean',ae),('draw0_plus',(ae+eps[0]).clamp(-1,1)),
                    ('draw0_minus',(ae-eps[0]).clamp(-1,1)),
                    ('draw1_plus',(ae+eps[1]).clamp(-1,1)),('draw1_minus',(ae-eps[1]).clamp(-1,1))]
        snapshot=TrainingBranch(env.task)
        extra=state_of(env,exclude=('task','proxy')); proxy=state_of(env.proxy)
        dump(dest/'full-start.ptz',dict(physical=source_state,metadata=source_meta,rng=global_rng(),
            proxy=proxy,spec=env.task.spec,checkpoint_manifest_sha=sha(incoming/'manifest.json')))
        branches=[]
        for name,candidate in candidates:
            snapshot.restore(); put(env,extra); put(env.proxy,proxy)
            assert exact(canonical(portable(env.task),metadata(e.sim.model,e.sim.data)),canonical(source_state,source_meta))
            assert tree_digest(global_rng()) == source_rng
            env.proxy.agent=agents['E']; env.proxy.stochastic=False
            install_gate(env,env.task.physics)
            start_rows=len(env.rows); start_ph=len(env.task.physics); start_ctrl=len(env.task.controls)
            scores={}
            with torch.no_grad():
                for arm,agent in agents.items():
                    aq,zq,iq=agent.action(current)
                    assert torch.equal(aq,ae) and torch.equal(zq,z)
                    qs=[float(agent.nets[k](agent.inp(current,candidate,z,iq))) for k in ('q1','q2')]
                    assert all(np.isfinite(q) and q <= 0 for q in qs)
                    scores[arm]=dict(q1=qs[0],q2=qs[1],q_min=min(qs))
            original=agents['E'].action; calls=[0]
            def first_only(ob,target=False,sample=False,z_override=None):
                assert not target and not sample
                a,zz,ii=original(ob,target,sample,z_override)
                with torch.no_grad():
                    aa,zz2,_=agents['I'].action(ob,target,sample,z_override)
                assert torch.equal(a,aa) and torch.equal(zz,zz2),'E/I physical continuation differs'
                if calls[0] == 0:
                    assert torch.equal(zz,z)
                    a=candidate
                calls[0]+=1
                return a,zz,ii
            agents['E'].action=first_only
            failure=None
            try:
                with torch.no_grad():
                    for _ in range(round(lock['window_s']/.1)):
                        env.collect_call()
            except RuntimeError as exc:
                if not str(exc).startswith('HARD_PHYSICS_BOUNDARY:'):raise
                failure=str(exc)
            finally:agents['E'].action=original
            rows=env.rows[start_rows:]; ph=env.task.physics[start_ph:]; controls=env.task.controls[start_ctrl:]
            assert rows and len(rows)==calls[0]==len(controls)
            assert torch.equal(rows[0]['action'],candidate.cpu()) and torch.equal(rows[0]['z'],z.cpu())
            assert torch.equal(torch.tensor(controls[0]['action']),candidate[0].cpu())
            assert torch.equal(torch.tensor(controls[0]['q_PD']),env.task.qof(candidate)[0].cpu())
            rewards=[float(row['aux_rewards']['contact_physical']) for row in rows]
            discounts=[.99**(float(row['duration'])/.1) for row in rows]
            w=1.; total=0.
            for reward,d in zip(rewards,discounts):total+=w*reward;w*=d
            # Tail-corrected value is reported separately; formal 70% gate remains
            # the preregistered actual finite-window return ordering.
            with torch.no_grad():
                no=to_device(rows[-1]['next']['observation'])
                for arm,agent in agents.items():
                    na,nz,ni=agent.action(no,target=True)
                    tq=min(float(agent.targets[k](agent.inp(no,na,nz,ni))) for k in ('q1','q2'))
                    scores[arm].update(target_tail_q=tq,bootstrapped_window_return=total+w*tq)
            trace=dict(rows=rows,physics=ph,controls=controls,candidate=candidate.cpu(),
                actual_previous_z=current['_previous_z'].cpu(),first_interpolated_z=z.cpu(),
                shared_mean_followup=True,initial_effective_state_digest=source_digest)
            archive=dump(dest/(name+'.ptz'),trace)
            active=[p for p in ph if p['active']]
            rec=dict(id=name,scores=scores,discounted_return=total,undiscounted_return=sum(rewards),
                quality=metrics(ph),active_error_rms_m=float(np.sqrt(np.mean([p['error']**2 for p in active]))) if active else None,
                active_force_rms_n=float(np.sqrt(np.mean([np.dot(p['force'],p['force']) for p in active]))) if active else None,
                first_action_linf_from_mean=float((candidate-ae).abs().max()),
                first_action_noise_l2_sigma=float(((candidate-ae)/.01).norm()),
                first_action_duration_s=float(rows[0]['duration']),controls=len(rows),
                failure=failure,archive=archive)
            branches.append(rec)
            (dest/'branches.json').write_text(json.dumps(branches,indent=2))
            print('Q_RANK_BRANCH',round_id,name,total,scores,'failure',failure,flush=True)
            if failure:
                dump(dest/'failed-complete-state.ptz',dict(physical=portable(env.task),rng=global_rng()))
                return dict(status='hard_physics_stop',round=round_id,candidate=name,results=results,branches=branches,
                            actor_updates=0,qualification='FAIL')
        results.append(dict(round=round_id,time=start_time,parent_id=env.task.spec['parent_id'],
            family=env.task.spec['demand_family'],start_effective_state_digest=source_digest,
            prefix_restore_exact=True,branches=branches))
        (out/'state-results.json').write_text(json.dumps(results,indent=2))
    for arm,agent in agents.items():assert tree_digest(agent.state()) == frozen[arm]
    verdict=adjudicate(results,lock)
    answer=dict(status='q_ranking_complete',results=results,verdict=verdict,
        training_updates=0,actor_updates=0,sealed_executed=False,physical_predictions_reused_across_arms=True,
        reuse_basis='all candidate start/continuation actual E/I action and z asserted equal',lock_sha256=sha(lock_path))
    (out/'qualification.json').write_text(json.dumps(answer,indent=2));return answer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from research.interaction_fb import benchmark
    benchmark.study=study
    sys.argv=[__file__,'--output',str(args.output),'--arm','A']
    benchmark.main()


if __name__=='__main__':main()
