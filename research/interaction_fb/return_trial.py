"""Approved Return pilot: complete real-prefix replay and isolated adapter updates.

This entry never calls Agent.update. All physics retains the original Task.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'research_notes')]
P = ROOT.parents[2]
S = ROOT.parent
OLD = P / 'experiments/interaction-pilot-v3-20261112/runs'


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_case(env, agent, case, out):
    import torch
    from research.interaction_fb.journal import load_state
    from research.interaction_fb.checkpoint import restore
    from diagnostic_fb_resume import tree_digest
    from dual_track_task import restore_global_rng
    from f1_mjdata_contract import exact, canonical, metadata
    from diagnostic_frozen import portable
    arm = 'A' if case['role'] == 'none_dev' else 'B'
    path = OLD / f'pilot-{arm}-v3/checkpoints' / f"round-{case['archived_round']}"
    saved = load_state(path)
    assert saved['task_specs'][saved['spec_id']]['parent_id'] == case['parent_id']
    env.specs = saved['task_specs']
    # Full archive restore first, including the original canonical arrays/history gate.
    restore(path, agent, env)
    completed_restore = dict(core_sha256=sha(path / 'core.ptz'),
                             manifest_sha256=sha(path / 'manifest.json'),
                             saved_time=float(env.task.e.tether.time),
                             original_exact_restore_passed=True)
    target = case['time']
    if abs(float(env.task.e.tether.time) - target) > 1e-8:
        assert target < float(env.task.e.tether.time)
        env.reset(saved['spec_id'])
        assert env.task.prefix_data['audit']['sha256'] == saved['source_sha']
        p, t = env.proxy, env.task
        count = round((target - float(t.e.tether.time)) / .1)
        assert count >= 0 and abs(float(t.e.tether.time) + .1 * count - target) < 1e-7
        assert 5 * count <= len(saved['actions'])
        p.replay_actions = iter(saved['actions'][:5 * count]); p.actions = []
        for raw in saved['commands'][:count]:
            p.slot = 0; p.start = t.last_command.clone(); p.target = raw.cuda() * 16
            p.take_step(raw.cuda(), 5)
            assert not t.done and not t.boundary
        try:
            next(p.replay_actions)
            raise AssertionError('Unplayed actual prefix action')
        except StopIteration:
            pass
        p.replay_actions = None
        # The archived current episode is the last monotone-time block.
        all_physics = saved['physical_stream']; begin = 0
        for i in range(1, len(all_physics)):
            if all_physics[i]['time'] < all_physics[i-1]['time']:
                begin = i
        expected = [r for r in all_physics[begin:] if r['time'] <= target + 1e-8]
        assert len(t.physics) == len(expected)
        for index, (a, b) in enumerate(zip(t.physics, expected)):
            assert exact(a, b), ('Actual prefix physics differs', index, a['time'])
        assert len(t.controls) == 5 * count
        assert all(torch.equal(a.cpu(), b) for a, b in zip(p.actions, saved['actions'][:5*count]))
        p.slot = 0; p.start = t.last_command.clone(); p.target = p.start.clone()
        env.commands = saved['commands'][:count]
        completed_restore.update(truncated_actual_action_prefix_exact=True,
                                 actual_prefix_controls=len(t.controls), actual_prefix_ticks=len(t.physics),
                                 actual_prefix_sha256=tree_digest(dict(physics=t.physics, controls=t.controls,
                                                                      actions=p.actions, commands=env.commands)))
        # Saved RNG belongs to the later archive endpoint; use it only as an
        # explicit matched evaluation stream, never claim historical time-target RNG.
        restore_global_rng(saved['rng'])
        env.task.rng.bit_generator.state = copy.deepcopy(saved['task_rng'])
        completed_restore['evaluation_rng_origin'] = 'explicit archived endpoint RNG, matched across branches; not historical intermediate RNG'
    else:
        completed_restore['evaluation_rng_origin'] = 'exact archived endpoint RNG'
    assert abs(float(env.task.e.tether.time) - target) < 1e-7
    assert target + 3 <= env.task.spec['end'] + 1e-8
    state = portable(env.task)
    completed_restore.update(state_time=float(env.task.e.tether.time),
                              state_digest=tree_digest(canonical(state, metadata(env.task.e.sim.model, env.task.e.sim.data))),
                              source_sha256=saved['source_sha'],
                              exact_previous_z_digest=tree_digest(env.task.last_command))
    write(out / 'prefix-qualification.json', completed_restore)
    return completed_restore


def metrics(rows, controls):
    import math
    import numpy as np
    assert rows
    active = [r for r in rows if r['active']]
    slip = np.asarray([float(np.max(r['stage_loaded_slip_mps'])) for r in rows])
    mat = np.asarray(rows[-1]['pelvis_matrix']).reshape(3, 3)
    result = dict(ticks=len(rows), duration_s=len(rows)*.005,
                  joint_excess_max_rad=max(r['joint_limit_excess_max_rad'] for r in rows),
                  no_support_ticks=sum(not any(r['support']) for r in rows),
                  slip_peak=float(slip.max()), slip_p95=float(np.percentile(slip, 95)),
                  yaw_final_deg=math.degrees(math.atan2(mat[1,0], mat[0,0])),
                  min_height=min(r['height'] for r in rows),
                  tilt_max_deg=max(r['tilt_deg'] for r in rows),
                  root_displacement_m=float(np.linalg.norm(np.asarray(rows[-1]['root'])[:2]-np.asarray(rows[0]['root'])[:2])),
                  final_root_speed_mps=float(np.linalg.norm(rows[-1]['qvel'][:3])),
                  force_peak_n=max(float(np.linalg.norm(r['force'])) for r in rows),
                  force_rms_n=float(np.sqrt(np.mean([np.dot(r['force'],r['force']) for r in rows]))),
                  active_ticks=len(active),
                  undiscounted_native_return=float(sum(c['reward_integral'] for c in controls)))
    if active:
        result.update(error_rms_m=float(np.sqrt(np.mean([r['error']**2 for r in active]))),
                      active_force_rms_n=float(np.sqrt(np.mean([np.dot(r['force'],r['force']) for r in active]))),
                      wrist_progress_m=float(active[-1]['wrist'][0]-active[0]['wrist'][0]),
                      partner_progress_m=float(active[-1]['anchor'][0]-active[0]['anchor'][0]))
    else:
        result.update(error_rms_m=None, active_force_rms_n=None,
                      wrist_progress_m=None, partner_progress_m=None)
    return result


def branch(env, agent, out, stochastic, mode):
    import torch
    from diagnostic_frozen import portable
    from diagnostic_fb_resume import tree_digest
    from dual_track_task import global_rng
    from research.interaction_fb.schema import deploy, robot
    from research.interaction_fb.env import Proxy
    initial = portable(env.task); rng = global_rng()
    frozen = tree_digest(agent.state()); original = agent.action; logs = []
    def observed(obs, target=False, sample=False, z_override=None):
        if mode == 'off':
            _, z, zi = original(obs, target, False, z_override)
            base = agent.base._actor(agent.base._normalize(robot(obs)), z, agent.base.cfg.actor_std).mean
            action = base.clamp(-1,1)
            if sample:
                action = (action+(agent.exploration_std*torch.randn_like(action)).clamp(-agent.exploration_clip,agent.exploration_clip)).clamp(-1,1)
            agent.last_action_audit = None
            audit = dict(base=base.detach().cpu(), executed=action.detach().cpu())
        else:
            action, z, zi = original(obs, target, sample, z_override)
            audit = {k:v.detach().cpu().clone() for k,v in agent.last_action_audit.items()}
        logs.append(dict(time=float(env.task.e.tether.time),
                         observation={k:v.detach().cpu().clone() for k,v in obs.items()},
                         z=z.detach().cpu().clone(), audit=audit))
        return action, z, zi
    agent.action = observed; env.proxy.agent = agent; env.proxy.stochastic = stochastic
    starts = [len(env.task.physics), len(env.task.controls), len(env.rows), len(env.commands)]
    status = 'completed_3s'; error = None
    try:
        with torch.no_grad():
            for _ in range(30):
                env.collect_call()
    except RuntimeError as exc:
        if not str(exc).startswith('HARD_PHYSICS_BOUNDARY:'):
            raise
        status = 'hard_physics_boundary'; error = str(exc)
    finally:
        agent.action = original
    assert tree_digest(agent.state()) == frozen, 'Frozen branch altered model/Adam/targets'
    physics=env.task.physics[starts[0]:]; controls=env.task.controls[starts[1]:]
    rows=env.rows[starts[2]:]
    assert len(rows)==len(controls)==len(logs)
    for row, control, log in zip(rows,controls,logs):
        a=log['audit']['executed']
        assert torch.equal(a,row['action'])
        assert torch.equal(a[0],torch.tensor(control['action']))
        assert torch.equal(env.task.qof(a.cuda())[0].cpu(),torch.tensor(control['q_PD']))
    trace=dict(initial=initial,rng=rng,actions=logs,rows=rows,controls=controls,physics=physics,
               commands=env.commands[starts[3]:],task_spec=copy.deepcopy(env.task.spec),
               mode=mode,stochastic=stochastic)
    torch.save(trace,out/'trace.pt')
    result=dict(status=status,error=error,model_digest=frozen,rng_digest=tree_digest(rng),
                trace_sha256=sha(out/'trace.pt'),exact_executed_replay_qpd=True,
                mode=mode,sample=stochastic,metrics=metrics(physics,controls))
    write(out/'summary.json',result)
    return result


def study(e,m,c,out):
    import torch,numpy as np
    from native_env import install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from research.interaction_fb.env import Environment
    from research.interaction_fb.agent import Agent
    from research.interaction_fb.journal import load_state
    from f1_mjdata_contract import install_restore_guard,canonical,metadata,exact
    from diagnostic_frozen import portable
    from rolling_snapshot import BranchState,state_of,put
    from dual_track_task import global_rng,restore_global_rng
    from diagnostic_fb_resume import tree_digest
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    install();speed(e);(out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix')
    install_restore_guard()
    cases=json.loads((ROOT/'research_notes/adapter_return_pilot/approved-preparation-lock.json').read_text())['parents']
    case=cases[ARGS.case_index]
    # Original source checkpoint arm must match to qualify its complete restore.
    arm='A' if case['role']=='none_dev' else 'B'
    env=Environment(e,m,c);env.reset(0)
    obs=env.proxy.enrich(env.task.extended_observation())
    agent=Agent(m,obs,e.standing,torch.tensor(c['q_PD_scale'],device='cuda'),arm)
    prefix=restore_case(env,agent,case,out)
    # Substitute B19 only after the real archived prefix is fully reconstructed.
    if arm!='B':
        obs=env.proxy.enrich(env.task.extended_observation())
        agent=Agent(m,obs,e.standing,torch.tensor(c['q_PD_scale'],device='cuda'),'B')
    incoming=load_state(OLD/'pilot-B-v3/checkpoints/round-19')['agent']
    rng=global_rng();agent.load(incoming);restore_global_rng(rng);env.proxy.agent=agent
    env.rows=[];env.next_cursor=0
    snap=BranchState(env.task);proxy=state_of(env.proxy);frozen_rng=global_rng()
    initial=canonical(portable(env.task),metadata(e.sim.model,e.sim.data))
    lists=dict(rows=list(env.rows),commands=list(env.commands),physical_stream=list(env.physical_stream),
               episodes=list(env.episodes),next_cursor=env.next_cursor)
    def reset():
        snap.restore();put(env.proxy,proxy);restore_global_rng(frozen_rng)
        for k,v in lists.items():setattr(env,k,list(v) if isinstance(v,list) else v)
        assert exact(canonical(portable(env.task),metadata(e.sim.model,e.sim.data)),initial),'Full state branch mismatch'
        env.proxy.agent=agent
    results=[]
    for stochastic in (False,True):
        reset();dest=out/('off-sample' if stochastic else 'off-mean');dest.mkdir()
        r=branch(env,agent,dest,stochastic,'off');results.append(r)
        write(out/'branches.json',results)
        print('RETURN_REFERENCE',case['parent_id'][:12],stochastic,r['status'],r['metrics'],flush=True)
        if r['status']!='completed_3s':break
    complete=len(results)==2 and all(r['status']=='completed_3s' and r['metrics']['ticks']==600 and
        r['metrics']['joint_excess_max_rad']<=.02 and r['metrics']['no_support_ticks']==0 and
        r['metrics']['min_height']>=.45 and r['metrics']['tilt_max_deg']<=30 for r in results)
    result=dict(status='reference_qualified' if complete else 'reference_failed',case=case,
                prefix=prefix,branches=results,adapter_updates=0,
                qualification='Safe same-z recovery reference only; not absolute slip/yaw success')
    write(out/'reference-gate.json',result)
    return result


def main():
    global ARGS
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case-index',type=int,required=True)
    ARGS=parser.parse_args()
    from research.interaction_fb import benchmark
    benchmark.study=study
    sys.argv=[str(__file__),'--output',str(ARGS.output),'--arm','B']
    benchmark.main()


if __name__=='__main__':main()
