"""Fixed-state CUDA geometry and full-prefix physical branches; no PPO training."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import m11_ipc

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT.parent


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def read_episode(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def study(env, model, contract, output):
    import numpy as np
    import torch
    from clean_policy import Expert, direct_command
    from locomotion_rollout import spherical
    from m11_runtime_speed import install as speed
    from p4000_env import Task, install
    from rolling_backend import install as backend
    from rolling_snapshot import BranchState

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    assert contract['model_sha256'] == '33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361'
    install()
    write(output / 'runtime-speed.json', speed(env))
    write(output / 'backend.json', backend(
        env, model, STAGE.parent / 'native-transfer-20261107/compatibility-prefix'))
    plan = json.loads((ROOT / 'research_notes/p4000_protocol.json').read_text())
    paths = {'initial': Path(plan['initial']['path'])}
    for mode in ['independent', 'joint']:
        folder = STAGE / f'runs/single5201-{mode}-32'
        for batch in [8, 16, 24, 32]:
            paths[f'{mode}_{batch}'] = folder / f'candidate-{batch:04d}.pt'
        paths[f'{mode}_last'] = folder / 'last.pt'
    nets, identities = {}, {}
    for name, path in paths.items():
        ck = torch.load(path, map_location='cuda', weights_only=True)
        net = Expert(1088, 'direct').cuda().eval().requires_grad_(False)
        net.load_state_dict(ck['state_dict'])
        nets[name] = net
        identities[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                iteration=ck.get('iteration'), runtime=ck.get('runtime'))
    assert identities['initial']['sha256'] == plan['initial']['sha256']
    write(output / 'models.json', identities)

    # Cohorts are fixed by provenance and stride, not by outcome or label fit.
    cohorts = {}
    for label, folder, version in [
        ('initial_visited', STAGE / 'runs/seed5201-32', 0),
        ('independent_visited', STAGE / 'runs/single5201-independent-32', 23),
        ('joint_visited', STAGE / 'runs/single5201-joint-32', 23),
    ]:
        obs = []
        for worker in range(3):
            with np.load(folder / f'worker-{worker}/rollout-{version:04d}.npz') as data:
                obs.append(data['obs'][::4].copy())
        cohorts[label] = torch.as_tensor(np.concatenate(obs), device='cuda')

    geometry = []
    for label, obs in cohorts.items():
        means = {name: net.distribution(obs).mean.double() for name, net in nets.items()}
        mu0 = means['initial']
        unit0 = mu0 / mu0.norm(dim=-1, keepdim=True)
        for name, mu in means.items():
            delta = mu - mu0
            radial = (delta * unit0).sum(-1, keepdim=True) * unit0
            tangent = delta - radial
            unit = mu / mu.norm(dim=-1, keepdim=True)
            cosine = (unit * unit0).sum(-1).clamp(-1, 1)
            sine = (unit - cosine[:, None] * unit0).norm(dim=-1)
            angle = torch.rad2deg(torch.atan2(sine, cosine))
            geometry.append(dict(
                cohort=label, model=name, states=len(obs),
                cumulative_z_angle_deg_mean=float(angle.mean()),
                cumulative_z_angle_deg_max=float(angle.max()),
                raw_delta_l2_RMS=float(delta.square().sum(-1).mean().sqrt()),
                radial_fraction_of_delta_energy=float(radial.square().sum() / delta.square().sum().clamp_min(1e-30)),
                tangent_delta_l2_RMS=float(tangent.square().sum(-1).mean().sqrt()),
                raw_norm_mean=float(mu.norm(dim=-1).mean()),
                fixed_covariance_joint_KL_from_initial=float(.5 * delta.square().sum(-1).mean() / .008**2),
                scope='same recorded high-level inputs; not a physical rollout or per-batch causal attribution'))
    write(output / 'fixed-state-geometry.json', geometry)

    source = STAGE / 'runs/single5201-independent-32/dev-0024'
    episodes = {case: read_episode(source / f'{case}__mean.json.gz')
                for case in ['normal72', 'stop66']}
    task = Task(env, model, contract, 5277)
    action_rows = []
    # Same true previous z and all five actual BFM observations for every model.
    for case, episode in episodes.items():
        controls = episode['control']
        decisions = episode['decisions']
        for index in np.linspace(2, len(decisions)-3, 6).astype(int):
            decision = decisions[index]
            start_time = decision['time'] - .1
            indices = [i for i, c in enumerate(controls) if abs(c['time'] - start_time) < 1e-6]
            if not indices:
                continue
            first = indices[0]
            if first == 0 or first+5 > len(controls):
                continue
            previous = torch.tensor(controls[first-1]['executed_z'], device='cuda', dtype=torch.float32)
            inp = torch.tensor(decision['obs'], device='cuda', dtype=torch.float32)[None]
            raw_by_model = {name: net.distribution(inp).mean for name, net in nets.items()}
            raw_by_model['recorded_source_command'] = torch.tensor(decision['raw'], device='cuda', dtype=torch.float32)[None]
            q_by_model = {}
            for name, raw in raw_by_model.items():
                z = direct_command(raw, env.standing)
                qs = []
                for substep, c in enumerate(controls[first:first+5]):
                    bfm_obs = {key: torch.tensor(c['bfm_obs'][key], device='cuda', dtype=torch.float32)
                               for key in env.keys}
                    qs.append(task.qof(model.act(bfm_obs, spherical(previous, z, (substep+1)/5), mean=True)))
                q_by_model[name] = torch.cat(qs)
            recorded = torch.tensor([c['q_PD'] for c in controls[first:first+5]], device='cuda')
            replay_error = float(torch.rad2deg(q_by_model['recorded_source_command']-recorded).abs().max())
            assert replay_error < 1e-5, ('Saved BFM control replay mismatch', case, index, replay_error)
            for name in nets:
                difference = torch.rad2deg(q_by_model[name]-q_by_model['initial'])
                action_rows.append(dict(case=case, source_decision=int(index), start_time=start_time, model=name,
                    recorded_source_replay_max_deg=replay_error,
                    five_substep_qpd_RMS_deg=float(difference.square().mean().sqrt()),
                    five_substep_qpd_max_deg=float(difference.abs().max()),
                    scope='fixed actual five BFM observations and actual previous z, not five physical alternative steps'))
    assert action_rows
    write(output / 'fixed-state-five-substep-actions.json', action_rows)

    branches = []
    prefix_physics = branch_physics = 0
    # This is actual diagnostic execution in the fixed environment, not oracle
    # forecasting or PPO/on-policy data. No new training conditions are selected.
    requested = [('normal72', 6.2), ('stop66', float(episodes['stop66']['condition']['stop_time']) + .8)]
    modes = ['initial_mean', 'independent_mean', 'joint_mean', 'learned_delta3',
             'opposite_delta', 'independent_sample008', 'independent_sample016']
    for case, desired in requested:
        episode = episodes[case]
        choices = [(abs(d['time']-.1-desired), i, d['time']-.1) for i, d in enumerate(episode['decisions'])]
        _, boundary_index, at = min(choices)
        assert abs(at-desired) < .051
        task.reset(episode['condition'], record=True, label=f'prefix_{case}')
        prefix_errors = []
        for decision in episode['decisions'][:boundary_index]:
            now = float(env.tether.time)
            assert abs(now-(decision['time']-.1)) < 1e-6
            expected = torch.tensor(decision['obs'], device='cuda', dtype=torch.float32)[None]
            error = float((task.features()-expected).abs().max())
            assert error == 0., ('Saved causal prefix observation mismatch', case, now, error)
            _, _, done, stats = task.step(torch.tensor(decision['raw'], device='cuda', dtype=torch.float32)[None])
            prefix_physics += stats['physics_steps']
            assert not done
            prefix_errors.append(error)
        assert abs(float(env.tether.time)-at) < 1e-6
        expected_decision = episode['decisions'][boundary_index]
        assert torch.equal(task.features(), torch.tensor(expected_decision['obs'], device='cuda', dtype=torch.float32)[None])
        source_control = min(episode['control'], key=lambda c: abs(c['time']-at))
        qpos_error = float(np.max(np.abs(env.sim.data.qpos-np.asarray(source_control['pre_qpos']))))
        qvel_error = float(np.max(np.abs(env.sim.data.qvel-np.asarray(source_control['pre_qvel']))))
        assert qpos_error == qvel_error == 0.
        snapshot = BranchState(task)
        for mode in modes:
            snapshot.restore()
            assert BranchState(task).fingerprint == snapshot.fingerprint
            task.label = f'{case}_{at:.1f}_{mode}'
            begin = len(task.physics)
            control_begin = len(task.controls)
            previous_qpd = task.lastq[0].cpu().numpy().copy()
            value0 = float(nets['independent_32'].value(task.features()))
            generator = torch.Generator(device='cuda').manual_seed(5277)
            total = discounted = 0.
            discount = 1.
            commands = []
            for step in range(24):
                inp = task.features()
                mu0 = nets['initial'].distribution(inp).mean
                mui = nets['independent_32'].distribution(inp).mean
                if mode == 'initial_mean':
                    raw = mu0
                elif mode == 'joint_mean':
                    raw = nets['joint_32'].distribution(inp).mean
                elif mode == 'learned_delta3':
                    raw = mu0+3*(mui-mu0)
                elif mode == 'opposite_delta':
                    raw = mu0-(mui-mu0)
                elif mode.startswith('independent_sample'):
                    scale = .016 if mode.endswith('016') else .008
                    raw = mui+scale*torch.randn(mui.shape, device='cuda', generator=generator)
                else:
                    raw = mui
                commands.append(raw[0].cpu().tolist())
                _, reward, done, stats = task.step(raw)
                branch_physics += stats['physics_steps']
                total += reward
                discounted += discount*reward
                discount *= .99**(stats['control_steps']/5)
                if done:
                    break
            tail = 0. if task.done else float(nets['independent_32'].value(task.features()))
            physical = task.physics[begin:]
            active = [row for row in physical if row['active']]
            forces = np.array([row['force'] for row in active])
            errors = np.array([np.array(row['anchor'])-row['wrist'] for row in active])
            joint_positions = np.asarray([row['qpos'][7:] for row in physical])
            targets = np.vstack([previous_qpd] + [c['q_PD'] for c in task.controls[control_begin:]])
            feet = np.asarray([row['feet_xyz'] for row in physical])
            yaw = lambda entry, key: np.arctan2(np.asarray(entry[key])[1, 0], np.asarray(entry[key])[0, 0])
            yaw_delta = lambda key: float(np.degrees(np.arctan2(
                np.sin(yaw(physical[-1], key)-yaw(physical[0], key)),
                np.cos(yaw(physical[-1], key)-yaw(physical[0], key)))))
            row = dict(case=case, start_time=at, mode=mode, duration=len(physical)*.005,
                common_full_state_fingerprint=snapshot.fingerprint,
                prefix_observation_max_error=max(prefix_errors, default=0.),
                prefix_qpos_max_error=qpos_error, prefix_qvel_max_error=qvel_error,
                window_reward=total, discounted_window_reward=discounted,
                common_critic_start_value=value0, common_critic_tail_value=tail,
                truncated_bootstrapped_value=discounted+discount*tail,
                truncated_bootstrapped_advantage=discounted+discount*tail-value0,
                active_force_RMS=float(np.sqrt(np.square(forces).sum(-1).mean())) if active else 0.,
                active_error_cm=float(100*np.sqrt(np.square(errors).sum(-1).mean())) if active else None,
                active_force_xyz_RMS=np.sqrt(np.square(forces).mean(0)).tolist() if active else [0., 0., 0.],
                active_signed_error_xyz_cm=(100*errors.mean(0)).tolist() if active else [0., 0., 0.],
                cap_fraction=float(np.mean([np.linalg.norm(row['force']) >= 9.999 for row in active])) if active else 0.,
                root_delta_cm=(100*(np.array(physical[-1]['root'])-physical[0]['root'])).tolist(),
                torso_yaw_delta_deg=yaw_delta('torso_matrix'), pelvis_yaw_delta_deg=yaw_delta('pelvis_matrix'),
                foot_path_cm=(100*np.linalg.norm(np.diff(feet, axis=0), axis=-1).sum(0)).tolist(),
                qpd_20ms_max_step_deg=float(np.degrees(np.abs(np.diff(targets, axis=0))).max()),
                actual_joint_5ms_max_step_deg=float(np.degrees(np.abs(np.diff(joint_positions, axis=0))).max()),
                max_tilt_deg=max(r['tilt_deg'] for r in physical), min_height=min(r['height'] for r in physical),
                max_slip_mps=max(r['contact_tangent_speed_max_mps'] for r in physical),
                max_torque_saturation=max(r['torque_saturation'] for r in physical),
                boundary=task.boundary, physics_steps=len(physical), active_steps=len(active),
                scope='2.4s feedback diagnostic, common current critic tail is not ground truth; no training labels or PPO data')
            task.save(output)
            write(output / f'{task.label}-commands.json', commands)
            branches.append(row)
            write(output / 'physical-branches.json', branches)
            print('MEAN_DIAG_BRANCH', case, mode, total, row['active_error_cm'], row['active_force_RMS'], row['boundary'], flush=True)
        snapshot.restore()
    result = dict(status='completed', models=identities, fixed_input_states=sum(len(v) for v in cohorts.values()),
        branches=len(branches), counted_prefix_physics_steps=prefix_physics,
        counted_branch_physics_steps=branch_physics, backend_reset_prefixes_not_in_step_counters=True,
        new_RL_training_decisions=0, optimizer_updates=0,
        limits='Short diagnostic windows, not whole-episode improvements or a new deployable controller; no sealed reselection')
    write(output / 'diagnosis-result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='0', BFM_RECOVERY_OUTPUT=str(args.output),
        MUJOCO_GL='disable', CLEAN_KIND='direct', P4000_CANONICAL_ROW_ACTOR='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    os.chdir(args.output)
    import m10_physics
    import m10_stage
    m10_physics.OUT = args.output
    m10_stage.run_study = study
    m10_physics.main()


if __name__ == '__main__':
    main()
