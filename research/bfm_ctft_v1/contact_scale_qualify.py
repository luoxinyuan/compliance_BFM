"""Prelocked S0 sampled-execution check and frozen train-only aux calibration.

No learner is updated and no row becomes formal replay. Original native actor
and PD execute all commands. A calibration scalar is published only if every
preregistered trajectory completes without an original hard physical stop.
"""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'research_notes'), str(Path(__file__).parent)]
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import m11_ipc


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def make_plan():
    rows = []
    for force in [False, True]:
        rows.append(dict(name='sampled_' + ('force' if force else 'none'),
            purpose='native_training_sampling_qualification', sampled=True,
            mode=1, apply_force=force, stiffness=80., direction_deg=0., speed=.12))
    # A numerical scale, not a new training curriculum or an effect comparison.
    # The 20 mean episodes have exactly 25% g0, 75% g1, and 20% unloaded g1.
    for i, (case, index) in enumerate(zip([1, 2, 1, 2, 1], [25, 75, 125, 175, 225])):
        rows.append(dict(name=f'cal_native_{i}', purpose='fixed_scale', sampled=False,
            mode=0, source_case=case, source_control=index, controls=200))
    for k in [40., 80.]:
        for speed in [.08, .12, .16]:
            for angle in [-15., 15.]:
                rows.append(dict(name=f'cal_contact_k{k:g}_v{speed:g}_d{angle:g}',
                    purpose='fixed_scale', sampled=False, mode=1, apply_force=True,
                    stiffness=k, direction_deg=angle, speed=speed))
    for speed in [.08, .12, .16]:
        rows.append(dict(name=f'cal_unloaded_{speed:g}', purpose='fixed_scale',
            sampled=False, mode=1, apply_force=False, stiffness=80., direction_deg=0., speed=speed))
    return dict(version='CTFT_S0_FIXED_AUX_CALIBRATION_V1', cases=rows,
        seed=681117, native_weight_source='original_matching_training_config',
        formula='fixed_std=sqrt(max(mean(r_total^2)-mean(r_total)^2,1e-8)); output=r_total/fixed_std; no mean subtraction',
        pooling='all actual mean-calibration control rows; sampled qualification excluded; no old task stats',
        population='explicit train-only physical prefixes; two native walk ancestors and one standing prefix family, NOT20 independent groups',
        boundaries='all original mechanical/tilt/undesired contacts and causal >50ms no-support; fail closed',
        duration=dict(contact_distance_m=.6, accel_s=.5, hold_s=1., recovery_s=4.),
        qualification_sampling='original model.act(mean=False), matching actor_std, no extra exploration transform',
        no_training=True, no_dev_or_sealed=True, physics_tick_cap=80000, wall_cap_s=2400)


def measure(self):
    import numpy as np
    from contact_interface_probe import partner
    from humanoidverse.force_compliance.interaction import spring_force
    spec = self.ctft_partner_spec
    p, v = partner(spec, self.time)
    a = np.deg2rad(spec['direction_deg'])
    R = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
    p, v = R @ p, R @ v
    self.anchor = self.ctft_partner_origin + p
    self.partner_velocity = v
    point, pv = self.point_state()
    self.active = bool(spec['apply_force'] and spec['onset_s'] <= self.time < spec['release_s'])
    self.ctft_raw_force = spec['stiffness'] * (self.anchor-point) + 8. * (v-pv) if self.active else np.zeros(3)
    self.force = spring_force(self.anchor, point, v, pv, spec['stiffness'], 8., 10.) if self.active else np.zeros(3)
    expected = self.ctft_raw_force * min(1., 10. / max(float(np.linalg.norm(self.ctft_raw_force)), 1e-12))
    assert np.array_equal(self.force, expected)
    return point, self.force.copy()


def restore_source(e, saved):
    import numpy as np
    import torch
    import mujoco
    from rolling_snapshot import put
    from f1_mjdata_contract import metadata, inactive, exact
    for obj, key in [(e, 'env'), (e.base, 'base'), (e.sim, 'sim'), (e.wrapped, 'wrapped'), (e.tether, 'tether')]:
        put(obj, saved[key])
    e.base.history_handler.history = copy.deepcopy(saved['bfm_history'])
    e.rng.bit_generator.state = copy.deepcopy(saved['env_generator_state'])
    e.contact_rng.bit_generator.state = copy.deepcopy(saved['contact_generator_state'])
    m, d = e.sim.model, e.sim.data
    mask = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(m, d, saved['integration'], mask)
    mujoco.mj_forward(m, d)
    ignored = inactive(metadata(m, d))
    for k, v in saved['mjdata_arrays'].items():
        if k in ignored:
            continue
        dest = getattr(d, k)
        if dest.shape == v.shape and dest.flags.writeable:
            dest[...] = v
    mujoco.mj_setState(m, d, saved['integration'], mask)
    assert all(exact(v, getattr(d, k)) for k, v in saved['mjdata_arrays'].items() if k not in ignored)
    random.setstate(saved['python_rng'])
    np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    assert exact(e.obs, saved['env']['obs']) and exact(e.base.history_handler.history, saved['bfm_history'])


def study(e, model, contract, out):
    import numpy as np
    import torch
    import mujoco
    from m8_stage import reset_common, advance
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from locomotion_rollout import UnsafePhysics
    from quality import OnlineSupport
    from reward import ForceEMA, contact_cost, unit_checks
    plan = json.loads((out/'protocol.json').read_text())
    source_cfg = json.loads(Path('/mnt/data-oss/users/yi.gao/compliance_BFM/pretrained/official/new_model_for_training_code_inference/config.json').read_text())['agent']
    scales = source_cfg['aux_rewards_scaling']
    assert model.cfg.actor_std == source_cfg['model']['actor_std']
    write(out/'unit-fixtures.json', unit_checks())
    speed(e)
    (out/'backend-prefix').mkdir()
    backend(e, model, out/'backend-prefix')
    m, d = e.sim.model, e.sim.data
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
    native_measure = e.tether.measure.__func__
    native_hook = e.sim.simulate_at_each_physics_step
    guard = OnlineSupport(float(m.body_mass.sum()*np.linalg.norm(m.opt.gravity)))

    def guarded():
        native_hook()
        s = guard.update(e.contact_sample['foot_normal_force_n'])
        e.trace[-1]['ctft_online_support'] = s
        if s['stop']:
            raise UnsafePhysics(dict(e.trace[-1], ctft_stop_reason='continuous_no_effective_support_gt_50ms'))

    results, calibration = [], []
    for case_index, case in enumerate(plan['cases']):
        e.qualification_active = False
        e.tether.measure = types.MethodType(native_measure, e.tether)
        e.cfg.max_force = 5.
        reset_common(e, model, model.act)
        t = e.tether
        source = None
        if case['mode'] == 0:
            path = STAGE/'runs/s0-native-sources-03'/f"source-{case['source_case']:02d}.pkl.gz"
            with gzip.open(path, 'rb') as f:
                source = pickle.load(f)
            assert source['boundary'] is None
            restore_source(e, source['states'][case['source_control']])
            e.cfg.enabled = t.cfg.enabled = False
            guard = OnlineSupport(**source['states'][case['source_control']]['online_support'])
            controls = case['controls']
        else:
            onset = float(t.time)
            release = onset + .6/case['speed'] + .5 + 1.
            spec = dict(name=case['name'], plateau_speed_mps=case['speed'], distance_m=.6,
                accel_s=.5, onset_s=onset, release_s=release, end_s=release+4.,
                stiffness=case['stiffness'], direction_deg=case['direction_deg'], apply_force=case['apply_force'])
            t.ctft_partner_spec = spec
            t.ctft_partner_origin = t.point_state()[0].copy()
            for cfg in [e.cfg, t.cfg]:
                cfg.enabled = case['apply_force']
                cfg.stiffness = case['stiffness']
                cfg.damping = 8.
                cfg.max_force = 10.
                cfg.activation_time = onset
                cfg.release_time = release
            t.measure = types.MethodType(measure, t)
            t.measure()
            controls = round((release+4.-onset)/.02)
            guard = OnlineSupport(float(m.body_mass.sum()*np.linalg.norm(m.opt.gravity)))
            for row in e.trace:
                guard.update(row['foot_normal_force_n'])
        e.sim.simulate_at_each_physics_step = guarded
        e.qualification_active = True
        torch.manual_seed(plan['seed']+case_index)
        ema = ForceEMA()
        rows, boundary = [], None
        first_trace = len(e.trace)
        start = time.monotonic()
        try:
            for i in range(controls):
                assert time.monotonic() < float(os.environ['CCL_DEADLINE_MONOTONIC'])
                z = e.standing if source is None else source['controls'][case['source_control']+i]['z'].to(e.standing.device)
                action = model.act(e.obs, z, mean=not case['sampled'])
                assert torch.isfinite(action).all()
                e.previous_z = z.clone()
                done = advance(e, action)
                filtered = ema.update(t.force)
                point = d.xpos[t.body].copy()
                jac = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jac, None, point, t.body)
                cost = contact_cost(filtered_force_n=filtered, applied_force_n=t.force,
                    wrist_velocity_mps=jac@d.qvel, root_velocity_mps=d.qvel[:3], guidance_mode=case['mode'])
                raw = {k:float(v.detach().cpu().reshape(-1)[0]) for k,v in e.base.extras['aux_rewards'].items()}
                native = sum(raw[k]*v for k,v in scales.items())
                total = native + cost['contact_native_reward']
                assert np.isfinite(total)
                rows.append(dict(control=i, time_s=float(t.time), native_raw_components=raw,
                    weighted_native=native, contact_reward=cost, total_native=total,
                    action=action.detach().cpu().tolist(), z=z.detach().cpu().tolist(),
                    terminated=bool(e.failed), truncated=bool(done and not e.failed)))
                if done:
                    boundary = dict(kind='native_terminal_or_time_limit', control=i, time_s=float(t.time))
                    break
        except UnsafePhysics as exc:
            boundary = dict(kind='hard_physics', row=exc.row)
        finally:
            e.qualification_active = False
        evidence = dict(case=case, rows=rows, physics=copy.deepcopy(e.trace[first_trace:]), boundary=boundary)
        path = out/(case['name']+'.pkl.gz')
        with gzip.open(path, 'wb', compresslevel=1) as f:
            pickle.dump(evidence, f, protocol=5)
        record = dict(name=case['name'], complete=boundary is None, boundary=boundary,
            rows=len(rows), physics_rows=len(evidence['physics']), wall_s=time.monotonic()-start,
            evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), mode=case['mode'],
            sampled=case['sampled'], purpose=case['purpose'])
        results.append(record)
        write(out/'progress.json', results)
        print('CTFT_SCALE_CASE', case['name'], record['complete'], len(rows), flush=True)
        if boundary:
            result = dict(status='S0_SAMPLED_OR_CALIBRATION_PHYSICS_BLOCKED',
                cases=results, failed_case=case['name'], scale_locked=False,
                formal_training=False, updates=0)
            write(out/'scale-result.json', result)
            return result
        if case['purpose'] == 'fixed_scale':
            calibration.extend(r['total_native'] for r in rows)
        del evidence, source
    a = np.asarray(calibration, dtype=np.float64)
    var = max(float(np.mean(a*a)-a.mean()**2), 1e-8)
    result = dict(status='S0_SAMPLING_AND_FIXED_SCALE_QUALIFIED',
        total_reward_scale=float(np.sqrt(var)), mean=float(a.mean()), variance=var,
        scalar_float32=float(np.float32(np.sqrt(var))), actual_calibration_rows=len(a),
        native_coefficients=scales, source_normalizer=source_cfg['model']['norm_aux_reward'],
        cases=results, mean_subtraction=False, formula=plan['formula'],
        formal_training=False, updates=0, scale_locked=True,
        applicability='same frozen scale for U/T; no old stats and no online adaptation')
    write(out/'scale-result.json', result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a=p.parse_args()
    if a.prepare_only:
        a.output.mkdir(parents=True, exist_ok=False)
        write(a.output/'protocol.json', make_plan())
        return
    assert (a.output/'protocol.json').exists()
    assert json.loads((a.output/'protocol.json').read_text()) == make_plan()
    assert not (a.output/'launch.json').exists()
    write(a.output/'launch.json', dict(pid=os.getpid(), started_unix=time.time(),
        runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        protocol_sha256=hashlib.sha256((a.output/'protocol.json').read_bytes()).hexdigest()))
    os.environ.update(BFM_RECOVERY_OUTPUT=str(a.output), CCL_DEADLINE_MONOTONIC=str(time.monotonic()+2400))
    import mujoco
    original=mujoco.mj_step
    counts=dict(actual=0, reserved=0)
    def counted(*args, **kw):
        n=int(kw.get('nstep',1))
        counts['reserved'] += n
        assert counts['reserved'] <= 80000
        result=original(*args, **kw)
        counts['actual'] += n
        return result
    mujoco.mj_step=counted
    import m10_stage, m10_physics
    m10_stage.run_study=study
    m10_physics.OUT=a.output
    start=time.monotonic()
    try:
        m10_physics.main()
    except BaseException as exc:
        write(a.output/'execution-failure.json', dict(type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        write(a.output/'cost.json', dict(wall_s=time.monotonic()-start, physics=counts,
            agent_updates=0, formal_training_transitions=0))


if __name__=='__main__':
    main()
