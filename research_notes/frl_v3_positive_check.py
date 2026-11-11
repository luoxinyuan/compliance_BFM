"""Actual positive-event reward wiring qualification; zero learning updates."""
import copy, json, time
from pathlib import Path
import numpy as np
import torch
import mujoco
import frl_v3_train_stage as entry
import frl_v3_checks as checks
from frl_v2_high import build, write
from frl_v2_task import Task as Original
from dual_track_runtime import TrainingBranch
from frl_v3_risk import event
import frl_v3_task

def run(e, m, contract, out, args):
    started = time.monotonic()
    frl_v3_task.configure(.33)
    source = args.p1_output
    archive = torch.load(source/'state-23.pt', map_location='cpu', weights_only=False)
    assert archive['meta']['frozen_owner'] == 'BC500'
    net = build().eval().requires_grad_(False)
    task = frl_v3_task.Task(e, m, contract, 5801)
    torch.manual_seed(56017); np.random.seed(56017)
    task.reset(archive['spec'], record=True, label='actual-positive-path-prefix')
    for raw in archive['actual_raw_prefix']:
        Original.step(task, torch.as_tensor(raw, device='cuda').reshape(1, -1))
    x = torch.cat([task.student_features(), task.features()], -1)
    assert torch.equal(x.cpu(), archive['input'])
    assert torch.equal(task.last_command.cpu(), archive['previous_z'])
    mask = mujoco.mjtState.mjSTATE_INTEGRATION
    integration = np.empty(mujoco.mj_stateSize(e.sim.model, mask))
    mujoco.mj_getState(e.sim.model, e.sim.data, integration, mask)
    assert np.array_equal(integration, archive['state']['integration'])
    snap = TrainingBranch(task)
    runs = []
    for mode in ('original', 'actual_lambda033'):
        snap.restore()
        p0, c0 = len(task.physics), len(task.controls)
        rewards, stats, inputs, commands = [], [], [], []
        for _ in range(21):
            x = torch.cat([task.student_features(), task.features()], -1)
            raw = net.distribution(x).mean
            result = Original.step(task, raw) if mode == 'original' else task.step(raw)
            rewards.append(result[1]); stats.append(copy.deepcopy(result[3]))
            inputs.append(x.cpu().clone()); commands.append(raw.cpu().clone())
            if task.done: break
        runs.append(dict(physics=copy.deepcopy(task.physics[p0:]),
                         controls=copy.deepcopy(task.controls[c0:]),
                         rewards=rewards, stats=stats, inputs=inputs, commands=commands,
                         episode_return=task.episode_return))
    old, new = runs
    assert checks.equal(old['physics'], new['physics'])
    assert checks.equal(old['inputs'], new['inputs'])
    assert checks.equal(old['commands'], new['commands'])
    zero = positive = 0
    total = 0.
    errors = []
    extra = {'legacy_optimized_reward_integral','slip_risk_event_cost',
             'slip_risk_rate_equivalent','slip_risk_max_loaded_mps'}
    for i, (a, b) in enumerate(zip(old['controls'], new['controls'])):
        end = old['controls'][i+1]['time'] if i+1<len(old['controls']) else a['time']+a['actual_duration_s']+1e-8
        rows = [r for r in old['physics'] if a['time']-1e-8<=r['force_time']<end-1e-8]
        penalty = event(rows, .33)
        assert a['reward_integral']-penalty == b['reward_integral']
        assert b['legacy_optimized_reward_integral'] == a['reward_integral']
        assert b['slip_risk_event_cost'] == penalty
        assert checks.equal({k:v for k,v in a.items() if k!='reward_integral'},
                            {k:v for k,v in b.items() if k not in extra and k!='reward_integral'})
        zero += penalty == 0.; positive += penalty > 0.; total += penalty
    for a, b, ra, rb in zip(old['stats'], new['stats'], old['rewards'], new['rewards']):
        penalties = b['native_risk_event_costs']
        assert len(penalties) == len(a['native_rewards'])
        assert [v-p for v,p in zip(a['native_rewards'], penalties)] == b['native_rewards']
        assert ra-sum(penalties) == rb
        for key in ('terminated','truncated','actual_duration_s','boundary','administrative_native_limit'):
            assert checks.equal(a[key], b[key]), key
        errors.append(abs((ra-rb)-sum(penalties)))
    assert zero>0 and positive>0
    # Repeated subtraction may round episode accumulators differently. Report, do not hide.
    result = dict(status='completed',actual_RiskTask_step_lambda033=True,
        actual_original_step_comparison=True,physics_bitwise=True,inputs_and_raw_bitwise=True,
        all_other_control_fields_bitwise=True,native_and_high_reward_exact_formula=True,
        terminal_truncation_duration_masks_unchanged=True,source_state_integration_exact=True,
        source_state_id=23,source_prefix_commands=len(archive['actual_raw_prefix']),
        compared_high_decisions=len(old['stats']),compared_physics_ticks=len(old['physics']),
        compared_controls=len(old['controls']),zero_event_controls=int(zero),positive_event_controls=int(positive),
        event_sum=total,subtraction_reconstruction_max_roundoff=max(errors),
        episode_return_delta=old['episode_return']-new['episode_return'],
        new_onpolicy_RL_decisions=0,optimizer_updates=0,job_s=time.monotonic()-started)
    torch.save(runs, out/'actual-positive-path.pt')
    write(out/'checks.json',result)
    return result

checks.run=run
if __name__ == '__main__': entry.main()
