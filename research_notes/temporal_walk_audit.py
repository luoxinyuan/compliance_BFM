"""Three fixed native motion controls; no learned actor, reward scan or labels."""
import json
from pathlib import Path
import transfer_student_validated as runner


def study(env, model, contract, out):
    import numpy as np
    import torch
    from native_env import Task, spec, install
    from m9_cyclic import sample
    from locomotion_rollout import spherical
    from m11_runtime_speed import install as speed
    install()
    runner.write(out/'runtime-speed.json', speed(env))
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    task = Task(env, model, contract, 4900)
    rows = []
    for label, rate, force in [('full-contact', 1., True),
                               ('half-free', .5, False),
                               ('half-contact', .5, True)]:
        s = spec(.72, 8., 4.9, until=3., enabled=force)
        task.reset(s, record=True, label=label)
        previous = env.standing.clone()
        commands = []
        while not task.done:
            t = float(env.tether.time)
            if t < 4.9:
                z = env.standing
            elif t < 12.9:
                u = float(np.clip((t-4.9)/1.5, 0., 1.))
                u = u*u*(3-2*u)
                phase = rate*max(0., t-6.4)/task.refs['straight']['period']
                z = spherical(env.standing, sample(task.refs['straight'], phase), u)
                previous = z.clone()
            elif t < 14.4:
                z = spherical(previous, env.standing, (t-12.9)/1.5)
            else:
                z = env.standing
            commands.append(dict(time=t, rate=rate, native_z=z[0].cpu().tolist()))
            task.step(z/16)
        row = task.save(out)
        row['native_control_decisions'] = row['rl_decisions']
        row['rl_decisions'] = 0
        row.update(native_phase_rate=rate,
                   scope='clocked native capability control, NOT feedback policy or PPO data',
                   no_learner_labels=True)
        rows.append(row)
        runner.write(out/'summaries.json', rows)
        runner.write(out/(label+'-commands.json'), commands)
        print('NATIVE_WALK_AUDIT', label, row['error_cm'], row['force_rms'],
              row['root_max_displacement'], row['boundary'], flush=True)
    return dict(status='completed', episodes=len(rows), new_candidates=1,
                original_motion=True, alternative='same motion half phase rate',
                RL_decisions=0, supervised_updates=0,
                old_high_actor_loaded=False,
                free_full_rate_evidence='Issue48 qualification-03/05-motion_walk_task-free.json.gz')


runner.study = study
if __name__ == '__main__':
    runner.main()
