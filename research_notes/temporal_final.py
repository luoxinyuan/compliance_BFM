"""Fixed sealed physical evaluation after all development/extension decisions."""
import json
from pathlib import Path
import transfer_student_validated as runner

O = Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/native-temporal-20261107')


def study(env, model, contract, out):
    import numpy as np
    import torch
    from native_env import Task, install, spec, RUN as OLD
    from native_policy import Expert as NativeRaw
    from native_policy_coordinate import Expert as NativeCoordinate
    from temporal_policy import Expert
    from clean_policy import Expert as OldExpert
    from geometry_policy import expand
    from m11_runtime_speed import install as speed
    install()
    runner.write(out/'runtime-speed.json', speed(env))
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    task = Task(env, model, contract, 4910)
    nets, identities = {}, {}
    groups = json.loads((O/'frozen-groups.json').read_text())
    for g in groups:
        for tag in ['best', 'last']:
            p = O/g['final_root']/(tag+'.pt')
            ck = torch.load(p, map_location='cuda', weights_only=True)
            n = Expert(ck['input_dim']).cuda()
            n.load_state_dict(ck['state_dict'])
            key = g['name']+'_'+tag
            nets[key] = n.eval().requires_grad_(False)
            identities[key] = dict(path=str(p), sha256=runner.sha(p), iteration=ck['iteration'],
                                   runtime=ck['runtime'], arm=g['arm'], seed=g['seed'])
    for name, path, kind in [('native_initial', OLD/'native-initial.pt', NativeRaw),
                             ('C32', OLD/'ppo-coordinate/last.pt', NativeCoordinate)]:
        ck = torch.load(path, map_location='cuda', weights_only=True)
        n = kind(ck['input_dim']).cuda()
        n.load_state_dict(ck['state_dict'])
        nets[name] = n.eval().requires_grad_(False)
        identities[name] = dict(path=str(path), sha256=runner.sha(path), runtime=ck['runtime'],
                                role='frozen evaluation only')
    base = O.parent
    for name, p, sha in [('I', base/'m39a-demand-20261104/B-ppo-03/job/best.pt',
                         '72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a'),
                         ('C0', base/'prior-recovery-20261105/correct-C0/C0.pt',
                          'ffe0eb0666da12c99788e039061b27e8738051c276a3ca41bebf656d9a96352c')]:
        assert runner.sha(p) == sha
        ck = torch.load(p, map_location='cuda', weights_only=True)
        n = OldExpert(1088, 'direct').cuda()
        expand(n, ck)
        nets[name] = n.eval().requires_grad_(False)
        identities[name] = dict(path=str(p), sha256=sha, role='frozen evaluation only, no native training source')
    runner.write(out/'models.json', identities)
    cases = {'new_near': spec(.061, 2.05, 5.31), 'new_mid': spec(.232, 3.35, 5.31),
             'new_long': spec(.735, 8.45, 5.31), 'new_stop': spec(.735, 8.45, 5.31, stop=8.76),
             'new_continue': spec(.735, 8.45, 5.31),
             'new_arm': spec(.232, 3.35, 6.57, 'goal_arm5', 5.83),
             'new_moving': spec(.465, 6.15, 6.91, 'motion_walk_task', 6.34),
             'new_moving_stop': spec(.465, 6.15, 6.91, 'motion_walk_task', 6.34, stop=9.56),
             'zero_connected': spec(0., 4., 5.31), 'no_contact': spec(0., 4., 5.31, enabled=False),
             'light5': spec(.061, 2.05, 5.31, cap=5.)}
    runner.write(out/'conditions.json', cases)
    rows, prefix = [], {}
    for name, net in nets.items():
        for case, s in cases.items():
            task.reset(s, record=True, label=name+'__'+case)
            commands = []
            while not task.done:
                x = task.features()
                raw = net.raw_mean(x) if hasattr(net, 'raw_mean') else net.distribution(x).mean
                commands.append(dict(time=float(env.tether.time), x=x.cpu().tolist(), raw=raw.cpu().tolist()))
                task.step(raw)
            row = task.save(out)
            row.update(model=name, case=case, execution='deterministic mean')
            rows.append(row)
            runner.write(out/'summaries.json', rows)
            print('TEMPORAL_FINAL', name, case, row['error_cm'], row['force_rms'], row['boundary'], flush=True)
            if case in ['new_stop', 'new_continue']:
                prefix[(name, case)] = dict(physics=[r for r in task.physics if r['force_time'] < 8.76-1e-8],
                                           commands=[r for r in commands if r['time'] < 8.76-1e-8])
    checks = []
    for name in nets:
        a, b = prefix[(name, 'new_stop')], prefix[(name, 'new_continue')]
        same = {}
        for key in ['qpos', 'qvel', 'force', 'wrist', 'anchor']:
            aa, bb = np.array([r[key] for r in a['physics']]), np.array([r[key] for r in b['physics']])
            same[key] = float(np.max(abs(aa-bb))) if aa.shape == bb.shape and aa.size else None
        for key in ['x', 'raw']:
            aa, bb = np.array([r[key] for r in a['commands']]), np.array([r[key] for r in b['commands']])
            same[key] = float(np.max(abs(aa-bb))) if aa.shape == bb.shape and aa.size else None
        checks.append(dict(model=name, prefix_end=8.76, max_abs_differences=same))
    runner.write(out/'common-prefix.json', checks)
    # Two fixed stochastic repetitions of long demand for every dev-selected arm/seed.
    # This diagnoses sampling/mean gaps, never selects a new model.
    for name in [g['name']+'_best' for g in groups]:
        net = nets[name]
        for noise_seed in [49401, 49402]:
            task.reset(cases['new_long'], record=True, label=name+'__stochastic-'+str(noise_seed))
            generator = torch.Generator(device='cuda').manual_seed(noise_seed)
            previous = torch.zeros_like(task.features())
            previous_u = torch.zeros((1, 256), device='cuda')
            valid = False
            while not task.done:
                x = task.features()
                d = net.distribution(net.pack(x, previous, previous_u, valid))
                u = d.mean+d.stddev*torch.randn(d.mean.shape, device='cuda', generator=generator)
                task.step(net.to_raw(u))
                previous, previous_u, valid = x.clone(), u.clone(), True
            row = task.save(out)
            row.update(model=name, case='new_long_noise'+str(noise_seed), execution='conditional stochastic', noise_seed=noise_seed)
            rows.append(row)
            runner.write(out/'summaries.json', rows)
    for case in ['new_near', 'new_long', 'zero_connected']:
        task.reset(cases[case], record=True, label='native_standing__'+case)
        while not task.done:
            task.step(env.standing/16)
        row = task.save(out)
        row.update(model='native_standing', case=case, execution='frozen standing control')
        rows.append(row)
        runner.write(out/'summaries.json', rows)
    return dict(status='completed', episodes=len(rows), models=len(nets),
                RL_training_decisions=0, no_final_reselection=True,
                stochastic_repetitions='2fixed long repeats per dev-selected arm/seed')


runner.study = study
if __name__ == '__main__':
    runner.main()
