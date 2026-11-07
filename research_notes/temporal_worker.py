"""Actual physics samples with complete, causal AR command history in replay."""
import os
import json
import time
import traceback
from pathlib import Path

CONNECTION = None
WORKER_ID = 0


def launch(connection, gpu, index, output):
    global CONNECTION, WORKER_ID
    CONNECTION, WORKER_ID = connection, index
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu, MUJOCO_GL='disable',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      BFM_RECOVERY_OUTPUT=str(output), PYTHONDONTWRITEBYTECODE='1',
                      CLEAN_KIND='direct')
    output = Path(output)
    output.mkdir()
    os.chdir(output)
    log = open(output/'worker.log', 'a', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    try:
        import m10_stage, m10_physics
        m10_stage.run_study = loop
        m10_physics.main()
    except BaseException:
        error = traceback.format_exc()
        (output/'failure.txt').write_text(error)
        try:
            connection.send(dict(kind='error', error=error, worker_id=index))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


def loop(env, model, contract, out):
    import torch
    import numpy as np
    from native_env import Task, install
    from temporal_policy import Expert, rho_for
    from m11_runtime_speed import install as speed
    install()
    (out/'runtime-speed.json').write_text(json.dumps(speed(env)))
    torch.set_num_threads(1)
    seed = int(os.environ['NATIVE_SEED'])
    arm = os.environ['TEMPORAL_ARM']
    torch.manual_seed(seed*997+WORKER_ID)
    task = Task(env, model, contract, seed*997+WORKER_ID+int(os.environ['TEMPORAL_SOURCE_OFFSET']))
    obs = task.reset(record=True, label=f'train-w{WORKER_ID}-0')
    net = Expert(obs.shape[-1], rho=rho_for(arm)).to(obs.device).eval()
    episodes = 0
    CONNECTION.send(dict(kind='ready', pid=os.getpid(), input_dim=obs.shape[-1],
                         worker=WORKER_ID, contract=contract,
                         initial_source_physics_steps=task.source_physics_total))
    while True:
        cmd = CONNECTION.recv()
        if cmd['op'] == 'close':
            break
        net.load_state_dict({k: torch.from_numpy(v).to(obs.device)
                             for k, v in cmd['policy'].items()})
        if cmd['op'] == 'evaluate':
            summaries = []
            for label, spec, mode in cmd['schedule']:
                obs = task.reset(spec, record=True, label=label)
                while not task.done:
                    raw = net.raw_mean(obs)
                    if mode == 'standing':
                        raw = env.standing/16
                    obs, _, _, _ = task.step(raw)
                dest = Path(cmd['output'])
                dest.mkdir(exist_ok=True)
                summaries.append(task.save(dest))
            obs = task.reset(record=True, label=f'train-w{WORKER_ID}-{episodes}')
            CONNECTION.send(dict(kind='evaluate', summaries=summaries))
            continue
        assert cmd['op'] == 'rollout'
        generator = torch.Generator(device=obs.device).manual_seed(
            seed*1000003+WORKER_ID*1001+cmd['version']*100003)
        # Every new frozen policy version starts the AR process in stationarity.
        # This resets ONLY exploration memory. Real physics/BFM history remains.
        previous = torch.zeros_like(obs)
        previous_u = torch.zeros((1, 256), device=obs.device)
        valid = False
        batch = {k: [] for k in ['obs', 'action', 'logp', 'value', 'next_value',
                                 'reward', 'discount', 'done']}
        audit = {k: [] for k in ['current_obs', 'previous_obs', 'previous_u',
                                 'conditional_mean', 'conditional_std', 'noise',
                                 'innovation', 'noise_reset', 'policy_version', 'time']}
        stats, eps = [], []
        start = time.monotonic()
        prefill, source = task.total_prefill, task.source_physics_total
        with torch.no_grad():
            for _ in range(cmd['steps']):
                if task.done:
                    obs = task.reset(record=True, label=f'train-w{WORKER_ID}-{episodes}')
                    valid = False
                packed = net.pack(obs, previous, previous_u, valid)
                d = net.distribution(packed)
                innovation = torch.randn(d.mean.shape, device=obs.device, generator=generator)
                u = d.mean+d.stddev*innovation
                raw = net.to_raw(u)
                noise = u-net.distribution(obs).mean
                lp, value = d.log_prob(u).sum(-1), net.value(obs)
                t = float(env.tether.time)
                nxt, reward, done, st = task.step(raw)
                nv = net.value(nxt)
                for k, v in dict(obs=packed[0], action=u[0], logp=lp[0],
                                 value=value[0], next_value=nv[0]).items():
                    batch[k].append(v.detach().cpu().numpy())
                batch['reward'].append(reward)
                batch['discount'].append(.99**(st['control_steps']/5))
                batch['done'].append(done)
                for k, v in dict(current_obs=obs[0], previous_obs=previous[0],
                                 previous_u=previous_u[0], conditional_mean=d.mean[0],
                                 conditional_std=d.stddev[0], noise=noise[0],
                                 innovation=innovation[0]).items():
                    audit[k].append(v.detach().cpu().numpy())
                audit['noise_reset'].append(not valid)
                audit['policy_version'].append(cmd['version'])
                audit['time'].append(t)
                st.update(time=t, raw_sample_norm=float(raw.norm()),
                          raw_mean_norm=float(net.raw_mean(obs).norm()),
                          raw_noise_norm=float(net.to_raw(noise).norm()),
                          sample_vs_mean_z_angle_deg=float(torch.acos(torch.nn.functional.cosine_similarity(raw,net.raw_mean(obs)).clamp(-1,1))[0]*180/torch.pi),
                          innovation_std=float(d.stddev.mean()), noise_reset=not valid,
                          probability_u_norm=float(u.norm()))
                stats.append(st)
                previous, previous_u, valid = obs.clone(), u.clone(), True
                obs = nxt
                if done:
                    eps.append(task.summary())
                    if task.boundary or episodes % 40 == 0:
                        task.save(out)
                    episodes += 1
        arrays = {k: np.asarray(v, dtype=bool if k == 'done' else np.float32)
                  for k, v in batch.items()}
        np.savez_compressed(out/f'rollout-{cmd["version"]:04d}.npz', **arrays)
        np.savez_compressed(out/f'rollout-{cmd["version"]:04d}-noise.npz', **audit)
        (out/f'rollout-{cmd["version"]:04d}-stats.json').write_text(json.dumps(stats))
        CONNECTION.send(dict(kind='rollout', version=cmd['version'], batch=arrays,
                             stats=stats, episodes=eps, seconds=time.monotonic()-start,
                             prefill_physics_steps=task.total_prefill-prefill,
                             source_physics_steps=task.source_physics_total-source))
    return dict(status='completed', episodes=episodes, physical_steps=task.total_steps,
                source_physics_steps=task.source_physics_total, old_actor_loaded=False)
