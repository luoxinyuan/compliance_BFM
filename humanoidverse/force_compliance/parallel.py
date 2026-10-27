"""Synchronous multi-GPU rollout collection; one shared PPO policy, frozen BFM workers.

Each spawned process owns one native MuJoCo environment (the upstream adapter is
process-global). CUDA_VISIBLE_DEVICES is set before importing torch in a worker.
Only arrays are sent through pipes, avoiding CUDA IPC and shared tensor lifetimes.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time
import traceback


def force_level_seeds(cfg, start=303):
    """Choose reproducible audit seeds covering every nominal level."""
    import numpy as np
    if not cfg.nominal_force_levels:
        return [start]
    selected = {}
    for seed in range(start, start + 100000):
        rng = np.random.default_rng(seed)
        if cfg.direction is None:
            rng.normal(size=3)
        level = float(rng.choice(cfg.nominal_force_levels))
        selected.setdefault(level, seed)
        if len(selected) == len(set(cfg.nominal_force_levels)):
            return [selected[level] for level in dict.fromkeys(cfg.nominal_force_levels)]
    raise RuntimeError('Could not cover all nominal force levels')


def worker(connection, gpu, worker_id, model_folder, config, output):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    env = None
    try:
        # Redirect verbose upstream initialization logs independently per worker.
        logfile = open(Path(output) / f'worker_{worker_id:02d}.log', 'a', buffering=1)
        os.dup2(logfile.fileno(), 1)
        os.dup2(logfile.fileno(), 2)
        import numpy as np
        import torch
        from .config import ComplianceConfig
        from .environment import ComplianceEnv
        from .policy import ForceLatentPolicy
        from .run import evaluate
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        config = dict(config, seed=config['seed'] + worker_id * 100003)
        cfg = ComplianceConfig(**config)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        env = ComplianceEnv(model_folder, cfg, device='cuda')
        assert not any(p.requires_grad for p in env.model.parameters())
        observation = env.observation()
        policy = ForceLatentPolicy(observation.shape[-1], env.standing.shape[-1]).cuda().eval()
        episode_reward, episode_force_squared, episode_steps = 0., 0., 0
        connection.send(dict(kind='ready', worker_id=worker_id, gpu=gpu, pid=os.getpid(),
                             observation_dim=observation.shape[-1], latent_dim=env.standing.shape[-1],
                             standing=env.standing.cpu().numpy(), dt=env.dt, interval=env.interval))
        while True:
            command = connection.recv()
            if command['op'] == 'close':
                break
            if 'policy' in command:
                policy.load_state_dict({k: torch.from_numpy(v).cuda() for k, v in command['policy'].items()})
            if 'standing' in command:
                env.standing = torch.from_numpy(command['standing']).cuda()
                observation = env.reset(seed=cfg.seed)
            if command['op'] in ('preflight', 'evaluate'):
                summaries = {}
                if command.get('no_force'):
                    cfg.enabled = False
                    _, _, summaries['no_force'] = evaluate(env, policy, True, cfg.seed, contact_index=0)
                    cfg.enabled = True
                for index in command['contacts']:
                    for seed in command.get('seeds', [101]):
                        trajectory, _, baseline = evaluate(env, policy, True, seed, contact_index=index)
                        key = f'{index:02d}_{seed}'
                        summaries[key] = dict(baseline=baseline)
                        if command['op'] == 'evaluate':
                            _, _, learned = evaluate(env, policy, False, seed, trajectory=trajectory, contact_index=index)
                            summaries[key]['learned'] = learned
                observation = env.reset(seed=cfg.seed + command.get('version', 0) * 65537)
                episode_reward, episode_force_squared, episode_steps = 0., 0., 0
                connection.send(dict(kind=command['op'], summaries=summaries))
                continue
            if command['op'] != 'rollout':
                raise ValueError(command['op'])
            batch = {k: [] for k in ('obs', 'action', 'logp', 'value', 'next_value', 'reward', 'discount', 'done')}
            contacts = defaultdict(lambda: dict(low_steps=0, force_squared=0., episodes=0, failures=0, returns=0.))
            started = time.monotonic()
            with torch.no_grad():
                for _ in range(command['steps']):
                    dist = policy.distribution(observation)
                    raw = dist.sample()
                    value = policy.value(observation)
                    logp = dist.log_prob(raw).sum(-1)
                    successor, reward, done, info = env.step(raw)
                    entries = dict(obs=observation[0], action=raw[0], logp=logp[0], value=value[0],
                                   next_value=policy.value(successor)[0])
                    for key, val in entries.items():
                        batch[key].append(val.cpu().numpy().copy())
                    batch['reward'].append(reward)
                    batch['discount'].append(info['discount'])
                    batch['done'].append(done)
                    stats = contacts[info['contact_name']]
                    stats['low_steps'] += len(info['rows'])
                    stats['force_squared'] += sum(r['force_squared'] for r in info['rows'])
                    episode_steps += len(info['rows'])
                    episode_reward += sum(r['reward'] for r in info['rows'])
                    episode_force_squared += sum(r['force_squared'] for r in info['rows'])
                    if done:
                        stats['episodes'] += 1
                        stats['failures'] += int(info['failed'])
                        stats['returns'] += episode_reward
                        episode_reward, episode_force_squared, episode_steps = 0., 0., 0
                        observation = env.reset()
                    else:
                        observation = successor
            arrays = {k: np.asarray(v, dtype=bool if k == 'done' else np.float32) for k, v in batch.items()}
            connection.send(dict(kind='rollout', version=command['version'], batch=arrays,
                                 contacts=dict(contacts), seconds=time.monotonic()-started))
    except BaseException:
        error = traceback.format_exc()
        try:
            connection.send(dict(kind='error', error=error, worker_id=worker_id))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


def receive(connection, process, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if connection.poll(1):
            result = connection.recv()
            if result.get('kind') == 'error':
                raise RuntimeError(f"Worker failed: {result['error']}")
            return result
        if not process.is_alive():
            raise RuntimeError(f'Worker {process.pid} exited ({process.exitcode})')
    raise TimeoutError(f'Worker {process.pid} did not respond within {timeout}s')


def merge_rollouts(results, trace_decay):
    """GAE runs along time separately for each worker, BEFORE flattening."""
    import numpy as np
    import torch
    from .ppo import advantages
    batch = {k: torch.from_numpy(np.stack([r['batch'][k] for r in results], axis=1))
             for k in results[0]['batch']}
    adv, returns = advantages(batch['reward'], batch['value'], batch['next_value'],
                              batch['discount'], batch['done'], trace_decay)
    batch = {k: v.flatten(0, 1) for k, v in batch.items()}
    batch['returns'] = returns.flatten()
    adv = adv.flatten()
    batch['advantages'] = (adv-adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
    return batch


def update(policy, optimizer, batch, epochs=4, minibatch=256, target_kl=.03):
    import torch
    kl_values, clips, losses = [], [], []
    for _ in range(epochs):
        stop = False
        for indices in torch.randperm(len(batch['reward'])).split(minibatch):
            dist = policy.distribution(batch['obs'][indices])
            log_ratio = dist.log_prob(batch['action'][indices]).sum(-1) - batch['logp'][indices]
            ratio = log_ratio.exp()
            kl = float(((ratio-1)-log_ratio).mean().detach())
            if kl > target_kl * 1.5:
                stop = True
                break
            adv = batch['advantages'][indices]
            actor = -torch.minimum(ratio*adv, ratio.clamp(.8, 1.2)*adv).mean()
            critic = .5*(policy.value(batch['obs'][indices])-batch['returns'][indices]).square().mean()
            loss = actor + critic - .001*dist.entropy().sum(-1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite PPO loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
            kl_values.append(kl)
            clips.append(float(((ratio-1).abs() > .2).float().mean()))
            losses.append(float(loss.detach()))
        if stop:
            break
    with torch.no_grad():
        final_log_ratio = policy.distribution(batch['obs']).log_prob(batch['action']).sum(-1) - batch['logp']
        final_kl = float(((final_log_ratio.exp()-1)-final_log_ratio).mean())
    return dict(policy_kl=final_kl, loss=sum(losses)/max(len(losses),1), approximate_kl=sum(kl_values)/max(len(kl_values),1),
                clip_fraction=sum(clips)/max(len(clips),1), gradient_steps=len(losses), kl_early_stop=stop)


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-folder', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('whole_body.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[2, 3, 4])
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--rollout', type=int, default=128)
    parser.add_argument('--total-decisions', type=int, default=1_000_000)
    parser.add_argument('--max-hours', type=float, default=24.)
    parser.add_argument('--checkpoint-every', type=int, default=5)
    parser.add_argument('--evaluate-every', type=int, default=50)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--actor-lr', type=float, default=3e-5)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    args = parser.parse_args()
    if min(args.workers_per_gpu, args.rollout, args.total_decisions, args.checkpoint_every, args.evaluate_every) < 1 or args.max_hours <= 0:
        parser.error('Counts and max-hours must be positive')
    if args.output.resolve().is_relative_to(args.model_folder.resolve()):
        parser.error('Output must be outside the pretrained model folder')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'status.json').exists():
        parser.error('Use a fresh output directory; --resume can point to an older run')
    # Learner is small and stays on CPU; only rollout processes allocate CUDA.
    import torch
    from .config import ComplianceConfig
    from .policy import ForceLatentPolicy
    cfg = ComplianceConfig(**json.loads(args.config.read_text()))
    if not cfg.contact_points or not cfg.enabled or cfg.release_time is not None:
        parser.error('Whole-body training requires contact_points and a maintained enabled tether')
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    atomic_json(args.output/'config.json', asdict(cfg))
    atomic_json(args.output/'launch.json', dict(pid=os.getpid(), **{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}))
    context = mp.get_context('spawn')
    processes, connections, ready = [], [], []
    status = 'initializing'
    decisions, iteration, best_score = 0, 0, -float('inf')
    policy = optimizer = None
    def stop_signal(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, stop_signal)
    started = time.monotonic()
    def save(filename='policy.pt'):
        snapshot = dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(), config=asdict(cfg),
                        observation_dim=ready[0]['observation_dim'], latent_dim=ready[0]['latent_dim'],
                        standing_latent=torch.from_numpy(ready[0]['standing']),
                        model_folder=str(args.model_folder.resolve()), decisions=decisions, iteration=iteration,
                        best_score=best_score, torch_rng_state=torch.get_rng_state(), observation_version='contact_one_hot_v1')
        temporary = args.output/(filename+'.tmp')
        torch.save(snapshot, temporary)
        temporary.replace(args.output/filename)
    def weights():
        return {k:v.detach().cpu().numpy() for k,v in policy.state_dict().items()}
    def audit(op, seeds):
        for index, connection in enumerate(connections):
            connection.send(dict(op=op, contacts=list(range(index,len(cfg.contact_points),len(connections))),
                                 no_force=(op=='preflight' and index==0), seeds=seeds,
                                 policy=weights(), standing=ready[0]['standing'], version=iteration))
        summaries = {}
        for connection, process in zip(connections, processes):
            summaries.update(receive(connection, process)['summaries'])
        return summaries
    try:
        for gpu in args.gpus:
            for _ in range(args.workers_per_gpu):
                parent, child = context.Pipe()
                index = len(processes)
                process = context.Process(target=worker, args=(child,gpu,index,str(args.model_folder.resolve()),asdict(cfg),str(args.output.resolve())))
                process.start()
                child.close()
                connections.append(parent)
                processes.append(process)
        for connection, process in zip(connections, processes):
            ready.append(receive(connection, process))
        dims = {(r['observation_dim'],r['latent_dim'],r['interval']) for r in ready}
        if len(dims) != 1:
            raise RuntimeError('Worker observation/action contracts differ')
        policy = ForceLatentPolicy(ready[0]['observation_dim'], ready[0]['latent_dim'])
        optimizer = torch.optim.Adam([
            dict(params=list(policy.actor.parameters())+[policy.log_std], lr=args.actor_lr),
            dict(params=policy.critic.parameters(), lr=args.critic_lr)])
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
            if checkpoint['config'] != asdict(cfg) or Path(checkpoint['model_folder']).resolve() != args.model_folder.resolve():
                raise ValueError('Resume configuration or BFM directory mismatch')
            policy.load_state_dict(checkpoint['policy'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            ready[0]['standing'] = checkpoint['standing_latent'].numpy()
            decisions, iteration = checkpoint['decisions'], checkpoint['iteration']
            best_score = checkpoint['best_score']
            torch.set_rng_state(checkpoint['torch_rng_state'])
        atomic_json(args.output/'workers.json', [{k:v for k,v in r.items() if k!='standing'} for r in ready])
        status = 'preflight'
        atomic_json(args.output/'status.json', dict(status=status, decisions=decisions))
        baseline = audit('preflight', force_level_seeds(cfg, 101) if cfg.nominal_force_levels else [101, 202])
        atomic_json(args.output/'preflight.json', baseline)
        if baseline['no_force']['fall_rate'] or any(v['baseline']['fall_rate'] for k,v in baseline.items() if k!='no_force'):
            raise RuntimeError('Frozen baseline fell during preflight; inspect per-contact results')
        if any(v['baseline']['peak_force'] < 1e-3 for k,v in baseline.items() if k!='no_force'):
            raise RuntimeError('Preflight has an interaction with no measured force')
        save('initial.pt')
        status = 'running'
        started = time.monotonic()
        initial_decisions = decisions
        while decisions < args.total_decisions and time.monotonic()-started < args.max_hours*3600:
            cycle = time.monotonic()
            payload = dict(op='rollout', policy=weights(), version=iteration, steps=args.rollout)
            for connection in connections:
                connection.send(payload)
            results = [receive(c,p) for c,p in zip(connections,processes)]
            if any(r['version'] != iteration for r in results):
                raise RuntimeError('Stale rollout policy version')
            sampling_seconds = time.monotonic()-cycle
            batch = merge_rollouts(results, cfg.gae_lambda)
            learning = time.monotonic()
            metrics = update(policy, optimizer, batch)
            update_seconds = time.monotonic()-learning
            decisions += len(batch['reward'])
            iteration += 1
            contacts = defaultdict(lambda: dict(low_steps=0, force_squared=0., episodes=0, failures=0, returns=0.))
            for result in results:
                for name, values in result['contacts'].items():
                    for key,value in values.items():
                        contacts[name][key] += value
            for values in contacts.values():
                values['rms_force'] = (values['force_squared']/max(values['low_steps'],1))**.5
            elapsed = time.monotonic()-started
            rate = (decisions-initial_decisions)/elapsed
            log = dict(status=status, iteration=iteration, decisions=decisions, elapsed_seconds=elapsed,
                       decisions_per_second=rate, eta_hours=max(0,args.total_decisions-decisions)/rate/3600,
                       sampling_seconds=sampling_seconds, update_seconds=update_seconds,
                       mean_reward=float(batch['reward'].mean()), contacts=dict(contacts), **metrics)
            with (args.output/'training.jsonl').open('a') as handle:
                handle.write(json.dumps(log)+'\n')
            atomic_json(args.output/'status.json', log)
            print(json.dumps({k:v for k,v in log.items() if k!='contacts'}),flush=True)
            if iteration % args.checkpoint_every == 0:
                save()
            if iteration % args.evaluate_every == 0:
                summaries = audit('evaluate', force_level_seeds(cfg))
                atomic_json(args.output/f'evaluation_{iteration:06d}.json',summaries)
                score = sum(v['learned']['episode_return'] for v in summaries.values())/len(summaries)
                if score > best_score:
                    best_score = score
                    save('best.pt')
                save()
        status = 'complete' if decisions >= args.total_decisions else 'time_budget_reached'
        save()
    except KeyboardInterrupt:
        status = 'interrupted'
        if policy is not None and optimizer is not None and ready:
            save()
    except BaseException:
        status = 'failed'
        (args.output/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        final = dict(status=status,decisions=decisions,iteration=iteration)
        atomic_json(args.output/'final_status.json', final)
        last = json.loads((args.output/'status.json').read_text()) if (args.output/'status.json').exists() else {}
        atomic_json(args.output/'status.json', dict(last, **final))
        for connection in connections:
            try:
                connection.send(dict(op='close'))
            except (BrokenPipeError,EOFError,OSError):
                pass
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for connection in connections:
            connection.close()


if __name__ == '__main__':
    main()
