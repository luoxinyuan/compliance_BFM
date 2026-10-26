"""python -m humanoidverse.force_compliance.run --help"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from .config import ComplianceConfig
from .environment import ComplianceEnv
from .policy import ForceLatentPolicy
from .ppo import train


def evaluate(env, policy, baseline, seed, trajectory=None, video_path=None):
    observation = env.reset(seed=seed, trajectory=trajectory)
    trajectory = (env.tether.origin.copy(), env.tether.delta.copy())
    rows, frames = [], []
    while True:
        with torch.no_grad():
            raw = policy.distribution(observation).mean
        observation, _, done, info = env.step(raw, baseline=baseline)
        rows.extend(info['rows'])
        if video_path is not None:
            frames.append(env.render())
        if done:
            break
    if video_path is not None:
        import mediapy
        import shutil
        if shutil.which('ffmpeg') is None:
            import imageio_ffmpeg
            mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        mediapy.write_video(video_path, frames, fps=env.actual_hz)
    forces = np.array([r['force'] for r in rows])
    active = np.array([r['time'] >= env.cfg.activation_time for r in rows])
    means = np.array([r['mean_force'] for r in rows])
    selected = means[active]
    peaks = np.array([r['peak_force'] for r in rows])
    settling = None
    stop = env.cfg.activation_time + env.cfg.anchor_delay + env.cfg.ramp_duration
    window = max(1, int(np.ceil(env.cfg.settling_window / env.dt)))
    if env.cfg.enabled and np.linalg.norm(trajectory[1]) > 0 and not info['failed']:
        for i, row in enumerate(rows):
            if row['time'] >= stop and len(rows)-i >= window and np.all(peaks[i:] <= env.cfg.settling_force):
                settling = row['time'] - stop
                break
    supports = np.asarray([r['support'] for r in rows], dtype=int)
    metrics = dict(peak_force=float(max(r['peak_force'] for r in rows)), mean_force=float(means.mean()),
                   rms_force=float(np.sqrt(np.mean([r['force_squared'] for r in rows]))),
                   active_mean_force=float(selected.mean()) if len(selected) else None,
                   settling_time=settling, fall_rate=float(info['failed']),
                   episode_return=float(sum(r['reward'] for r in rows)), duration=rows[-1]['time'],
                   point_displacement=rows[-1]['point_displacement'], root_displacement=rows[-1]['root_displacement'],
                   joint_movement=rows[-1]['joint_movement'], mean_anchor_error=float(np.mean([r['error'] for r in rows])),
                   support_changes=int(np.abs(np.diff(supports, axis=0)).sum()),
                   latent_variation=float(sum(r['latent_variation'] for r in rows)),
                   anchor_displacement=float(np.linalg.norm(trajectory[1])))
    return trajectory, rows, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['baseline', 'train', 'evaluate'])
    parser.add_argument('--model-folder', type=Path, required=True)
    parser.add_argument('--standing-file', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/force_compliance'))
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--rollout', type=int, default=32)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    if args.iterations < 1 or args.rollout < 2 or args.episodes < 1:
        parser.error('Require iterations >= 1, rollout >= 2 and episodes >= 1')
    saved = torch.load(args.policy, map_location=args.device, weights_only=True) if args.policy else None
    cfg = ComplianceConfig(**(json.loads(args.config.read_text()) if args.config else saved['config'] if saved else {}))
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    # Never allow high-level artifacts anywhere inside the pretrained model directory.
    if args.output.resolve().is_relative_to(args.model_folder.resolve()):
        parser.error('--output must be outside the pretrained model directory')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'config.json').write_text(json.dumps(asdict(cfg), indent=2))
    env = ComplianceEnv(args.model_folder, cfg, args.standing_file, args.device)
    try:
        (args.output / 'runtime.json').write_text(json.dumps(dict(standing_source=env.standing_source,
             actual_high_level_hz=env.actual_hz, low_level_dt=env.dt,
             latent_dim=env.standing.shape[-1], latent_norm=float(env.standing.norm())), indent=2))
        torch.save(env.standing.cpu(), args.output / 'standing_latent.pt')
        dim = env.observation().shape[-1]
        policy = ForceLatentPolicy(dim, env.standing.shape[-1]).to(args.device)
        if saved:
            if Path(saved['model_folder']).resolve() != args.model_folder.resolve():
                raise ValueError('Policy checkpoint was trained with a different BFM directory')
            policy.load_state_dict(saved['policy'])
            env.standing = saved['standing_latent'].to(args.device)
            env.reset(seed=cfg.seed)
        if args.mode == 'train':
            # Actual frozen-policy smoke validation precedes any optimization.
            original = cfg.enabled
            cfg.enabled = False
            _, _, zero = evaluate(env, policy, True, cfg.seed)
            cfg.enabled = original
            _, baseline_rows, pulling = evaluate(env, policy, True, cfg.seed)
            validation = dict(no_force=zero, pulling=pulling)
            (args.output / 'preflight.json').write_text(json.dumps(validation, indent=2))
            if zero['fall_rate'] or pulling['fall_rate']:
                raise RuntimeError('Frozen standing baseline fell; inspect preflight.json before training')
            if cfg.enabled and pulling['peak_force'] < 1e-3:
                raise RuntimeError('No measurable interaction; refusing an empty training task')
            env.reset(seed=cfg.seed)
            optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
            assert not any(p.requires_grad for p in env.model.parameters())
            frozen_ids = {id(p) for p in env.model.parameters()}
            assert not any(id(p) in frozen_ids for g in optimizer.param_groups for p in g['params'])
            logs = train(env, policy, optimizer, args.iterations, args.rollout)
            torch.save(dict(policy=policy.state_dict(), config=asdict(cfg), observation_dim=dim,
                            latent_dim=env.standing.shape[-1], standing_latent=env.standing.cpu(),
                            model_folder=str(args.model_folder.resolve())), args.output / 'policy.pt')
            (args.output / 'training.json').write_text(json.dumps(logs, indent=2))
        else:
            if args.mode == 'evaluate' and not saved:
                parser.error('evaluate requires --policy; random policy is not a trained comparison')
            metrics = {}
            for episode in range(args.episodes):
                trajectory = None
                for name in (['standing'] if args.mode == 'baseline' else ['standing', 'learned']):
                    prefix = f'{episode:03d}_{name}'
                    trajectory, rows, summary = evaluate(env, policy, name == 'standing', cfg.seed+episode, trajectory,
                                                        args.output / f'{prefix}.mp4' if args.video else None)
                    metrics[prefix] = summary
                    (args.output / f'{prefix}.json').write_text(json.dumps(dict(trajectory=[v.tolist() for v in trajectory], rows=rows), indent=2))
            (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                for path in sorted(args.output.glob('*_standing.json')) + sorted(args.output.glob('*_learned.json')):
                    rows = json.loads(path.read_text())['rows']
                    plt.plot([r['time'] for r in rows], [r['force'] for r in rows], label=path.stem)
                plt.xlabel('Time (s)'); plt.ylabel('Applied tether force (N)'); plt.legend()
                plt.savefig(args.output / 'forces.png'); plt.close()
            except ImportError:
                print('matplotlib unavailable; force traces are stored as JSON')
            print(json.dumps(metrics, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
