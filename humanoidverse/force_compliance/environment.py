import json
from pathlib import Path
import joblib
import numpy as np
import torch
import mujoco
from .interaction import Tether
from .policy import latent_command
from .contacts import select_contact


class ComplianceEnv:
    """One real G1 through the existing BFM observation/action adapter."""
    def __init__(self, model_folder, cfg, standing_file=None, device='cpu'):
        from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
        from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
        folder = Path(model_folder)
        if not (folder / 'checkpoint/model/config.json').is_file():
            raise FileNotFoundError(f'Missing BFM checkpoint: {folder}/checkpoint/model/config.json')
        self.cfg, self.device = cfg, device
        self.model = load_model_from_checkpoint_dir(folder / 'checkpoint', device=device)
        self.model.eval()
        self.model.requires_grad_(False)
        if not self.model.cfg.archi.norm_z:
            raise ValueError('This experiment requires norm_z=True')
        standing_file = Path(standing_file) if standing_file else folder / 'reward_inference/reward_locomotion.pkl'
        env_config = json.loads((folder / 'config.json').read_text())['env']
        env_config.update(device=device, disable_domain_randomization=True, disable_obs_noise=True,
                          max_episode_length_s=cfg.episode_duration + 10)
        data_path = Path(__file__).parents[1] / 'data/lafan_29dof.pkl'
        if not Path(env_config.get('lafan_tail_path', '')).is_file():
            env_config['lafan_tail_path'] = str(data_path)
        overrides = env_config.setdefault('hydra_overrides', [])
        overrides.extend(['simulator=mujoco', 'env.config.headless=True',
                          'env.config.lie_down_init=False', 'env.config.lie_down_init_prob=0'])
        self.wrapped, _ = HumanoidVerseIsaacConfig(**env_config).build(1)
        self.base = self.wrapped._env
        self.sim = self.base.simulator
        for name, angle in cfg.initial_joint_positions.items():
            index = self.sim.dof_names.index(name)
            self.wrapped._default_pose_target_reset['dof_states'][:, index, 0] = angle
        if cfg.initial_root_height is not None:
            self.wrapped._default_pose_target_reset['root_states'][2] = cfg.initial_root_height
        self.dt = float(self.base.dt)
        self.interval = max(1, round(1 / (cfg.high_level_hz * self.dt)))
        self.actual_hz = 1 / (self.interval * self.dt)
        initial_obs, _ = self.wrapped.reset(seed=cfg.seed, to_numpy=False, reset_to_default_pose=True)
        if standing_file.is_file():
            latents = joblib.load(standing_file)
            standing = latents['move-ego-0-0'][0]
            self.standing_source = str(standing_file)
        else:
            if standing_file is not None and str(standing_file) != str(folder / 'reward_inference/reward_locomotion.pkl'):
                raise FileNotFoundError(standing_file)
            with torch.no_grad():
                standing = self.model.goal_inference(initial_obs)
            self.standing_source = 'goal_inference(default standing reset observation); requires stability validation'
        self.standing = torch.as_tensor(standing, device=device, dtype=torch.float32).reshape(1, -1)
        if self.standing.shape[1] != self.model.cfg.archi.z_dim or not torch.isfinite(self.standing).all() or self.standing.norm() < 1e-6:
            raise ValueError('Invalid standing latent')
        self.standing = self.model.project_z(self.standing)
        self.tether = Tether(self.sim.model, self.sim.data, cfg)
        self.keys = list(self.model.cfg.archi.actor.input_filter.key)
        if any(k not in ('state', 'last_action', 'history_actor') for k in self.keys):
            raise ValueError(f'Actor has unsupported/privileged inputs: {self.keys}')
        self._original_step = self.sim.simulate_at_each_physics_step
        self.sim.simulate_at_each_physics_step = self._physics_step
        self.rng = np.random.default_rng(cfg.seed)
        self.contact_rng = np.random.default_rng(cfg.seed + 104729)
        self.renderer = None
        self.reset()

    def _physics_step(self):
        wrench = self.tether.apply()
        force_squared = float(self.tether.force @ self.tether.force)
        previous_velocity = self.sim.data.qvel.copy()
        previous_warnings = self.sim.data.warning.number.copy()
        previous_time = self.sim.data.time
        try:
            self._original_step()
        finally:
            self.sim.data.xfrc_applied[self.tether.body] -= wrench
        self.tether.time += self.sim.model.opt.timestep
        mujoco.mj_forward(self.sim.model, self.sim.data)
        d, m, c = self.sim.data, self.sim.model, self.cfg
        # Capture physics BEFORE the original environment can auto-reset.
        finite = np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
        bad_physics = np.any(d.warning.number[4:] != previous_warnings[4:]) or d.time < previous_time
        self.failed |= bool(bad_physics)
        pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        fallen = not finite or d.qpos[2] < c.min_root_height or d.xmat[pelvis].reshape(3, 3)[2, 2] < c.min_upright
        self.failed |= fallen
        limited = np.flatnonzero(m.jnt_limited)
        joint_positions = d.qpos[m.jnt_qposadr[limited]]
        limits = m.jnt_range[limited]
        excess = np.maximum(limits[:, 0] - joint_positions, 0) + np.maximum(joint_positions - limits[:, 1], 0)
        undesired, support = 0, [False, False]
        for contact in d.contact[:d.ncon]:
            bodies = [int(m.geom_bodyid[contact.geom1]), int(m.geom_bodyid[contact.geom2])]
            if 0 not in bodies:
                continue
            body = max(bodies)
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body) or ''
            if 'ankle' in name or 'foot' in name:
                support[0 if 'left' in name else 1] = True
            else:
                undesired += 1
        reward = (-c.force_weight * force_squared / c.force_scale**2
                  -c.angular_weight * float(d.qvel[3:6] @ d.qvel[3:6])
                  -c.joint_limit_weight * float(excess @ excess)
                  -c.ground_contact_weight * undesired
                  -c.torque_weight * float(d.ctrl @ d.ctrl)
                  -c.acceleration_weight * float(np.square((d.qvel[6:] - previous_velocity[6:]) / m.opt.timestep).sum()))
        if not np.isfinite(reward):
            self.failed = True
            reward = -c.failure_penalty
        self.sub_rewards.append(reward)
        self.sub_force_squared.append(force_squared)
        self.sub_force_peak.append(float(np.sqrt(force_squared)))
        self.last_support = support
        self.last_qpos = d.qpos.copy()
        self.last_point = self.tether.point_state()[0]

    def reset(self, seed=None, trajectory=None, contact_index=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.contact_rng = np.random.default_rng(seed + 104729)
        self.failed = False
        self.sub_rewards, self.sub_force_squared, self.sub_force_peak = [], [], []
        # reset_all performs a physics step: disable old interaction during reset.
        self.sim.simulate_at_each_physics_step = self._original_step
        try:
            self.obs, _ = self.wrapped.reset(seed=seed, to_numpy=False, reset_to_default_pose=True)
        finally:
            self.sim.simulate_at_each_physics_step = self._physics_step
        self.contact_index, self.contact_name, contact_cfg = select_contact(self.cfg, self.contact_rng, contact_index)
        self.tether = Tether(self.sim.model, self.sim.data, contact_cfg)
        self.tether.reset(self.rng, trajectory)
        self.previous_z = self.standing.clone()
        self.previous_action = None
        self.history = np.zeros((self.cfg.force_history, 3), dtype=np.float32)
        self.steps = 0
        self.done = False
        self.initial_qpos = self.sim.data.qpos.copy()
        self.last_qpos = self.initial_qpos.copy()
        self.initial_point = self.tether.point_state()[0]
        self.last_point = self.initial_point.copy()
        self.last_support = [False, False]
        return self.observation()

    def observation(self):
        # Normalize exactly as the frozen BFM does; filter out privileged state.
        with torch.no_grad():
            normalized = self.model._normalize(self.obs)
            proprio = torch.cat([normalized[k].reshape(1, -1).float() for k in self.keys], -1)
        contact = np.concatenate((self.history.reshape(-1) / self.cfg.force_scale,
                                  [float(self.tether.active)], self.tether.cfg.local_point,
                                  [self.tether.body / self.sim.model.nbody]))
        if self.cfg.contact_points:
            one_hot = np.zeros(len(self.cfg.contact_points), dtype=np.float32)
            one_hot[self.contact_index] = 1.
            contact = np.concatenate((contact, one_hot))
        return torch.cat((proprio, torch.tensor(contact, dtype=torch.float32, device=self.device)[None],
                          self.previous_z / np.sqrt(self.standing.shape[-1])), -1)

    @torch.no_grad()
    def step(self, raw_action, baseline=False):
        if self.done:
            raise RuntimeError('Reset required after episode end')
        c = self.cfg
        z = self.standing if baseline else latent_command(self.model, self.standing, raw_action, np.linalg.norm(self.history[-1]), c)
        latent_delta = float((z - self.previous_z).square().mean())
        total, discount, rows = 0., 1., []
        for low_index in range(self.interval):
            action = self.model.act(self.obs, z, mean=True)
            if not torch.isfinite(action).all():
                raise FloatingPointError('Nonfinite frozen actor output')
            self.sub_rewards, self.sub_force_squared, self.sub_force_peak = [], [], []
            self.obs, _, terminated, truncated, _ = self.wrapped.step(action, to_numpy=False)
            self.steps += 1
            failure = self.failed or bool(torch.as_tensor(terminated).any())
            timeout = self.steps * self.dt >= c.episode_duration or bool(torch.as_tensor(truncated).any())
            self.done = failure or timeout
            rate = 0. if self.previous_action is None else float((action - self.previous_action).square().mean())
            change = latent_delta if low_index == 0 else 0.
            reward = float(np.mean(self.sub_rewards)) - c.action_rate_weight * rate - c.latent_rate_weight * change
            if failure:
                # Absorbing failure cost, never counted as successful force relief.
                remaining = max(0, int(np.ceil(c.episode_duration / self.dt)) - self.steps)
                absorbing = remaining if c.gamma == 1 else c.gamma * (1-c.gamma**remaining)/(1-c.gamma)
                reward -= c.failure_penalty + absorbing*c.force_weight*(c.max_force/c.force_scale)**2
            total += discount * reward
            discount *= c.gamma
            self.previous_action = action.clone()
            self.history = np.roll(self.history, -1, axis=0)
            self.history[-1] = self.tether.force
            rows.append(dict(time=self.steps*self.dt, force=float(np.linalg.norm(self.tether.force)),
                             force_squared=float(np.mean(self.sub_force_squared)),
                             peak_force=max(self.sub_force_peak), mean_force=float(np.mean(self.sub_force_peak)),
                             force_vector=self.tether.force.tolist(),
                             anchor=self.tether.anchor.tolist(), point=self.last_point.tolist(),
                             error=float(np.linalg.norm(self.tether.anchor-self.last_point)),
                             root_displacement=float(np.linalg.norm(self.last_qpos[:3]-self.initial_qpos[:3])),
                             joint_movement=float(np.linalg.norm(self.last_qpos[7:]-self.initial_qpos[7:])),
                             point_displacement=float(np.linalg.norm(self.last_point-self.initial_point)),
                             support=self.last_support, latent_variation=change,
                             reward=reward, failed=failure))
            if self.done:
                break
        self.previous_z = z.clone()
        # Fixed-horizon episodic objective: timeout is terminal, no cross-episode bootstrap.
        return self.observation(), total, self.done, dict(discount=discount, rows=rows, failed=failure, contact_index=self.contact_index, contact_name=self.contact_name)

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.sim.model, height=480, width=640)
        self.renderer.update_scene(self.sim.data)
        scene = self.renderer.scene
        def geom(kind, size, position, color):
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, kind, np.asarray(size), np.asarray(position), np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
            scene.ngeom += 1
            return g
        geom(mujoco.mjtGeom.mjGEOM_SPHERE, [.025]*3, self.tether.anchor, [1, .2, .2, 1])
        g = geom(mujoco.mjtGeom.mjGEOM_CAPSULE, [.005]*3, self.last_point, [.2, 1, .2, 1])
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, .005, self.last_point, self.tether.anchor)
        g = geom(mujoco.mjtGeom.mjGEOM_ARROW, [.01]*3, self.last_point, [1, .8, 0, 1])
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, .01, self.last_point, self.last_point + .01*self.tether.force)
        g.label = f'{np.linalg.norm(self.tether.force):.1f} N'
        return self.renderer.render().copy()

    def close(self):
        self.sim.simulate_at_each_physics_step = self._original_step
        if self.renderer is not None:
            self.renderer.close()
        self.wrapped.close()
