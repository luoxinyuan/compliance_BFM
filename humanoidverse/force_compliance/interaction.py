"""Maintained grasp/tether, not unilateral pushing contact. All vectors world-frame."""
import numpy as np
import mujoco


def spring_force(anchor, point, anchor_velocity, point_velocity, stiffness, damping, limit):
    raw = stiffness * (anchor - point) + damping * (anchor_velocity - point_velocity)
    return raw * min(1., limit / max(float(np.linalg.norm(raw)), 1e-12))


class Tether:
    def __init__(self, model, data, config):
        self.model, self.data, self.cfg = model, data, config
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.body)
        if self.body <= 0:
            raise ValueError(f'Unknown robot body: {config.body}')
        self.jac = np.zeros((3, model.nv))
        self.force = np.zeros(3)
        self.active = False
        self.reset(np.random.default_rng(config.seed))

    def point_state(self):
        mujoco.mj_forward(self.model, self.data)
        point = self.data.xpos[self.body] + self.data.xmat[self.body].reshape(3, 3) @ np.asarray(self.cfg.local_point)
        mujoco.mj_jac(self.model, self.data, self.jac, None, point, self.body)
        return point.copy(), self.jac @ self.data.qvel

    def reset(self, rng, trajectory=None):
        self.origin = self.point_state()[0]
        if trajectory is None:
            direction = rng.normal(size=3) if self.cfg.direction is None else np.asarray(self.cfg.direction, dtype=float)
            if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-9:
                raise ValueError('Direction must be a finite nonzero 3-vector')
            self.delta = direction / np.linalg.norm(direction) * rng.uniform(self.cfg.displacement_min, self.cfg.displacement_max)
        else:
            # Absolute anchors copied across baseline and policy, never re-anchored mid-episode.
            self.origin, self.delta = (np.array(v, dtype=float).copy() for v in trajectory)
        self.time = 0.
        self.force[:] = 0
        self.anchor = self.origin.copy()
        self.active = False

    def measure(self):
        c = self.cfg
        start = c.activation_time + c.anchor_delay
        u = np.clip((self.time - start) / c.ramp_duration, 0, 1)
        moving = start < self.time < start + c.ramp_duration
        s, ds = (3*u*u - 2*u**3, 6*u*(1-u)) if c.trajectory == 'smooth' else (u, 1.)
        self.anchor = self.origin + s * self.delta
        velocity = self.delta * ds / c.ramp_duration if moving else np.zeros(3)
        self.active = c.enabled and self.time >= c.activation_time and (c.release_time is None or self.time < c.release_time)
        point, point_velocity = self.point_state()
        self.force = spring_force(self.anchor, point, velocity, point_velocity, c.stiffness, c.damping, c.max_force) if self.active else np.zeros(3)
        return point, self.force.copy()

    def apply(self):
        point, force = self.measure()
        wrench = np.concatenate((force, np.cross(point - self.data.xipos[self.body], force)))
        self.data.xfrc_applied[self.body] += wrench
        return wrench
