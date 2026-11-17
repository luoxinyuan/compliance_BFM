"""Causal CTFT observation fields; no partner truth enters the Actor fields.

History contains ten ACTUAL preceding 20ms samples, in the body frame at their
measurement time. Mount is known installation geometry, not active truth.
This layout is a preregistered implementation of contract1.3, not old97D input.
"""
from collections import deque
import numpy as np

ACTOR_LAYOUT = (
    ('applied_force_history_body_N_div10', 30),
    ('known_wrist_mount_body_m', 3),
    ('own_wrist_relative_root_body_m', 3),
    ('own_wrist_velocity_body_mps_div025', 3),
    ('own_root_velocity_body_mps_div025', 3),
    ('own_world_up_body', 3),
    ('public_guidance_mode', 1),
)
PRIVILEGED_LAYOUT = (
    ('current_partner_error_body_m_div020', 3),
    ('current_partner_velocity_body_mps_div025', 3),
    ('current_uncapped_force_body_N_div10', 3),
)
assert sum(n for _, n in ACTOR_LAYOUT) == 46
assert sum(n for _, n in PRIVILEGED_LAYOUT) == 9


class ForceHistory:
    def __init__(self, actual_prior_body_forces):
        a = np.asarray(actual_prior_body_forces, dtype=np.float64)
        assert a.shape == (10, 3) and np.isfinite(a).all()
        # Caller must supply real prefix samples. No synthetic zero padding.
        self.values = deque((v.copy() for v in a), maxlen=10)

    def append_actual(self, force_world, world_from_body):
        R = np.asarray(world_from_body, dtype=np.float64).reshape(3, 3)
        force = np.asarray(force_world, dtype=np.float64).reshape(3)
        assert np.isfinite(R).all() and np.isfinite(force).all()
        self.values.append(R.T @ force)

    def state(self):
        return np.array(self.values, dtype=np.float64).copy()


def actor_fields(*, force_history, known_mount, wrist_world, root_world,
                 wrist_velocity_world, root_velocity_world, world_from_body,
                 public_mode):
    assert public_mode in (0, 1)
    R = np.asarray(world_from_body, dtype=np.float64).reshape(3, 3)
    v = np.r_[force_history.state().reshape(-1) / 10.,
              np.asarray(known_mount).reshape(3),
              R.T @ (np.asarray(wrist_world) - np.asarray(root_world)),
              R.T @ np.asarray(wrist_velocity_world) / .25,
              R.T @ np.asarray(root_velocity_world) / .25,
              R.T @ np.array([0., 0., 1.]), float(public_mode)]
    assert v.shape == (46,) and np.isfinite(v).all()
    return v.astype(np.float32)


def critic_fields(*, current_partner_world, current_partner_velocity_world,
                  current_raw_force_world, wrist_world, world_from_body):
    R = np.asarray(world_from_body, dtype=np.float64).reshape(3, 3)
    v = np.r_[R.T @ (np.asarray(current_partner_world) - np.asarray(wrist_world)) / .20,
              R.T @ np.asarray(current_partner_velocity_world) / .25,
              R.T @ np.asarray(current_raw_force_world) / 10.]
    assert v.shape == (9,) and np.isfinite(v).all()
    return v.astype(np.float32)
