"""20 ms native-unit auxiliary reward for short wrist guidance."""
import numpy as np

def step_reward(native, mode, applied_force, physics_ticks, *, position=True):
    if mode not in (0, 1) or len(physics_ticks) != 4:
        raise ValueError("Expected a public mode and four executed physics ticks")
    burden = 0.
    tracking = 0.
    if mode == 1:
        force = np.asarray(applied_force, dtype=float)
        if force.shape != (3,) or not np.all(np.isfinite(force)):
            raise ValueError("Invalid actual force")
        burden = .25 * np.dot(force, force) / 100.
        if position:
            for tick in physics_ticks:
                if tick["active"]:
                    error = np.asarray(tick["wrist"]) - np.asarray(tick["anchor"])
                    tracking += np.dot(error, error) / (.10**2 * 4)
    return float(native - burden - tracking), float(burden), float(tracking)
