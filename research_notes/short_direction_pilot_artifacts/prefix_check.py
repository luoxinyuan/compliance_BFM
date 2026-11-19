"""Read aligned deterministic matched branch logs; exogenous schedules define the fork."""
import numpy as np

def check_fork(a, b, *, source_sha_a, source_sha_b, rng_sha_a, rng_sha_b):
    if source_sha_a != source_sha_b or rng_sha_a != rng_sha_b:
        raise ValueError("Unmatched physical source or stochastic stream")
    for i, (left, right) in enumerate(zip(a, b)):
        if left["force_time"] != right["force_time"]:
            raise ValueError(("time", i))
        if any(not np.array_equal(left[k],right[k]) for k in ("anchor","anchor_velocity")):
            if i == 0:
                raise ValueError("No common prefix")
            return i, left["force_time"]
        for k in ("qpos","qvel","force","active","actor_input_or_hash","executed_action_or_hash"):
            if not np.array_equal(left[k], right[k]):
                raise ValueError(("premature branch difference",i,k))
    raise ValueError("No observed schedule fork before the shorter trace ended")
