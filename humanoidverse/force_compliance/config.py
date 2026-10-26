from dataclasses import dataclass, field


@dataclass
class ComplianceConfig:
    body: str = 'right_wrist_yaw_link'
    local_point: list[float] = field(default_factory=lambda: [0., 0., 0.])
    stiffness: float = 80.
    damping: float = 8.
    max_force: float = 60.
    displacement_min: float = .03
    displacement_max: float = .10
    direction: list[float] | None = None  # None: uniform sphere, fixed per episode
    trajectory: str = 'smooth'  # cubic smoothstep or linear ramp
    activation_time: float = 0.
    anchor_delay: float = 1.
    ramp_duration: float = 2.
    release_time: float | None = None  # maintained tether by default
    episode_duration: float = 6.
    initial_joint_positions: dict[str, float] = field(default_factory=dict)
    initial_root_height: float | None = None
    enabled: bool = True
    seed: int = 0
    high_level_hz: float = 10.
    force_scale: float = 20.
    gate_force: float = 5.
    residual_scale: float = .3
    force_history: int = 3
    force_weight: float = 1.
    angular_weight: float = .01
    joint_limit_weight: float = .1
    ground_contact_weight: float = 1.
    torque_weight: float = 1e-6
    acceleration_weight: float = 1e-7
    action_rate_weight: float = .01
    latent_rate_weight: float = .01
    failure_penalty: float = 1000.
    min_root_height: float = .4
    min_upright: float = .4
    settling_force: float = 5.
    settling_window: float = .5
    gamma: float = .99  # discount per LOW-level step
    gae_lambda: float = .95  # trace decay per HIGH-level decision

    def __post_init__(self):
        import math
        numeric = [v for v in vars(self).values() if isinstance(v, (float, int))]
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError('Configuration must be finite')
        if min(self.max_force, self.force_scale, self.gate_force, self.ramp_duration,
               self.episode_duration, self.high_level_hz, self.settling_window) <= 0:
            raise ValueError('Durations, frequencies and force scales must be positive')
        if self.stiffness < 0 or self.damping < 0 or not 0 <= self.displacement_min <= self.displacement_max:
            raise ValueError('Invalid spring parameters')
        if self.activation_time < 0 or self.anchor_delay < 0 or self.residual_scale < 0:
            raise ValueError('Activation time and residual scale must be nonnegative')
        if any(v < 0 for k, v in vars(self).items() if k.endswith('_weight')):
            raise ValueError('Penalty weights must be nonnegative')
        if not all(math.isfinite(v) for v in self.local_point):
            raise ValueError('Contact point must be finite')
        if not all(math.isfinite(v) for v in self.initial_joint_positions.values()):
            raise ValueError('Initial joint positions must be finite')
        if self.force_history < 1 or len(self.local_point) != 3:
            raise ValueError('Invalid history or point')
        if self.trajectory not in ('smooth', 'linear'):
            raise ValueError('Unknown trajectory')
        if self.release_time is not None and self.release_time < self.activation_time:
            raise ValueError('Release must follow activation')
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError('Invalid discount')
