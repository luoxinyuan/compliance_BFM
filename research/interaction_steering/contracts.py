"""Explicit information and execution contracts. No permissive feature dictionaries."""
from dataclasses import dataclass
from enum import Enum
import math

class Tier(str, Enum):
    CS = 'C-S'
    CP = 'C-P'
    OF = 'O-F'

@dataclass(frozen=True)
class CausalObservation:
    q: tuple
    dq: tuple
    gravity: tuple
    angular_velocity: tuple
    action_history: tuple
    proprio_history: tuple
    force_history: tuple
    last_z: tuple
    contact_link: str
    contact_local_point: tuple
    known_mode: str

    def __post_init__(self):
        for name, n in [('q',29),('dq',29),('gravity',3),('angular_velocity',3),('last_z',256),('contact_local_point',3)]:
            v=getattr(self,name)
            if len(v)!=n or not all(math.isfinite(float(x)) for x in v): raise ValueError(name)
        for name,n,m in [('action_history',4,29),('proprio_history',4,64),('force_history',25,3)]:
            rows=getattr(self,name)
            if len(rows)!=n or any(len(v)!=m or not all(math.isfinite(float(x)) for x in v) for v in rows): raise ValueError(name)
        if self.contact_link!='right_wrist_yaw_link' or self.known_mode!='forward_guidance': raise ValueError('S0 scope')

    @classmethod
    def from_mapping(cls, x):
        # Unexpected anchor, time, active, phase, future or task-ID fields fail closed.
        return cls(**x)

@dataclass(frozen=True)
class Guidance:
    vx: float
    vy: float
    yaw_rate: float
    valid: bool
    tier: Tier
    frame: str = 'yaw_body'
    def __post_init__(self):
        if self.frame!='yaw_body' or not all(math.isfinite(x) for x in (self.vx,self.vy,self.yaw_rate)): raise ValueError('guidance coordinate')
        if not isinstance(self.valid,bool) or not isinstance(self.tier,Tier): raise ValueError('explicit validity/tier required')

@dataclass(frozen=True)
class BackendContract:
    name: str
    source_commit: str
    weight_sha256: str
    joint_names: tuple
    observation_dimensions: tuple
    latent_dim: int
    latent_norm: float
    action_multiplier: float
    action_clip: float
    target_scale: tuple
    target_offset: tuple
    kp: tuple
    kd: tuple
    high_dt: float
    control_dt: float
    physics_dt: float
    simulator: str
    qualified_physics: bool
    root_command_native: bool
    def __post_init__(self):
        if len(self.joint_names)!=29 or len(set(self.joint_names))!=29: raise ValueError('joint order')
        for name in ('target_scale','target_offset','kp','kd'):
            if len(getattr(self,name))!=29: raise ValueError(name)
        if self.latent_dim!=256 or self.latent_norm!=16: raise ValueError('latent contract')
        if self.control_dt!=.02 or self.physics_dt!=.005 or self.high_dt!=.1: raise ValueError('cadence')
