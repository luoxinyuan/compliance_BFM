"""Parent/branch provenance only; physics remains an isolated audited service."""
from dataclasses import dataclass

@dataclass(frozen=True)
class StateReference:
    parent_id:str
    state_id:str
    snapshot_sha256:str
    runtime_sha:str
    exposed:bool
    solver_contract:str
    includes_control_history:bool
    def __post_init__(self):
        if not self.exposed or len(self.snapshot_sha256)!=64 or not self.includes_control_history:raise ValueError('S0/S1 approved-state contract')

def require_shared_prefix(observations_a,observations_b,commands_a,commands_b):
    if observations_a!=observations_b or commands_a!=commands_b:raise ValueError('prefix diverged')
