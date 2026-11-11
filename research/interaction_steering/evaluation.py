"""Evaluation identity boundary. Actual metrics reuse qualified immutable worker."""
from dataclasses import dataclass
from .contracts import Tier

@dataclass(frozen=True)
class EvaluationIdentity:
    tier:Tier
    kind:str
    parent_id:str
    runtime_sha:str
    def __post_init__(self):
        if self.kind not in ('actual_execution','predicted_branch','archived_replay'):raise ValueError('identity')
    def permits_on_policy_use(self):
        # S0 creates no training samples, including from stored actual trajectories.
        return False
