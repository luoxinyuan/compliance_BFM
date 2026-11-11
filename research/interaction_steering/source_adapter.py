"""Explicit progress adapter hypothesis, NOT a qualified root-velocity controller.

No force/anchor/onset/task ID input. The latent bank is caller-owned and versioned.
S1 must test physical progress/stop; a stalled latent phase is not a stopped robot.
"""
from dataclasses import dataclass
import torch
from .contracts import Guidance
from .executor import direct_command

@dataclass
class ProgressMemory:
    phase: float = 0.

class ProgressToLatent:
    def __init__(self, bank, standing, period_s, nominal_speed_mps, source_sha):
        if bank.ndim!=2 or bank.shape[1]!=256 or len(bank)<2: raise ValueError('bank')
        if not torch.isfinite(bank).all() or period_s<=0 or nominal_speed_mps<=0: raise ValueError('source')
        if len(source_sha)!=64: raise ValueError('explicit source SHA required')
        self.bank=bank;self.standing=standing;self.period=period_s;self.speed=nominal_speed_mps;self.source_sha=source_sha
    def map(self, command: Guidance, memory: ProgressMemory, dt=.1):
        # This S0 prototype only exposes forward progress. Unsupported dimensions fail.
        if command.vy!=0 or command.yaw_rate!=0 or command.vx<0 or dt!=.1: raise ValueError('unqualified guidance dimension')
        if not command.valid: return self.standing.clone(),ProgressMemory(memory.phase)
        ratio=command.vx/self.speed
        phase=memory.phase+dt*ratio/self.period
        x=(phase%1)*len(self.bank);i=int(x);u=x-i
        z=(1-u)*self.bank[i:i+1]+u*self.bank[(i+1)%len(self.bank):(i+1)%len(self.bank)+1]
        alpha=min(1.,ratio)
        # Static, transparent experimental T; no claim to preserve exact goal semantics.
        z=direct_command((1-alpha)*self.standing+alpha*z,self.standing)
        return z,ProgressMemory(phase)
