"""Signed CTFT contact cost, native-aux units and separate physical integral.

Reward only; v_ref is never a command or actor observation. No task, future,
partner hidden state, policy, goal selector or old learned weights are imported.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ForceEMA:
    # Real causal20ms samples. Snapshot/restore includes this actual state.
    value: np.ndarray = field(default_factory=lambda: np.zeros(3,dtype=np.float64))
    tau_s: float = .05

    def update(self, applied_force_n, dt_s=.02):
        assert dt_s > 0 and self.tau_s == .05
        f=np.asarray(applied_force_n,dtype=np.float64)
        assert f.shape==(3,) and np.isfinite(f).all()
        # Exact exponential decay preserves the signed .05s time constant.
        self.value += (-np.expm1(-dt_s/self.tau_s))*(f-self.value)
        return self.value.copy()


def contact_cost(*, filtered_force_n, applied_force_n, wrist_velocity_mps,
                 root_velocity_mps, guidance_mode, dt_s=.02):
    """Physical world-frame horizontal response; one native step, no double dt."""
    f=np.asarray(filtered_force_n,dtype=np.float64)
    applied=np.asarray(applied_force_n,dtype=np.float64)
    vw=np.asarray(wrist_velocity_mps,dtype=np.float64)
    vr=np.asarray(root_velocity_mps,dtype=np.float64)
    assert all(a.shape==(3,) and np.isfinite(a).all() for a in [f,applied,vw,vr])
    assert guidance_mode in (0,1) and dt_s>0
    h=f[:2];fn=float(np.linalg.norm(h))
    vref=.025*max(0.,fn-.5)*h/(fn+1e-8)
    norm=float(np.linalg.norm(vref))
    if norm>.25:vref*=.25/norm
    idle=max(0.,1.-float(np.linalg.norm(f))/1.)**2
    parts=dict(response=2.*float(np.square((vw[:2]-vref)/.25).sum()),
        burden=.25*float(np.square(applied/10.).sum()),
        vertical=.1*float((vw[2]/.15)**2),
        idle=.5*idle*float(np.square(vr[:2]/.20).sum()))
    cost=sum(parts.values())
    # Approved amendment A1: original aux is per-step, contact converted likewise.
    # Keep separate physical-time integral for reporting and cross-duration use.
    return dict(parts=parts, guidance_mode=int(guidance_mode),
        velocity_reference_mps=vref.tolist(), idle_weight=idle,
        contact_native_reward=-guidance_mode*cost,
        contact_integrated_reward=-guidance_mode*dt_s*cost,
        dt_s=dt_s, unit_contract='A1_native_aux_per_step_and_separate_physical_integral')


def unit_checks():
    args=dict(filtered_force_n=[0,0,0],applied_force_n=[0,0,0],
              wrist_velocity_mps=[0,0,0],root_velocity_mps=[0,0,0],guidance_mode=1)
    quiet=contact_cost(**args)
    assert quiet['contact_native_reward']==quiet['contact_integrated_reward']==0
    hold=contact_cost(**dict(args,filtered_force_n=[10,0,0],applied_force_n=[10,0,0]))
    follow=contact_cost(**dict(args,filtered_force_n=[10,0,0],applied_force_n=[10,0,0],wrist_velocity_mps=[.2375,0,0]))
    assert abs(hold['velocity_reference_mps'][0]-.2375)<1e-8
    assert follow['contact_native_reward']>hold['contact_native_reward']
    assert all(abs(v['contact_native_reward']*.02-v['contact_integrated_reward'])<1e-12 for v in [quiet,hold,follow])
    native=contact_cost(**dict(args,guidance_mode=0,wrist_velocity_mps=[.2,0,0],root_velocity_mps=[.2,0,0]))
    assert native['contact_native_reward']==0 and native['contact_integrated_reward']==0
    f=ForceEMA();assert np.array_equal(f.value,np.zeros(3))
    expected=np.array([10.,0,0])*(-np.expm1(-.02/.05))
    assert np.array_equal(f.update([10,0,0]),expected)
    saved=f.value.copy();a=f.update([0,0,0]);f.value=saved.copy();b=f.update([0,0,0]);assert np.array_equal(a,b)
    return dict(zero_demand_zero_motion=quiet,ten_N_stationary=hold,ten_N_matched_velocity=follow,
                native_mode_contact_cost_zero=True,force_filter_replay_exact=True,
                boundary='arithmetic fixtures, not a physical success proof')
