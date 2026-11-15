"""Locked physical coordinates; old >1e-6N slip risk is never rethresholded."""
import math
import numpy as np

def quality(ph):
    assert ph
    yaw=np.unwrap([math.atan2(p['pelvis_matrix'][1][0],p['pelvis_matrix'][0][0]) for p in ph]);delta=np.degrees(yaw-yaw[0]);root=np.asarray([p['root'][:2] for p in ph]);slip=np.asarray([p['stage_loaded_slip_mps'] for p in ph])
    from f1_sequence_search import semantics
    has_post=all('f1_post_contacts' in p for p in ph)
    return dict(slip_peak=float(slip.max()),slip_p95=float(np.quantile(slip,.95)),slip_p90=float(np.quantile(slip,.9)),yaw_excursion_deg=float(abs(delta).max()),yaw_net_deg=float(delta[-1]),yaw_range_deg=float(delta.max()-delta.min()),yaw_TV_deg=float(abs(np.diff(delta)).sum()),root_path_m=float(np.linalg.norm(np.diff(root,axis=0),axis=1).sum()),unsupported_ticks=sum(not any(p['support']) for p in ph),max_joint_excess_rad=max(p['joint_limit_excess_max_rad'] for p in ph),old_loaded_threshold_N=1e-6,slip_semantics=semantics(ph) if has_post else 'UNMEASURED_NO_POST_VELOCITY',material_point_displacement='UNMEASURED; body-local/world points retained, speed integral is not material displacement')

def protection(q,domain,reference=None):
    bad=[]
    if q['max_joint_excess_rad']>.02:bad.append('hard_joint_excess')
    if q['unsupported_ticks']>0:bad.append('support_lost')
    if domain in ('none','zero'):
        for k,b in [('slip_peak',.10),('yaw_excursion_deg',15.),('root_path_m',.10)]:
            if q[k]>b:bad.append(domain+'_'+k)
    elif domain=='recovery':
        for k,b in [('slip_peak',.45),('yaw_excursion_deg',15.)]:
            if q[k]>b:bad.append('recovery_'+k)
        if reference is None:bad.append('UNMEASURED_no_same_state_reference')
        elif q['root_path_m']>reference['root_path_m']+.10:bad.append('recovery_extra_root_path')
    return bad

class EnvBranch:
    """Existing full Task/solver branch plus all NEW real controller memory and list state."""
    def __init__(self,env):
        import copy
        from dual_track_runtime import TrainingBranch
        from .ei_v2_checkpoint import OWN,LISTS
        self.env=env;self.branch=TrainingBranch(env.task)
        names=list(OWN)+list(LISTS)+['raw_target','reference_constant_option']
        self.own={k:copy.deepcopy(getattr(env,k)) for k in names if hasattr(env,k)};self.absent=[k for k in names if not hasattr(env,k)]
        self.proxy={k:copy.deepcopy(getattr(env.proxy,k)) for k in ('source_mode','source_pairs','records','pending','previous_z','slot','start','target','actions','stochastic')}
    def restore(self):
        import copy
        self.branch.restore()
        for k,v in self.own.items():setattr(self.env,k,copy.deepcopy(v))
        for k in self.absent:
            if hasattr(self.env,k):delattr(self.env,k)
        for k,v in self.proxy.items():setattr(self.env.proxy,k,copy.deepcopy(v))
