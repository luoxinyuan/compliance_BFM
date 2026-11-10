"""Issue58 standing-only causal paired curriculum, isolated from #56 runtime."""
import copy
import functools
import numpy as np
from dual_track_task import Task as OriginalTask, CFG, digest, qtime
from native_env import spec as native_spec

VERSION='F_RL_V2_STANDING_MID_STOP_V1'


def configure():
    CFG.update(seed=5801, source_probs={'standing':1.},
        request_probs={'near':.30,'mid':.30,'stop':.20,'zero':.10,'none':.10},
        eval_every=16,dev_parent_seed=58031,sealed_parent_seed=58061)


def sample(rng, family):
    until=qtime(rng.uniform(*CFG['source_until']['standing']))
    profile=dict(arm_fraction=float(rng.uniform(.85,1.)),pulse_fraction=float(rng.uniform(.02,.08)),
        pulse_duration=float(rng.uniform(.3,.8)),phase_offset=float(rng.uniform(-.1,.1)),
        phase_rate=float(rng.uniform(.9,1.1)))
    profile['pulse_duration']=min(profile['pulse_duration'],until-3.05-.3)
    onset=qtime(until+rng.uniform(*CFG['onset_delay']))
    request='mid' if family=='stop' else family
    if family in ('zero','none'):distance,ramp=0.,4.
    else:
        distance=float(rng.uniform(*CFG['distance_ranges'][request]))
        velocity=float(rng.uniform(*CFG['speed_ranges'][request]));ramp=distance/velocity
    stop=qtime(onset+rng.uniform(.35,.75)*ramp) if family=='stop' else None
    s=native_spec(distance,ramp,onset,'standing',until,family!='none',stop,10.)
    s['deceleration']=float(rng.uniform(*CFG['deceleration'])) if stop is not None else 0.
    end_move=stop+s['deceleration'] if stop is not None else onset+ramp
    s.update(release=end_move+float(rng.uniform(*CFG['hold'])),demand_family=family,source_profile=profile)
    s['end']=s['release']+CFG['recovery']
    source=digest(dict(source='standing',until=until,profile=profile))
    partner=digest({k:s.get(k) for k in ('distance','ramp','onset','stop_time','deceleration','release','end')})
    parent=digest(dict(version=VERSION,source=source,partner=partner))
    s.update(source_parent_id=source,partner_parent_id=partner,parent_id=parent,common_prefix_parent_id=parent)
    assert family in ('zero','none') or (CFG['distance_ranges'][request][0]<=distance<=CFG['distance_ranges'][request][1])
    return s


def continuation(s):
    assert s['demand_family']=='stop' and 'stop_time' in s
    c=copy.deepcopy(s);hold=s['release']-s['stop_time']-s['deceleration']
    del c['stop_time'];c['deceleration']=0.
    c['release']=s['onset']+s['ramp']+hold;c['end']=c['release']+CFG['recovery']
    c['demand_family']='mid';c['paired_role']='continue'
    return c


@functools.lru_cache(maxsize=1)
def manifests():
    result={}
    for split,seed,n in [('dev',58031,8),('sealed',58061,16)]:
        r=np.random.default_rng(seed);groups=[]
        for family in ['near']*n+['mid']*n+['stop']*n+['zero']*4+['none']*4:
            s=sample(r,family)
            group={k:s[k] for k in ('parent_id','source_parent_id','partner_parent_id','common_prefix_parent_id')}
            group.update(source='standing',family=family,spec=s)
            if family=='stop':group['continued_spec']=continuation(s)
            groups.append(group)
        result[split]=groups
    from dual_track_admission import assert_parent_separation
    assert_parent_separation(result)
    return result


class Task(OriginalTask):
    def sample_spec(self):
        if self.pending_continue_spec is not None:
            s=self.pending_continue_spec;self.pending_continue_spec=None
        else:
            family=str(self.rng.choice(['near','mid','stop','zero','none'],p=[.375,.125,.25,.125,.125]))
            s=sample(self.rng,family)
            if family=='stop':self.pending_continue_spec=continuation(s)
        for split in manifests().values():
            for g in split:
                assert all(s[k]!=g[k] for k in ('parent_id','source_parent_id','partner_parent_id'))
        return s

    def reset(self,*a,**kw):
        result=super().reset(*a,**kw)
        self.ever_contact=False
        return result

    def precontact_eligible(self):
        # Current/previous real physics only. Never reads future onset/family.
        return (not self.ever_contact and not self.e.tether.active and self.boundary is None
                and np.linalg.norm(self.e.tether.force)<1e-12)

    def step(self,*a,**kw):
        start=len(self.physics);result=super().step(*a,**kw)
        self.ever_contact=self.ever_contact or any(r['active'] for r in self.physics[start:])
        return result
