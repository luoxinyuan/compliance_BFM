"""Yi-authorized P0 source-identity implementation correction (2026-11-15).
Only the new v2 arm-ready prefix changes; shared/old Task code stays untouched.
"""
import math
import numpy as np
from locomotion_rollout import spherical
SOURCE_VERSION='ei-v2-arm-parent-pulse-v1'

def arm_fraction(t,profile):
    # Same locked arm5 endpoint/ramp; existing per-parent preparation amplitude
    # and duration now enter the actual execution. Start is the existing 3.05s
    # source-preparation boundary, not a task/partner or force threshold.
    u=float(np.clip((float(t)-3.)/1.5,0.,1.));u=u*u*(3.-2.*u)
    v=(float(t)-3.05)/float(profile['pulse_duration'])
    pulse=float(profile['pulse_fraction'])*math.sin(math.pi*v)**2 if 0.<v<1. else 0.
    assert 0.<=pulse<1.
    return float(profile['arm_fraction'])*u*(1.-pulse)

def source_key(spec):
    import hashlib,json
    if spec['source']=='goal_arm5':
        p=spec['source_profile']
        effective=dict(version=SOURCE_VERSION,source='goal_arm5',until=spec['source_until'],
            arm_fraction=p['arm_fraction'],pulse_fraction=p['pulse_fraction'],pulse_duration=p['pulse_duration'])
    else:
        effective=dict(version='existing-standing-native-pulse',source=spec['source'],until=spec['source_until'],profile=spec['source_profile'])
    return hashlib.sha256(json.dumps(effective,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def install():
    from dual_track_task import Task
    if getattr(Task.native,'_ei_source_version',None)==SOURCE_VERSION:return
    original=Task.native
    def native(self,name,t):
        p=self.spec.get('source_profile')
        if name!='goal_arm5' or p is None:return original(self,name,t)
        return spherical(self.e.standing,self.native_z[name],arm_fraction(t,p))
    native._ei_source_version=SOURCE_VERSION;Task.native=native
