"""F-RL-v2 B environment; only a specified executed-step risk cost is added."""
import copy,json
from pathlib import Path
from frl_v2_task import Task as Original,configure as configure_v2,manifests as old_manifests
from frl_v3_risk import event
STAGE=Path(__file__).resolve().parents[2]
RISK_COEFFICIENT=.33
_CURRENT_MANIFEST=None

def configure(coefficient=.33,regression=False):
    global RISK_COEFFICIENT,_CURRENT_MANIFEST
    assert coefficient in (0.,.33)
    configure_v2();RISK_COEFFICIENT=coefficient
    if regression:
        _CURRENT_MANIFEST=dict(dev=old_manifests()['dev'],sealed=[])
    else:_CURRENT_MANIFEST=json.loads((STAGE/'prep/new-development-manifest.json').read_text())
    import frl_v2_task
    frl_v2_task.manifests=lambda:_CURRENT_MANIFEST
    return _CURRENT_MANIFEST

def manifests():
    return _CURRENT_MANIFEST if _CURRENT_MANIFEST is not None else json.loads((STAGE/'prep/new-development-manifest.json').read_text())

class Task(Original):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        from frl_v3_counterfactual import install_contact_details
        install_contact_details(self.e)
    def step(self,*a,**kw):
        n=len(self.controls);p=len(self.physics);result=super().step(*a,**kw)
        # Lambda0 executes the exact old return path, not an approximately-zero loss.
        if RISK_COEFFICIENT==0.:return result
        x,reward,done,st=result;cs=self.controls[n:];rows=self.physics[p:];events=[]
        for i,c in enumerate(cs):
            end=cs[i+1]['time'] if i+1<len(cs) else float(self.e.tether.time)+1e-8
            rr=[r for r in rows if c['time']-1e-8<=r['force_time']<end-1e-8]
            cost=event(rr,RISK_COEFFICIENT);c.update(legacy_optimized_reward_integral=c['reward_integral'],slip_risk_event_cost=cost,slip_risk_rate_equivalent=cost/c['actual_duration_s'],slip_risk_max_loaded_mps=max(r['stage_loaded_slip_mps'] for r in rr));events.append(cost)
            if cost!=0.:c['reward_integral']-=cost
        penalty=sum(events)
        st['native_risk_event_costs']=events;st['risk_event_sum']=penalty;st['risk_coefficient']=RISK_COEFFICIENT
        st['reward_source']='original native reward minus one unscaled event risk cost per actual20ms step; native_terms remain legacy rates'
        if penalty!=0.:
            st['legacy_optimized_native_rewards']=st['native_rewards'][:]
            st['native_rewards']=[v-p for v,p in zip(st['native_rewards'],events)]
            reward-=penalty;self.episode_return-=penalty;st['reward']=reward
        self.stats[-1]=st
        return x,reward,done,st
