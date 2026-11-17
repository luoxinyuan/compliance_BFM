"""Prelocked exogenous curriculum and common phase-stratified replay.

All phase labels stay OUTSIDE observations; they only control replay sampling
and accounting. Eager original FB losses still receive1024 real transitions.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils._pytree import tree_map
from humanoidverse.agents.buffers.transition import DictBuffer
from source_reset_manifest import sample


def make_plan(reset_lock,cycles=512,seed=681117):
    rng=np.random.default_rng(seed);episodes=[];partners=[]
    for cycle in range(cycles):
        block=[]
        for p in range(3):
            u=rng.random(4).tolist()
            partner=dict(distance_m=float(rng.uniform(.60,.90)),plateau_speed_mps=float(rng.choice([.08,.12,.16])),
                accel_s=.5,hold_s=1.5,stiffness=float(rng.choice([40.,80.])),direction_deg=float(rng.uniform(-15.,15.)))
            stop=copy.deepcopy(partner);stop['distance_m']=float(rng.uniform(.30,.45))
            parent=f'train-family-{cycle:04d}-{p}'
            # Three independent exogenous families; siblings are not independent
            # parents. Actual policy versions/RNG exposures are recorded later.
            for branch,force in [('continue',True),('stop',True),('continue',True),('no_force',False),('stop',True)]:
                block.append(dict(mode=1,force=force,parent=parent,branch=branch,source_random=u,
                    partner=copy.deepcopy(stop if branch=='stop' else partner)))
            partners.append(partner)
        for p in range(5):block.append(dict(mode=0,force=False,parent=f'native-family-{cycle:04d}-{p}',
            branch='native_continuation',source_random=rng.random(4).tolist()))
        for i in rng.permutation(len(block)):episodes.append(block[int(i)])
    moving=sum(p['distance_m']/p['plateau_speed_mps']+p['accel_s'] for p in partners)
    # Use all prelocked actual partner schedules, including stop siblings, for
    # nominal-duration weighting; never adapt shares to learned success.
    g1=[p for p in episodes if p['mode']==1]
    durations=np.array([[p['partner']['distance_m']/p['partner']['plateau_speed_mps']+.5,1.5,4.] for p in g1]).sum(0)
    quotas={0:.25,**{i+1:float(.75*x/durations.sum()) for i,x in enumerate(durations)}}
    return dict(protocol='CTFT_EXOGENOUS_CURRICULUM_V1',seed=seed,episodes=episodes,replay_quotas=quotas,
        episode_share_g1=.75,g1_no_force_episode_share=.20,actor_phase_input=False,
        initial_contact='At real source handoff, anchor initialized at actual wrist; continuous smooth trajectory. No mid-episode reanchor.',
        g0_contract='Phase-compatible original native motion continuation. Original .2/.6/.2 mixed-z and .8 relabel remain in native update; no g1 relabel.',
        g0_duration='At most250 real controls, bounded by recorded safe native continuation. All actual row counts separately reported.',
        phase_quota_basis='g0 rowquota.25;g1 rowquota.75;g1 phase shares from nominal exogenous moving/holding/released durations, fixed before learner results.',
        independent_groups_are_source_ancestors_not_cycle_IDs=True)


def episode(plan,reset_lock,index,arm):
    spec=copy.deepcopy(plan['episodes'][index]);spec['source']=sample(reset_lock,arm,spec.pop('source_random'))
    return spec


class PhaseReplay(DictBuffer):
    def __init__(self,*,capacity,quotas):
        super().__init__(capacity=capacity,device='cpu');self.quotas={int(k):float(v) for k,v in quotas.items()}
        assert abs(sum(self.quotas.values())-1)<1e-12;self.members={k:[] for k in self.quotas}

    def extend(self,data):
        if not data:return
        phases=data['ctft_replay_phase'].reshape(-1).cpu().tolist();before=self._idx
        assert not self._is_full and before+len(phases)<self.capacity,'No replay wrap permitted in this finite stage'
        super().extend(data)
        for i,phase in enumerate(phases):self.members[int(phase)].append(before+i)

    def sample(self,n):
        available=[k for k,v in self.members.items() if v]
        assert available
        weights=np.array([self.quotas[k] for k in available]);weights/=weights.sum()
        counts=np.floor(weights*n).astype(int)
        for i in np.argsort(-(weights*n-counts),kind='stable')[:n-counts.sum()]:counts[i]+=1
        indices=[]
        for k,count in zip(available,counts):
            members=torch.as_tensor(self.members[k],dtype=torch.long)
            indices.append(members[torch.randint(len(members),(int(count),))])
        selected=torch.cat(indices);selected=selected[torch.randperm(n)]
        return tree_map(lambda x:x[selected],self.storage)
