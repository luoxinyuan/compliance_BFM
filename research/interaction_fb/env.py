"""One isolated process owns one legacy Task and all its callback bindings.
No legacy learner or replay is imported as the new training engine.
"""
import copy
import numpy as np
import torch
from .schema import KEY
from dual_track_task import Task,global_rng,restore_global_rng,digest
from dual_track_native_data import NativeActor,cpu_obs
from native_env import spec

class TrialTask(Task):
    def sample_spec(self):raise RuntimeError('Explicit prelocked T0 specs only')

def specs():
    out=[]
    for i,(family,d,stop) in enumerate([('near',.08,None),('mid',.24,None),('stop',.24,5.6),('continue',.24,None),('zero',0,None),('none',0,None)]):
        s=spec(d,3.,4.6,'standing',4.,family!='none',stop,10.)
        s.update(demand_family=family)
        pid=digest(dict(protocol='Issue62-T0-v1',source='standing-4.0',request='pair' if family in ('stop','continue') else family))
        s.update(parent_id=pid,source_parent_id=digest({'standing':4.0}),partner_parent_id=pid,common_prefix_parent_id=pid)
        out.append(s)
    return out

class Proxy(NativeActor):
    def __init__(self,t,m):
        super().__init__(t,m,None,False);self.agent=None;self.slot=0;self.start=None;self.target=None;self.actions=[];self.replay_actions=None;self.stochastic=True
    def enrich(self,o):
        prev=self.task.last_command if self.start is None else self.start
        target=prev if self.target is None else self.target
        return dict(o,_previous_z=prev.detach().clone(),_target_z=target.detach().clone(),
                    _slot=torch.tensor([[self.slot]],device=prev.device))
    def finish_pending(self):
        if self.pending is not None:
            self.pending['next_observation']=cpu_obs(self.enrich(self.task.extended_observation()));self.pending=None
    def act(self,obs,z,mean=True):
        ext=self.enrich(self.task.extended_observation())
        if self.pending is not None:self.pending['next_observation']=cpu_obs(ext)
        if self.source_mode:
            a=self.original.act(obs,z,True);zi=torch.zeros(1,32,device=z.device)
        elif getattr(self,'reference_original',False):
            # Frozen external BC reference chooses once per 100ms in collect_call.
            # Task supplies its exact five-step interpolated z; do not requery
            # BC against a different, already-updated last-z history here.
            a=self.original.act(obs,z,True);zi=torch.zeros(1,32,device=z.device)
        elif self.replay_actions is not None:
            a=next(self.replay_actions).to(z.device);zi=self.agent.zi(1)
        else:
            a,calculated_z,zi=self.agent.action(ext,sample=self.stochastic)
            assert torch.equal(calculated_z,z),'actual and training interpolation differ'
        row=dict(observation=cpu_obs(ext),action=a.detach().cpu(),z=z.detach().cpu(),zi=zi.detach().cpu(),
                 previous_z=self.previous_z,target_z=(self.target if self.target is not None else z).detach().cpu(),
                 time=float(self.task.e.tether.time),source='standing')
        self.previous_z=z.detach().cpu();self.pending=row
        (self.source_pairs if self.source_mode else self.records).append(row)
        if not self.source_mode:self.actions.append(a.detach().cpu());self.slot+=1
        return a

class Environment:
    def __init__(self,e,m,contract,task_specs=None):
        self.specs=specs() if task_specs is None else task_specs
        self.task=TrialTask(e,m,contract,6201);self.proxy=Proxy(self.task,m);self.task.m=self.proxy
        self.physical_stream=[];self.rows=[];self.native=[];self.commands=[];self.spec_id=0;self.episodes=[];self.next_cursor=0
    def reset(self,i=0):
        self.spec_id=i;self.proxy.begin_source();self.proxy.slot=0;self.proxy.start=None;self.proxy.target=None
        self.task.reset(self.specs[i],record=True,label='T0-'+str(i));self.proxy.finish_pending()
        self.native.extend(self.proxy.source_pairs);self.proxy.begin_online();self.proxy.actions=[];self.commands=[]
    def collect_call(self):
        t=self.task;p=self.proxy
        if t.done:
            self.episodes.append(dict(spec=copy.deepcopy(t.spec),boundary=t.boundary,controls=len(t.controls)))
            if t.boundary:raise RuntimeError('HARD_PHYSICS_BOUNDARY:'+str(t.boundary))
            self.reset((self.spec_id+1)%len(self.specs))
        p.slot=0;p.start=t.last_command.clone();p.target=p.start.clone()
        ob=p.enrich(t.extended_observation())
        with torch.no_grad():
            from clean_policy import direct_command
            raw=p.agent.command(ob);p.target=direct_command(raw,t.e.standing)
        # Exactly five native controls. No phase/task/clock is a neural input.
        before=len(t.physics)
        rs,done,stats=p.take_step(raw,5)
        self.physical_stream.extend(copy.deepcopy(t.physics[before:]))
        for r,source in zip(rs,p.records):
            r['zi']=source['zi']
            r['provenance']['spec_index']=torch.tensor([[self.spec_id]],dtype=torch.int64)
            r['provenance']['online_episode']=torch.tensor([[len(self.episodes)]],dtype=torch.int64)
        self.rows.extend(rs);self.commands.append(raw.detach().cpu());self.next_cursor+=len(rs)
        # Next action at a high-level boundary must recompute Guidance.
        p.slot=0;p.start=t.last_command.clone();p.target=p.start.clone()
        self.rows[-1]['next']['observation']=cpu_obs(p.enrich(t.extended_observation()))
        if t.boundary:raise RuntimeError('HARD_PHYSICS_BOUNDARY:'+str(t.boundary))
        return stats
