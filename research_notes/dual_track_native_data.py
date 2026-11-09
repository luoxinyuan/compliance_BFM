"""Actual native-action collection through the unchanged original PD/Tether.

An explicit proxy selects the separate mutable actor only for controlled online
execution. Real native source prefixes still use the original checkpoint. No
search/Teacher override is ever labeled on-policy PPO data.
"""
import copy
import torch
from dual_track_fb_extension import CONTACT_KEY
from dual_track_admission import validate_transition


def cpu_obs(obs):
    return {k:v.detach().cpu().clone() for k,v in obs.items()}


class NativeActor:
    def __init__(self, task, original, policy_model=None, stochastic=False):
        self.task, self.original, self.policy_model = task, original, policy_model
        self.stochastic = stochastic
        self.source_mode = True
        self.source_pairs = []
        self.records = []
        self.pending = None
        self.previous_z = task.e.standing.detach().cpu().clone()

    def __getattr__(self, name):
        return getattr(self.original, name)

    def begin_source(self):
        self.source_mode=True; self.source_pairs=[]; self.records=[]; self.pending=None
        self.previous_z=self.task.e.standing.detach().cpu().clone()

    def begin_online(self):
        self.finish_pending()
        self.source_mode=False; self.records=[]; self.pending=None
        self.previous_z=self.task.last_command.detach().cpu().clone()

    def finish_pending(self):
        if self.pending is not None:
            self.pending['next_observation']=cpu_obs(self.task.extended_observation())
            self.pending=None

    def act(self, obs, z, mean=True):
        t=self.task
        extended=t.extended_observation()
        # Verify the actual BFM input used by the executor, not a similarly named
        # privileged vector or a reconstruction from another controller's state.
        assert all(torch.equal(extended[k],v) for k,v in obs.items())
        if self.pending is not None:self.pending['next_observation']=cpu_obs(extended)
        if self.source_mode or self.policy_model is None:
            action=self.original.act(obs,z,mean=True)
        elif self.stochastic:
            action=self.policy_model.actor(extended,z,self.policy_model.cfg.actor_std).sample(clip=.3).float()
        else:action=self.policy_model.act(extended,z,mean=True)
        from clean_policy import direct_command
        row=dict(observation=cpu_obs(extended),action=action.detach().cpu().clone(),
                 z=z.detach().cpu().clone(),previous_z=self.previous_z,
                 target_z=direct_command(t.raw,t.e.standing).detach().cpu().clone(),
                 time=float(t.e.tether.time),source=t.spec['source'])
        self.previous_z=row['z'];self.pending=row
        (self.source_pairs if self.source_mode else self.records).append(row)
        return action

    def take_step(self, raw, native_limit=5):
        assert not self.source_mode
        self.records=[]; self.pending=None
        self.previous_z=self.task.last_command.detach().cpu().clone()
        _,reward,done,stats=self.task.step(raw,native_limit=native_limit)
        self.finish_pending()
        assert len(self.records)==stats['control_steps']==len(stats['native_rewards'])
        result=[]
        for i,row in enumerate(self.records):
            last=i==len(self.records)-1
            term=bool(last and stats['terminated']); trunc=bool(last and stats['truncated'])
            validate_transition(observation=row['observation'],next_observation=row['next_observation'],
                action=row['action'],z=row['z'],target_z=row['target_z'],
                terminal_observation=row['next_observation'] if term or trunc else None,
                terminated=term,truncated=trunc,episode_id=self.task.spec['parent_id'],
                successor_episode_id=self.task.spec['parent_id'],origin='actual_online_execution')
            c=self.task.controls[-len(self.records)+i]
            assert abs(c['time']-row['time'])<1e-7
            assert torch.equal(row['action'][0],torch.tensor(c['action'],dtype=torch.float32))
            assert torch.equal(row['z'],torch.tensor(c['executed_z'],dtype=torch.float32))
            result.append(dict(observation=row['observation'],action=row['action'],z=row['z'],
                previous_z=row['previous_z'],target_z=row['target_z'],
                provenance=dict(time=torch.tensor([[row['time']]],dtype=torch.float64),
                    parent_sha256=torch.tensor([list(bytes.fromhex(self.task.spec['parent_id']))],dtype=torch.uint8),
                    source_parent_sha256=torch.tensor([list(bytes.fromhex(self.task.spec['source_parent_id']))],dtype=torch.uint8),
                    online_episode=torch.tensor([[getattr(self.task,'online_episode',0)]],dtype=torch.int64),
                    origin_actual_online_execution=torch.ones(1,1,dtype=torch.uint8)),
                next=dict(observation=row['next_observation'],terminated=torch.tensor([[term]]),
                          truncated=torch.tensor([[trunc]])),
                aux_rewards={'contact_physical':torch.tensor([[stats['native_rewards'][i]]],dtype=torch.float32)},
                duration=torch.tensor([[c['actual_duration_s']]],dtype=torch.float32)))
        assert abs(sum(float(r['aux_rewards']['contact_physical']) for r in result)-reward)<1e-4
        return result,done,stats


class NativeSequences:
    """Source-balanced native no-contact8-frame snippets, no reset crossings."""
    def __init__(self, max_parents_per_source=200):
        self.parents={k:[] for k in ('standing','goal_arm5','motion_walk_task')}
        self.max_parents=max_parents_per_source
        self.seen={k:0 for k in self.parents}

    def add(self, source, parent_id, rows):
        assert len(rows)>=8 and all('next_observation' in r for r in rows)
        assert all(abs(b['time']-a['time']-.02)<1e-6 for a,b in zip(rows,rows[1:]))
        assert all(not bool(r['observation'][CONTACT_KEY][0,21]) for r in rows)
        starts=torch.linspace(0,len(rows)-8,steps=min(16,len(rows)-7)).long().unique().tolist()
        blocks=[]
        for start in starts:
            part=rows[start:start+8]
            blocks.append(dict(observation={k:torch.cat([r['observation'][k] for r in part]) for k in part[0]['observation']},
                next=dict(observation={k:torch.cat([r['next_observation'][k] for r in part]) for k in part[0]['observation']})))
        self.seen[source]+=1; group=dict(parent_id=parent_id,blocks=blocks)
        if len(self.parents[source])<self.max_parents:self.parents[source].append(group)
        else:
            index=int(torch.randint(self.seen[source],()).item())
            if index<self.max_parents:self.parents[source][index]=group

    def sample(self, batch_size):
        assert batch_size%8==0
        available=[k for k,v in self.parents.items() if v];assert available
        blocks=[]
        for _ in range(batch_size//8):
            source=available[int(torch.randint(len(available),()).item())]
            parents=self.parents[source];group=parents[int(torch.randint(len(parents),()).item())]
            blocks.append(group['blocks'][int(torch.randint(len(group['blocks']),()).item())])
        return dict(observation={k:torch.cat([b['observation'][k] for b in blocks]) for k in blocks[0]['observation']},
            next=dict(observation={k:torch.cat([b['next']['observation'][k] for b in blocks]) for k in blocks[0]['observation']}))
