"""New entry only: privileged zero branch, explicit KL config, complete live RNG branches."""
import copy, dataclasses, random
import numpy as np
import torch
from clean_policy import Expert
from interactive_snapshot import CrossResetState

@dataclasses.dataclass(frozen=True)
class PPOContract:
    logprob_atol: float = 3e-4
    clip: float = .15
    joint_kl_limit: float = .02
    joint_kl_stop: float = .015
    joint_kl_assert: float = .02001
    retries: int = 7
    backtrack: float = .5
    def __post_init__(self):
        assert 0 < self.joint_kl_stop <= self.joint_kl_limit <= self.joint_kl_assert
        assert self.logprob_atol == 3e-4, 'Original probability assertion must not be loosened'

class PrivilegedBranch(torch.nn.Module):
    """Keep BC500's entire 1639D path and normalizer; extra current 1088D input is Teacher-only."""
    def __init__(self, checkpoint):
        super().__init__()
        self.base = Expert(1639, 'direct')
        self.base.load_state_dict(checkpoint['state_dict'], strict=True)
        self.branch = torch.nn.Sequential(torch.nn.Linear(1088,128),torch.nn.Tanh(),torch.nn.Linear(128,256))
        torch.nn.init.zeros_(self.branch[-1].weight); torch.nn.init.zeros_(self.branch[-1].bias)
        self.critic = torch.nn.Sequential(torch.nn.Linear(1639+1088,128),torch.nn.Tanh(),torch.nn.Linear(128,1))
        self.actor = torch.nn.ModuleList([self.base.actor,self.branch])
    @property
    def log_std(self): return self.base.log_std
    def distribution(self,x):
        s,p=x[:,:1639],x[:,1639:]
        return torch.distributions.Normal(self.base.distribution(s).mean+torch.cat([self.branch(row[None]) for row in p],0),self.log_std.exp())
    def value(self,x):return 10*self.critic(torch.cat([self.base.normalized(x[:,:1639]),x[:,1639:].clamp(-10,10)],-1)).squeeze(-1)

class RNGState:
    def __init__(self,task,generators=()):
        self.task=task; self.python=random.getstate();self.numpy=np.random.get_state()
        self.cpu=torch.get_rng_state();self.cuda=torch.cuda.get_rng_state_all()
        self.task_rng=copy.deepcopy(task.rng.bit_generator.state)
        self.generators=[(g,g.get_state().clone()) for g in generators]
    def restore(self):
        random.setstate(self.python);np.random.set_state(self.numpy);torch.set_rng_state(self.cpu);torch.cuda.set_rng_state_all(self.cuda)
        self.task.rng.bit_generator.state=copy.deepcopy(self.task_rng)
        for g,s in self.generators:g.set_state(s)

class TrainingBranch(CrossResetState):
    def __init__(self,task,generators=()):
        super().__init__(task);self.rng_all=RNGState(task,generators)
    def restore(self,check=True):
        super().restore(check);self.rng_all.restore()

def save_training_checkpoint(path,task,net,opt,generators=(),counters=None):
    """Serialize actual command prefix + all RNG streams; never pickle callback closures."""
    ss=TrainingBranch(task,generators)
    rng=ss.rng_all
    torch.save(dict(version=1,model=net.state_dict(),optimizer=opt.state_dict() if opt else None,
        counters=copy.deepcopy(counters or {}),spec=copy.deepcopy(task.spec),
        actual_raw_prefix=[copy.deepcopy(d['raw']) for d in task.decision_rows],
        physical_history_digest=ss.fingerprint,
        rng=dict(python=rng.python,numpy=rng.numpy,cpu=rng.cpu,cuda=rng.cuda,task=rng.task_rng,
                 generators=[state for _,state in rng.generators])),path)

def restore_training_checkpoint(path,task,net,opt,generators=()):
    """Reconstruct callback graph by native reset + actual prefix; require exact full history."""
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['version']==1 and len(generators)==len(ck['rng']['generators'])
    task.reset(ck['spec'],record=True,label='resumed')
    for raw in ck['actual_raw_prefix']:
        task.step(torch.tensor(raw,device=task.e.standing.device,dtype=torch.float32)[None])
    check=TrainingBranch(task,generators)
    assert check.fingerprint==ck['physical_history_digest'],'resume physics/history mismatch'
    net.load_state_dict(ck['model'],strict=True)
    if opt is not None:opt.load_state_dict(ck['optimizer'])
    r=ck['rng'];random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state_all(r['cuda'])
    task.rng.bit_generator.state=copy.deepcopy(r['task'])
    for g,state in zip(generators,r['generators']):g.set_state(state)
    return ck['counters']

def load_ppo_contract(path=None):
    """Versioned new entry configuration; never mutates archived runtime modules."""
    import json
    from pathlib import Path
    path=Path(path) if path else Path(__file__).with_name('dual_track_runtime_config_v1.json')
    cfg=json.loads(path.read_text())
    assert cfg['version']=='DUAL_TRACK_RUNTIME_V1'
    return PPOContract(**cfg['ppo'])
