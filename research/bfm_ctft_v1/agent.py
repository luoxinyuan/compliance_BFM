"""Native FB/CPR/aux CTFT agent, original weights only; no steering network.

Original Actor/F/all native critics are retained. B and robot normalization are
frozen; fresh auxiliary critic, all fresh optimizers. Contact is new zero input
columns, with the original input LayerNorm domain preserved exactly.
"""
import ast
import copy
import hashlib
import inspect
import json
import textwrap
from pathlib import Path
import types
import gymnasium as gym
import numpy as np
import safetensors.torch
import torch
import torch._functorch.config
# Yi-approved A2: retain the native compiled math and ordered FP32 gradients;
# prohibit donation of a backward tape that later loss components still need.
torch._functorch.config.donated_buffer = False
from torch import nn
from torch.nn import functional as F
from humanoidverse.agents.fb.agent import FBAgent
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgent,FBcprAuxAgentConfig
from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelConfig
from humanoidverse.agents.envs.utils.gym_spaces import json_to_space
from humanoidverse.agents.nn_filters import DictInputFilterConfig
from humanoidverse.agents.normalizers import IdentityNormalizerConfig
from humanoidverse.agents.nn_models import weight_init

P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
ASSET=Path('/mnt/data-oss/users/yi.gao/compliance_BFM/pretrained/official/new_model_for_training_code_inference')
SOURCE_SHA='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361'
ACTOR_KEY='ctft_actor46_v1';ACTOR_DIM=46
CRITIC_KEY='ctft_current_priv9_v1';CRITIC_DIM=9


def split(x,old_dim,new_dim):
    return torch.cat([x[...,:old_dim],x[...,old_dim+new_dim:]],-1),x[...,old_dim:old_dim+new_dim]


class AddedInputColumns(nn.Module):
    def __init__(self,original,old_dim,new_dim):
        super().__init__();self.mlp=original.mlp;self.old_dim=old_dim;self.new_dim=new_dim
        w=self.mlp[1].weight
        shape=(w.shape[0],new_dim,w.shape[-1]) if w.ndim==3 else (w.shape[0],new_dim)
        self.contact_weight=nn.Parameter(w.new_zeros(shape))

    def forward(self,x):
        old,extra=split(x,self.old_dim,self.new_dim)
        y=self.mlp[1](self.mlp[0](old))
        y=y+(torch.bmm(extra,self.contact_weight) if self.contact_weight.ndim==3 else F.linear(extra,self.contact_weight))
        for layer in self.mlp[2:]:y=layer(y)
        return y


class FixedAuxScale(nn.Module):
    def __init__(self,scale):
        super().__init__();assert np.isfinite(scale) and scale>0
        self.register_buffer('scale',torch.tensor(float(scale),dtype=torch.float32))
    def forward(self,x):return x/self.scale


def extend(model):
    old=model.obs_space
    space=gym.spaces.Dict(dict(old.spaces,**{
        ACTOR_KEY:gym.spaces.Box(-10.,10.,(ACTOR_DIM,),np.float32),
        CRITIC_KEY:gym.spaces.Box(-10.,10.,(CRITIC_DIM,),np.float32)}))
    records=[]
    for name in ['_actor','_forward_map','_target_forward_map','_critic','_target_critic','_aux_critic','_target_aux_critic']:
        module=getattr(model,name);old_dim=module.input_filter.output_space.shape[0]
        extras=[ACTOR_KEY] if name=='_actor' else [ACTOR_KEY,CRITIC_KEY]
        module.input_filter=DictInputFilterConfig(key=[*module.input_filter.cfg.key,*extras]).build(space)
        dim=ACTOR_DIM if name=='_actor' else ACTOR_DIM+CRITIC_DIM
        for emb in ['embed_s','embed_sa','embed_z']:
            if hasattr(module,emb):
                seq=getattr(module,emb);assert hasattr(seq[0],'mlp')
                seq[0]=AddedInputColumns(seq[0],old_dim,dim)
        records.append(dict(module=name,original_obs_dim=old_dim,extra_keys=extras,added_dim=dim))
    model._obs_normalizer._normalizers[ACTOR_KEY]=nn.Identity()
    model._obs_normalizer._normalizers[CRITIC_KEY]=nn.Identity()
    model._obs_normalizer.cfg=model._obs_normalizer.cfg.model_copy(update={'normalizers':{
        **model._obs_normalizer.cfg.normalizers,ACTOR_KEY:IdentityNormalizerConfig(),CRITIC_KEY:IdentityNormalizerConfig()}})
    model.obs_space=space
    return records


def transformed_native_methods():
    """Only explicit signed adaptations; original losses and full batch retained.

    Audit source is copied into the runtime and SHA recorded. This is not a
    hand-reimplemented approximation of the original pairwise FB objective.
    """
    fb=textwrap.dedent(inspect.getsource(FBAgent.update_fb))
    for line in ['    self.backward_optimizer.zero_grad(set_to_none=True)\n',
                 '        torch.nn.utils.clip_grad_norm_(self._model._backward_map.parameters(), clip_grad_norm)\n',
                 '    self.backward_optimizer.step()\n']:
        assert fb.count(line)==1;fb=fb.replace(line,'')
    up=textwrap.dedent(inspect.getsource(FBcprAuxAgent.update))
    old='        train_z = torch.where(mask, z, train_z)'
    # infer mode from public input, not phase or a latent future task answer.
    assert up.count(old)==1
    up=up.replace(old,'        mask = mask & (train_obs[ACTOR_KEY][:, -1:] == 0)\n'+old)
    old_b='''        _soft_update_params(
            self._backward_map_paramlist,
            self._target_backward_map_paramlist,
            self.cfg.train.fb_target_tau,
        )'''
    assert up.count(old_b)==1;up=up.replace(old_b,'        # B and target-B retain their distinct original checkpoint values.')
    old_actor='''    metrics.update(
        self.update_actor(
            obs=train_obs,
            action=train_action,
            z=train_z,
            clip_grad_norm=clip_grad_norm,
        )
    )'''
    assert up.count(old_actor)==1
    up=up.replace(old_actor,'    if self.actor_enabled:\n'+textwrap.indent(old_actor,'    '))
    ns=dict(FBAgent.update_fb.__globals__);ns.update(FBcprAuxAgent.update.__globals__);ns['ACTOR_KEY']=ACTOR_KEY
    exec(compile(fb+'\n'+up,'<locked-native-ctft-diff>','exec'),ns)
    return ns['update_fb'],ns['update'],dict(update_fb_source=fb,update_source=up,
        original_fb_sha256=hashlib.sha256(inspect.getsource(FBAgent.update_fb).encode()).hexdigest(),
        original_update_sha256=hashlib.sha256(inspect.getsource(FBcprAuxAgent.update).encode()).hexdigest())


class CTFTAgent(FBcprAuxAgent):
    @property
    def optimizer_dict(self):
        return {n:getattr(self,n).state_dict() for n in self.optimizer_names}


def build(*,device='cuda',aux_scale=1.,seed=681117,compile_execution=None):
    raw=json.loads((ASSET/'config.json').read_text())['agent']
    modelcfg=copy.deepcopy(raw['model']);modelcfg['device']=device
    train=copy.deepcopy(raw['train'])
    train.update(lr_actor=3e-5,lr_f=1e-4,lr_critic=1e-4,lr_aux_critic=1e-4,lr_discriminator=1e-4,
                 weight_decay=0.,weight_decay_discriminator=0.,clip_grad_norm=1.,batch_size=1024,
                 fb_target_tau=.01,critic_target_tau=.005)
    # Selected before formal learning after strict fresh-process qualification.
    # Preserve source compile=True in provenance, not as an implicit default
    # that can reintroduce the failed cache-dependent continuation path.
    execution_compile=False if compile_execution is None else bool(compile_execution)
    cfg=FBcprAuxAgentConfig(model=FBcprAuxModelConfig(**modelcfg),train=train,
        aux_rewards=['ctft_total_native'],aux_rewards_scaling={'ctft_total_native':1.},compile=False,cudagraphs=raw['cudagraphs'])
    torch.manual_seed(seed)
    init=json.loads((ASSET/'checkpoint/model/init_kwargs.json').read_text())
    agent=cfg.build(json_to_space(init['obs_space']),init['action_dim'])
    # setup_training has already applied weight_init. Load AFTER it, exactly once.
    safetensors.torch.load_model(agent._model,ASSET/'checkpoint/model/model.safetensors',strict=True,device=device)
    # Fully fresh aux topology; not only reinitializing the few nn.Linear leaves
    # of a pretrained ParallelLinear model. Its target is an exact fresh copy.
    new_aux=agent._model.cfg.archi.aux_critic.build(agent._model.obs_space,256,init['action_dim'],output_dim=1).to(device)
    new_aux.apply(weight_init);new_aux.requires_grad_(True)
    agent._model._aux_critic=new_aux
    agent._model._target_aux_critic=copy.deepcopy(new_aux)
    migration=extend(agent._model);agent.obs_space=agent._model.obs_space
    agent._model._aux_reward_normalizer=FixedAuxScale(aux_scale).to(device)
    for name in ['_backward_map','_target_backward_map','_obs_normalizer',
                 '_target_forward_map','_target_critic','_target_aux_critic']:
        getattr(agent._model,name).requires_grad_(False)
    agent._model._obs_normalizer.train(False)
    names={'actor':'_actor','forward':'_forward_map','critic':'_critic','aux_critic':'_aux_critic','discriminator':'_discriminator'}
    rates={'actor':3e-5,'forward':1e-4,'critic':1e-4,'aux_critic':1e-4,'discriminator':1e-4}
    for key,module in names.items():
        getattr(agent._model,module).requires_grad_(True)
        setattr(agent,key+'_optimizer',torch.optim.Adam(getattr(agent._model,module).parameters(),lr=rates[key],weight_decay=0.))
    # No empty/no-op backward optimizer survives in the actual learning agent.
    del agent.backward_optimizer
    agent.__class__=CTFTAgent
    agent.optimizer_names=[k+'_optimizer' for k in names]
    for a,b in [('_forward_map_paramlist','_forward_map'),('_target_forward_map_paramlist','_target_forward_map'),
                ('_critic_map_paramlist','_critic'),('_target_critic_map_paramlist','_target_critic'),
                ('_aux_critic_map_paramlist','_aux_critic'),('_aux_target_critic_map_paramlist','_target_aux_critic')]:
        setattr(agent,a,tuple(getattr(agent._model,b).parameters()))
    fb,up,diff=transformed_native_methods()
    agent.update_fb=types.MethodType(fb,agent);agent.update=types.MethodType(up,agent)
    agent.actor_enabled=False
    agent.cfg=agent.cfg.model_copy(update={'compile':execution_compile})
    # Pin original Adam steps to one execution path independent of compiler
    # cache history. All network methods retain native compile=True. The five
    # eager optimizer calls are exactly equivalent on actual saved gradients.
    from execution import eager_adam_boundaries
    agent.execution_contract=eager_adam_boundaries(agent)
    # Deterministic accumulation preserves all original discriminator/GP terms.
    from ordered_grad import install
    install(agent,record_gradients=False)
    if execution_compile:agent.setup_compile()
    assert all(not s['state'] for s in agent.optimizer_dict.values())
    assert all(not p.requires_grad for p in agent._model._backward_map.parameters())
    assert all(not p.requires_grad for p in agent._model._target_backward_map.parameters())
    assert agent.z_buffer.empty()
    return agent,dict(original_training_config=raw,actual_config=agent.cfg.model_dump(),
        migration=migration,source_math_diff=diff,aux_scale=aux_scale,
        actor_full_network=True,B_frozen=True,no_backward_optimizer=True,
        no_old_learning_weights=True,source_sha256=SOURCE_SHA,execution_compile=execution_compile,
        optimizer_execution=agent.execution_contract,
        aot_donated_buffer=torch._functorch.config.donated_buffer)
