"""Field permissions: no partner truth reaches Guidance or Actor."""
import torch
KEY='contact_privileged_v1'
ROBOT=('state','last_action','history_actor')
def deploy(obs):
    c=obs[KEY]
    return torch.cat([*(obs[k] for k in ROBOT), c[:,:9],c[:,15:18],c[:,22:],obs['_previous_z']/16],-1)
def privileged(obs):
    return torch.cat([deploy(obs),obs['privileged_state'],obs[KEY][:,9:15],obs[KEY][:,18:22]],-1)
def interaction(obs):
    # B_I successor result, not a reward scalar or actor input.
    return torch.cat([obs['state'],obs[KEY][:,6:22]],-1)
def robot(obs):return {k:v for k,v in obs.items() if k!=KEY and not k.startswith('_')}
def to_device(obs):return {k:v.cuda() for k,v in obs.items()}
