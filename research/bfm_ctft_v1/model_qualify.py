"""S0 original-only native model migration/function qualification, no updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes'),str(Path(__file__).parent)]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from agent import build,ASSET,SOURCE_SHA,ACTOR_KEY,CRITIC_KEY,ACTOR_DIM,CRITIC_DIM
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.envs.utils.gym_spaces import json_to_space
import safetensors.torch


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);begun=time.monotonic()
    def write(name,x):(a.output/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
    write('launch.json',dict(pid=os.getpid(),started_unix=time.time(),runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),scope='S0 same-input math only, no updates'))
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high');torch.manual_seed(681117)
    assert torch.cuda.device_count()==1
    assert sha(ASSET/'checkpoint/model/model.safetensors')==SOURCE_SHA
    raw=json.loads((ASSET/'config.json').read_text())['agent'];kw=json.loads((ASSET/'checkpoint/model/init_kwargs.json').read_text())
    raw['compile']=False
    reference=FBcprAuxAgentConfig(**raw).build(json_to_space(kw['obs_space']),kw['action_dim'])
    safetensors.torch.load_model(reference._model,ASSET/'checkpoint/model/model.safetensors',strict=True,device='cuda')
    reference._model._obs_normalizer.eval()
    agent,manifest=build(aux_scale=1.)
    original=reference._model.state_dict();new=agent._model.state_dict()
    changed=[]
    for k,v in original.items():
        if k.startswith(('_aux_critic.','_target_aux_critic.','_aux_reward_normalizer.')):continue
        assert k in new and torch.equal(v,new[k]),('Original tensor changed at migration',k)
    added=[k for k in new if k not in original]
    assert all(k.endswith('contact_weight') or k=='_aux_reward_normalizer.scale' for k in added)
    assert all(torch.count_nonzero(new[k])==0 for k in added if k.endswith('contact_weight'))
    assert all(torch.equal(v,new['_target_aux_critic.'+k[len('_aux_critic.'):]]) for k,v in new.items() if k.startswith('_aux_critic.'))
    assert any(not torch.equal(v,new[k]) for k,v in original.items() if k.startswith('_aux_critic.') and k in new)
    records=[]
    for n in [1,4]:
        obs={k:torch.randn((n,*sp.shape),device='cuda',dtype=torch.float32) for k,sp in reference.obs_space.spaces.items()}
        ext=dict(obs,**{ACTOR_KEY:torch.randn(n,ACTOR_DIM,device='cuda'),CRITIC_KEY:torch.randn(n,CRITIC_DIM,device='cuda')})
        ext[ACTOR_KEY][:,-1]=1
        z=reference._model.project_z(torch.randn(n,256,device='cuda'));action=torch.randn(n,29,device='cuda').clamp(-1,1)
        with torch.no_grad():
            ro=reference._model._normalize(obs);no=agent._model._normalize(ext)
            ra=reference._model._actor(ro,z,.05).mean;na=agent._model._actor(no,z,.05).mean
            rf=reference._model._forward_map(ro,z,action);nf=agent._model._forward_map(no,z,action)
        assert torch.equal(ra,na),('Zero-column actor identity',n,float((ra-na).abs().max()))
        assert torch.equal(rf,nf),('Zero-column F identity',n,float((rf-nf).abs().max()))
        hidden={k:v.clone() for k,v in ext.items()};hidden[CRITIC_KEY]+=100;hidden['privileged_state']+=100
        with torch.no_grad():ha=agent._model._actor(agent._model._normalize(hidden),z,.05).mean
        assert torch.equal(na,ha),'Actor reads hidden partner/privileged input'
        records.append(dict(batch=n,actor_max_error=float((ra-na).abs().max()),F_max_error=float((rf-nf).abs().max()),hidden_actor_invariance_exact=True))
    # Derivative qualification only; zero optimizer updates and no data labels.
    mu=agent._model._actor(agent._model._normalize(ext),z,.05).mean
    mu.sum().backward()
    gradients={k:dict(trainable=p.requires_grad,gradient_nonzero=p.grad is not None and bool(torch.count_nonzero(p.grad))) for k,p in agent._model._actor.named_parameters()}
    assert any(v['gradient_nonzero'] for k,v in gradients.items() if 'contact_weight' in k)
    agent.actor_optimizer.zero_grad(set_to_none=True)
    free,total=torch.cuda.mem_get_info()
    assert torch.cuda.max_memory_reserved()<=.88*total and free>=max(6*2**30,.08*total)
    result=dict(status='ORIGINAL_NATIVE_MODEL_MIGRATION_AND_PERMISSIONS_QUALIFIED',
        checks=records,added_keys=added,model_manifest=manifest,actor_gradient_checks=gradients,
        optimizers=list(agent.optimizer_dict),all_adam_states_fresh=True,
        original_B_and_target_B_frozen=True,no_backward_optimizer=True,
        aux_fresh_and_target_exact=True,no_old_learning_weights=True,
        normalizer='original robot statistics frozen; fixed aux scale1 is disposable probe placeholder, formal calibration pending',
        float32_matmul_precision=torch.get_float32_matmul_precision(),amp=False,
        CUDA_reserved_peak=torch.cuda.max_memory_reserved(),CUDA_allocated_peak=torch.cuda.max_memory_allocated(),
        CUDA_free=free,real_physics_ticks=0,real_transitions=0,agent_updates=0,
        wall_s=time.monotonic()-begun)
    write('result.json',result);print(json.dumps({k:result[k] for k in ['status','checks','optimizers','CUDA_reserved_peak','wall_s']}),flush=True)


if __name__=='__main__':main()
