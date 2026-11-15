"""P0 static causal/interface/gradient/probability qualification on one actual train source.
Not physical task qualification, not formal or scratch optimizer training.
"""
import os,sys,argparse,json,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc

def study(e,base,c,out):
    import torch,numpy as np
    from native_env import install as contact_install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from dual_track_task import Task,global_rng,restore_global_rng
    from diagnostic_fb_resume import tree_digest
    from research.interaction_fb.ei_v2_sources import install
    from research.interaction_fb.ei_v2_agent import ContactGoalAgent,SIGMA
    from research.interaction_fb.schema import deploy,robot
    from locomotion_rollout import spherical
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    contact_install();install();speed(e);(out/'backend-prefix').mkdir();backend(e,base,out/'backend-prefix')
    parent=json.loads((STAGE/'locks/expanded-corrected-train.json').read_text())['parents'][0]
    task=Task(e,base,c,660301);task.reset(parent['specs'][0],record=True,label='P0-native-interface')
    obs=task.extended_observation();dev=e.standing.device
    obs.update(_previous_z=task.last_command.clone(),_target_z=task.last_command.clone(),_slot=torch.zeros(1,1,device=dev),_goal_y=torch.zeros(1,24,device=dev),_goal_mask=torch.zeros(1,24,device=dev),_goal_mode=torch.tensor([[1.,0.]],device=dev),_option_memory=torch.tensor([[1.,0.,0.,0.,0.]],device=dev))
    obs['_goal_mask'][:,0:3]=1;obs['_goal_mask'][:,6:9]=1;obs['_goal_mask'][:,21]=1;obs['_goal_y'][:,21]=1
    a=ContactGoalAgent(base,obs,'E',dev);b=ContactGoalAgent(base,obs,'I',dev)
    basebefore=tree_digest(base._actor.state_dict());normbefore=tree_digest(base._obs_normalizer.state_dict())
    common={}
    for name in ('tail','goal_encoder','film','selector','value'):
        common[name]=tree_digest(getattr(a,name).state_dict())==tree_digest(getattr(b,name).state_dict());assert common[name],name
    params={arm:sum(p.numel() for p in ag.modules.parameters() if p.requires_grad) for arm,ag in [('E',a),('I',b)]}
    relative=abs(params['E']-params['I'])/min(params.values());assert relative<=.01
    z=task.last_command.clone();option=torch.tensor([2],device=dev)
    with torch.no_grad():
        ref=base._actor(base._normalize(robot(obs)),z,base.cfg.actor_std).mean
        ma=a.mean(obs,z,option);mb=b.mean(obs,z,option)
        assert torch.equal(ref,ma) and torch.equal(ma,mb),'zero-FiLM initial function differs'
    hidden={k:v.clone() for k,v in obs.items()};hidden['privileged_state']+=100
    for lo,hi in [(9,15),(18,22)]:hidden['contact_privileged_v1'][:,lo:hi]+=100
    assert torch.equal(deploy(obs),deploy(hidden))
    with torch.no_grad():
        assert torch.equal(a.mean(obs,z,option),a.mean(hidden,z,option))
        for x,y in zip(a.context(obs,z,option),a.context(hidden,z,option)):assert torch.equal(x,y)
    # Initial zero-FiLM has zero input derivative but valid/nonzero output-layer learning gradient.
    changed={k:v.clone() for k,v in obs.items()};changed['contact_privileged_v1'][:,6]+=.2
    def output_gradient(o):
        a.modules.zero_grad(set_to_none=True)
        weights=torch.linspace(.1,1.,29,device=dev)
        with torch.enable_grad():
            (a.mean(o,z,option)*weights).sum().backward()
        g=a.film[-1].weight.grad.detach().clone();assert torch.isfinite(g).all() and g.abs().max()>0
        return g
    g=output_gradient(obs);gc=output_gradient(changed);assert not torch.equal(g,gc),'observed force absent from FiLM learning path'
    a.modules.zero_grad(set_to_none=True)
    rng=global_rng()
    with torch.no_grad():
        u,act,mu,lp=a.sample(obs,z,option,True);after=global_rng();restore_global_rng(rng)
        expected=mu+SIGMA*torch.randn_like(mu)
        assert torch.equal(expected,u) and torch.equal(act,u.clamp(-1,1))
        assert tree_digest(global_rng())==tree_digest(after)
        # Use the actual same causal row in 5 saved substep positions; this tests probability arithmetic only.
        ob={k:v.repeat(5,*([1]*(v.ndim-1))) for k,v in obs.items()};zz=z.repeat(5,1);op=option.repeat(5);uu=u.repeat(5,1)
        means=a.mean(ob,zz,op);new=a.logp(uu,means);old=lp.repeat(5)
        lp_error=float(abs(new.sum()-old.sum()));assert lp_error<=3e-4,('original strict joint logprob tolerance',lp_error)
        assert torch.isfinite(new).all()
    restore_global_rng(rng)
    assert basebefore==tree_digest(base._actor.state_dict()) and normbefore==tree_digest(base._obs_normalizer.state_dict())
    assert not a.opt.state and not b.opt.state and not a.vopt.state and not b.vopt.state
    result=dict(status='STATIC_INTERFACE_QUALIFIED_NO_PHYSICAL_TASK_PASS',common_initial_tensors=common,initial_original_action_exact=True,normalizer_unchanged=True,base_unchanged=True,param_counts=params,param_relative_difference=relative,tail_width=a.width,deploy_dim=a.ds,value_privileged_dim=a.dp,goal_fields=dict(value24=24,mask24=24,mode2=2,own_option_age5=5),hidden_input_invariance_exact=True,force_to_film_output_gradient_nonzero=True,force_gradient_changes=True,one_true_gaussian_draw_exact=True,no_relative_noise_clip=True,joint_five_substep_logprob_error=lp_error,strict_logprob_tolerance=3e-4,formal_updates=0,optimizer_steps=0,physical_task_safety='UNMEASURED',joint_categorical_sampling='UNMEASURED',GAE_bootstrap='UNMEASURED',new_process_recovery='UNMEASURED')
    (out/'interface.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from research.interaction_fb import benchmark
    benchmark.study=study;sys.argv=[__file__,'--output',str(a.output),'--arm','A'];benchmark.main()
if __name__=='__main__':main()
