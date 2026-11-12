"""Locked frozen branches. No learning, no task/RNG/history substitution."""
import json,copy,hashlib
from pathlib import Path
import torch,numpy as np
from .agent import Agent
from .envelope import Envelope
from .journal import load_state
from .schema import robot,deploy
from diagnostic_fb_resume import tree_digest
from dual_track_task import global_rng,restore_global_rng
from diagnostic_frozen import portable
from f1_mjdata_contract import metadata,canonical

def run(env,agent,out,config,case,mode,stochastic):
    model=load_state(Path(config['model_checkpoint']))['agent']
    rng=global_rng();e=env.task.e;contract=env.task.c
    if agent.arm!='B':
        ob=env.proxy.enrich(env.task.extended_observation())
        agent=Agent(env.proxy.original,ob,e.standing,torch.tensor(contract['q_PD_scale'],device='cuda'),'B')
    agent.load(model);restore_global_rng(rng);env.proxy.agent=agent
    model_digest=tree_digest(agent.state());initial=portable(env.task)
    initial_digest=tree_digest(canonical(initial,metadata(e.sim.model,e.sim.data)))
    prefix_digest=tree_digest(dict(controls=env.task.controls,physics=env.task.physics,commands=env.commands,actions=env.proxy.actions))
    assert abs(e.tether.time-case['time'])<1e-8 and env.task.spec['parent_id']==case['parent_id']
    assert e.tether.time+3<=env.task.spec['end']+1e-8
    if mode=='envelope':agent.envelope=Envelope(config['operator'],agent.goal.device)
    elif mode=='off':
        original=agent.action
        def off(obs,target=False,sample=False,z_override=None):
            _,z,zi=original(obs,target,False,z_override)
            a=agent.base._actor(agent.base._normalize(robot(obs)),z,agent.base.cfg.actor_std).mean.clamp(-1,1)
            if sample:a=(a+(agent.exploration_std*torch.randn_like(a)).clamp(-agent.exploration_clip,agent.exploration_clip)).clamp(-1,1)
            return a,z,zi
        agent.action=off
    else:assert mode=='on'
    logs=[];original=agent.action
    def observed(obs,target=False,sample=False,z_override=None):
        a,z,zi=original(obs,target,sample,z_override)
        base=agent.base._actor(agent.base._normalize(robot(obs)),z,agent.base.cfg.actor_std).mean
        if agent.envelope is not None:
            au={k:v.detach().cpu() for k,v in agent.last_action_audit.items()}
            assert torch.equal(a.cpu(),au['executed'])
        else:
            adapter=agent.nets['adapter'](torch.cat([deploy(obs),z,zi],-1));au=dict(base=base.detach().cpu(),adapter=adapter.detach().cpu(),executed=a.detach().cpu())
        logs.append(dict(time=float(e.tether.time),observation={k:v.detach().cpu() for k,v in obs.items()},z=z.detach().cpu(),audit=au))
        return a,z,zi
    agent.action=observed;env.proxy.stochastic=stochastic
    begin=len(env.physical_stream);rowstart=len(env.rows);ctrlstart=len(env.task.controls);commandstart=len(env.commands)
    status='completed_3s';error=None
    try:
        with torch.no_grad():
            for _ in range(30):env.collect_call()
    except RuntimeError as exc:
        if not str(exc).startswith('HARD_PHYSICS_BOUNDARY:'):raise
        status='hard_physics_boundary';error=str(exc)
    frozen=agent.state();frozen.pop('envelope',None);assert tree_digest(frozen)==model_digest
    trace=dict(physics=env.physical_stream[begin:],controls=env.task.controls[ctrlstart:],rows=env.rows[rowstart:],commands=env.commands[commandstart:],actions=logs,task_spec=env.task.spec,initial=initial,rng=rng)
    # Replay action, actual control action and original q_PD reconstruction are exact.
    assert len(trace['rows'])==len(trace['controls'])==len(logs), '20ms action/replay/control row counts differ'
    for row,ctrl,log in zip(trace['rows'],trace['controls'],logs):
        executed=log['audit']['executed'];assert torch.equal(row['action'],executed)
        assert torch.equal(executed[0],torch.tensor(ctrl['action']))
        q=env.task.qof(executed.cuda()).cpu()[0]
        assert torch.equal(q,torch.tensor(ctrl['q_PD']))
        if mode=='envelope':assert torch.equal(row['envelope']['qexec'][0],q)
    torch.save(trace,out/'qualification-trace.pt')
    result=dict(status=status,error=error,mode=mode,sample=stochastic,case=case,initial_digest=initial_digest,prefix_digest=prefix_digest,rng_digest=tree_digest(rng),model_digest=model_digest,trace_sha256=hashlib.sha256((out/'qualification-trace.pt').read_bytes()).hexdigest(),ticks=len(trace['physics']),controls=len(trace['controls']),updates=agent.updates,exact_execution_replay_qPD=True)
    if mode=='envelope' and case['id']=='B19-release' and stochastic and status=='completed_3s':
        result['path_audit']=audit_update_paths(agent,trace['rows'],env.native,config['operator'])
    (out/'qualification.json').write_text(json.dumps(result,indent=2));return result

def audit_update_paths(agent,rows,native,operator):
    """One disposable engineering update on actual new transitions; never adopts weights."""
    from .schema import privileged,to_device
    from dual_track_fb_online import stack
    rng=global_rng();original_digest=tree_digest(agent.state())
    obs=to_device(rows[0]['observation'])
    test=Agent(agent.base,obs,agent.goal,agent.scale,agent.arm);test.load(agent.state());test.envelope=Envelope(operator,agent.goal.device)
    captures=[];original=test.action
    def observed(o,target=False,sample=False,z_override=None):
        result=original(o,target,sample,z_override);audit={k:v.detach().cpu().clone() for k,v in test.last_action_audit.items()}
        captures.append(dict(target=target,sample=sample,audit=audit,action=result[0].detach().cpu().clone()))
        return result
    test.action=observed;qinputs=[];tinputs=[]
    h=test.nets['q'].register_forward_pre_hook(lambda mod,args:qinputs.append(args[0].detach().cpu().clone()))
    ht=test.targets['q'].register_forward_pre_hook(lambda mod,args:tinputs.append(args[0].detach().cpu().clone()))
    try:
        metric=test.update(rows,native)
        dp=privileged(obs).shape[-1]
        replay=torch.cat([rows[i]['action'] for i in metric['replay_indices']],0)
        assert torch.equal(qinputs[0][:,dp:dp+29],replay)
        assert captures[0]['target'] and not captures[1]['target']
        assert torch.equal(tinputs[0][:,dp:dp+29],captures[0]['action'])
        assert torch.equal(qinputs[1][:,dp:dp+29],captures[1]['action'])
        for cap in captures:
            au=cap['audit'];independent=Envelope(operator,agent.goal.device)(au['base'].cuda(),au['raw'].cuda())[0].cpu()
            assert torch.equal(independent,cap['action'])
        result=dict(exact_replay_critic=True,exact_target_bootstrap=True,exact_actor_critic=True,
                    exact_common_operator=True,gradient_norms=metric['gradient_norms'],disposable_engineering_updates=1,
                    paths=[dict(target=x['target'],sample=x['sample'],sha={k:tree_digest(v) for k,v in x['audit'].items()},trigger_rate=float(x['audit']['triggered'].float().mean())) for x in captures])
    finally:
        h.remove();ht.remove();restore_global_rng(rng)
    assert tree_digest(agent.state())==original_digest
    return result
