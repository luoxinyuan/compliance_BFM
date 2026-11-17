"""Disposable saved-state cold/warm continuation diagnosis, not formal learning.

Calls the unchanged native compiled update. Boundary observers run outside its
compiled methods. No comparator tolerance is introduced and no weights migrate
from this qualification job into a scientific run.
"""
import argparse
import os
from pathlib import Path
import sys
import time
import json
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes'),str(Path(__file__).parent)]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from learning_checkpoint import cpu,digest,fingerprint,rng,set_rng,write,file_sha,Audited


def restore(agent,state,model):
    from humanoidverse.agents.buffers.transition import DictBuffer
    agent._model.load_state_dict(model,strict=True)
    for name,s in state['optimizers'].items():getattr(agent,name).load_state_dict(cpu(s))
    for name,p in agent._model.named_parameters():
        g=state['gradients'][name];p.grad=None if g is None else g.to(agent.device).clone()
    for name,m in agent._model.named_modules():m.training=state['module_modes'][name]
    zb=state['z_buffer'];agent.z_buffer._storage.copy_(zb['storage'].to(agent.device))
    agent.z_buffer._idx=zb['idx'];agent.z_buffer._is_full=zb['is_full'];agent.actor_enabled=state['actor_enabled']
    replay=DictBuffer(capacity=state['replay_capacity'],device='cpu');replay.extend(state['replay'])
    assert replay._idx==state['replay_idx'] and replay._is_full==state['replay_full']
    groups=state['expert_groups']
    class Sequences:
        def sample(self,n):
            rows=[]
            for _ in range(n//8):
                g=groups[int(torch.randint(len(groups),()).item())]
                start=int(torch.randint(len(g)-7,()).item());rows.extend(g[start:start+8])
            def combine(xs):
                if isinstance(xs[0],dict):return {k:combine([v[k] for v in xs]) for k in xs[0]}
                return torch.cat(xs)
            x=combine(rows);return dict(observation=x['observation'],next=dict(observation=x['next']['observation']))
    set_rng(state['rng'])
    return dict(train=replay,expert_slicer=Sequences())


def tree_info(x):
    if torch.is_tensor(x):return dict(sha256=digest(x),shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),device=str(x.device))
    if isinstance(x,dict):return {k:tree_info(v) for k,v in x.items()}
    return dict(sha256=digest(x))


def numerical_diff(a,b):
    result=[]
    def walk(x,y,path=''):
        if isinstance(x,dict):
            for k in x:walk(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,(list,tuple)):
            for i,(u,v) in enumerate(zip(x,y)):walk(u,v,path+'/'+str(i))
        elif torch.is_tensor(x):
            y=y.detach().cpu()
            if not torch.equal(x,y):
                z=(x.double()-y.double()).abs()
                result.append(dict(path=path,shape=list(x.shape),different=int((x!=y).sum()),max_abs=float(z.max()),l2=float(z.norm())))
        elif x!=y:result.append(dict(path=path,old=str(x),new=str(y)))
    walk(a,b);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();updates=0
    write(args.output/'launch.json',dict(pid=os.getpid(),started_unix=time.time(),scope='saved learner cold/warm diagnosis; no physics or formal learning'))
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high');torch.manual_seed(681117)
    try:
        from agent import build
        manifest=json.loads((args.package/'manifest.json').read_text())
        for name,info in manifest['files'].items():assert file_sha(args.package/name)==info['sha256']
        state=torch.load(args.package/'learner.pt',map_location='cpu',weights_only=False)
        model=torch.load(args.package/'model.pt',map_location='cpu',weights_only=False)
        agent,_=build(aux_scale=state['aux_scale'])
        boundary=[]
        for name in ['sample_mixed_z','update_discriminator','update_fb','update_critic','update_aux_critic','update_actor']:
            fn=getattr(agent,name)
            def observer(*a,_name=name,_fn=fn,**kw):
                item=dict(method=_name,inputs=tree_info(kw),rng_before=digest(rng()))
                if _name=='update_fb':
                    with torch.no_grad():
                        fs=agent._model._forward_map(kw['obs'],kw['z'],kw['action'])
                        b=agent._model._backward_map(kw['goal'])
                    item['eager_current_F']=tree_info(fs);item['eager_current_B']=tree_info(b)
                r=_fn(*a,**kw)
                item.update(outputs=tree_info(r),rng_after=digest(rng()))
                boundary.append(item);return r
            setattr(agent,name,observer)
        first_numeric=None;results=[]
        for iteration in range(2):
            buffers=restore(agent,state,model)
            assert fingerprint(agent)==manifest['expected_initial']
            boundary.clear();trace=[]
            torch.cuda.synchronize();t=time.monotonic()
            metrics=agent.update({k:Audited(v,k,trace) for k,v in buffers.items()},state['step'])
            torch.cuda.synchronize();updates+=1
            fp=fingerprint(agent)
            result=dict(iteration=iteration,initial_exact=True,wall_s=time.monotonic()-t,boundary=cpu(boundary),
                        samples=trace,state=fp,metrics={k:digest(v) for k,v in metrics.items()})
            write(args.output/f'pass-{iteration}.json',result);results.append(result)
            print('RESTORE_DIAG_PASS',iteration,result['wall_s'],flush=True)
            numeric=dict(model=cpu(agent._model.state_dict()),optimizers=cpu(agent.optimizer_dict),
                         gradients={k:cpu(p.grad) for k,p in agent._model.named_parameters()},metrics=cpu(metrics))
            if first_numeric is None:first_numeric=numeric
            else:
                differences=numerical_diff(first_numeric,numeric)
                write(args.output/'numerical-differences.json',differences)
        old=json.loads((args.package.parent/'uninterrupted-update5.json').read_text())
        old_resume=json.loads((args.package.parent.parent/'restore/comparison.json').read_text())['restored_update']
        write(args.output/'result.json',dict(cold_vs_warm_exact=all(results[0][k]==results[1][k] for k in ['state','samples','metrics']),
            old_uninterrupted_equal=[all(r[k if k!='samples' else 'samples']==old[k if k!='samples' else 'sample_trace'] for k in ['state','metrics','samples']) for r in results],
            old_restored_equal=[all(r[k if k!='samples' else 'samples']==old_resume[k if k!='samples' else 'sample_trace'] for k in ['state','metrics','samples']) for r in results],
            numerical_differing_leaves=len(differences),complete_updates=updates,scope='disposable learner only'))
    except BaseException as e:
        write(args.output/'failure.json',dict(type=type(e).__name__,error=str(e)));raise
    finally:write(args.output/'cost.json',dict(wall_s=time.monotonic()-started,complete_agent_updates=updates,new_physics_ticks=0,formal_updates=0))


if __name__=='__main__':main()
