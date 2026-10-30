"""Fixed paired supervised fits; original actor differentiable only w.r.t. alpha."""
import copy,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from alpha_memory import ForceMemory
from alpha_student import AlphaStudent,student_input,path_z
from repair_data_contract import target_fields,phase_indices


def check_time():
    assert time.monotonic()<float(os.environ['BFM_STAGE_DEADLINE_MONOTONIC']), 'Compute budget exhausted'


def state_digest(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fit_pair(env,model,goal,contract,out,protocol):
    dev=env.standing.device;assert dev.type=='cuda'
    root=Path(__file__).resolve().parents[1]
    runs=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs')
    keys=protocol['memory']['keys']
    def save(name,x):(out/name).write_text(json.dumps(x,indent=2,allow_nan=False))
    observations=[];memory=[];controls=[];episode_index=[];groups=[];strata=[];manifest=[];targets=[]
    for ei,entry in enumerate(protocol['training_sources']):
        path=runs/entry['path'];raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==entry['sha256']
        ep=json.loads(raw);assert ep['boundary'] is None
        assert ep['condition']['name'] not in ('new_short03','new_long07')
        start=len(controls);fifo=ForceMemory(protocol['memory']['force_samples'])
        for c in ep['control']:
            memory.append(fifo.append(c['measured_force_xyz']));observations.append(c['bfm_obs'])
            controls.append(c);episode_index.append(ei);targets.append(target_fields(c,entry['label']))
        groups.append(torch.arange(start,len(controls),device=dev))
        pools={'whole':list(range(len(ep['control'])))} if entry['kind']=='normal' else phase_indices(ep['control'],ep['condition'])
        assert entry['batch_count']%len(pools)==0
        for phase,ix in pools.items():
            strata.append((torch.tensor([start+i for i in ix],device=dev),entry['batch_count']//len(pools)))
        manifest.append(dict(**entry,start=start,samples=len(ep['control']),strata_sizes={k:len(v) for k,v in pools.items()}))
    assert sum(count for _,count in strata)==protocol['optimization']['batch_size']==128
    obs={k:torch.tensor([o[k][0] for o in observations],device=dev,dtype=torch.float32) for k in keys}
    x=student_input(model,obs,memory,keys).detach();assert x.shape==(protocol['training_samples'],protocol['memory']['student_dim'])
    normalized={k:v.detach() for k,v in model._normalize(obs).items()}
    target_alpha=torch.tensor([[v[0]] for v in targets],device=dev)
    target_action=torch.tensor([v[1] for v in targets],device=dev)
    target_q=torch.tensor([v[2] for v in targets],device=dev)
    standing=env.standing;scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev)
    def q_of(action):return offset+(action*5).clamp(-contract['clip'],contract['clip'])*scale/5
    def predict_action(alpha,ids):
        z=path_z(standing,goal,alpha)
        action=model._actor({k:v[ids] for k,v in normalized.items()},z,model.cfg.actor_std).mean.float()
        return action,q_of(action),z
    allids=torch.arange(len(x),device=dev)
    # Same saved o,z and exact original public inference/PD contract.
    parity=dict(public_actor_max=0.,saved_teacher_action_max=0.,saved_teacher_q_max=0.,z_norm_error=0.)
    for ids in allids.split(64):
        a,q,z=predict_action(target_alpha[ids],ids)
        public=model.act({k:v[ids] for k,v in obs.items()},z,mean=True)
        parity['public_actor_max']=max(parity['public_actor_max'],float(abs(public-a).max()))
        parity['saved_teacher_action_max']=max(parity['saved_teacher_action_max'],float(abs(a-target_action[ids]).max()))
        parity['saved_teacher_q_max']=max(parity['saved_teacher_q_max'],float(abs(q-target_q[ids]).max()))
        parity['z_norm_error']=max(parity['z_norm_error'],float(abs(z.norm(dim=-1)-16).max()))
    assert parity['public_actor_max']<1e-6 and parity['saved_teacher_q_max']<2e-5 and parity['z_norm_error']<1e-4
    i=int(abs(target_alpha[:,0]-.5).argmin());ids=torch.tensor([i],device=dev)
    with torch.enable_grad():
        u=torch.tensor([[float(target_alpha[i])+.1]],device=dev,requires_grad=True)
        _,q,_=predict_action(u,ids);loss=((q-target_q[ids])/.1).square().mean();gradient=torch.autograd.grad(loss,u)[0]
        with torch.no_grad():
            eps=.001
            _,qp,_=predict_action(u+eps,ids);_,qm,_=predict_action(u-eps,ids)
            fd=((((qp-target_q[ids])/.1).square().mean()-((qm-target_q[ids])/.1).square().mean())/(2*eps))
        assert torch.isfinite(gradient).all() and abs(float(gradient))>1e-6
        relative=abs(float(gradient)-float(fd))/max(abs(float(fd)),1e-5)
        assert relative<.03 and all(p.grad is None for p in model.parameters())
    save('gradient-contract.json',dict(parity=parity,index=i,alpha_probe=float(u),analytic_gradient=float(gradient),finite_difference=float(fd),relative_error=relative,bfm_has_grad=False))
    initial_path=runs/protocol['initialization']['source']
    assert hashlib.sha256(initial_path.read_bytes()).hexdigest()==protocol['initialization']['sha256']
    prior=torch.load(initial_path,map_location=dev,weights_only=True)
    initial={k:v.detach().clone() for k,v in prior['state_dict'].items()}
    template=AlphaStudent(x.shape[1]).to(dev);template.load_state_dict(initial)
    initial_sha=state_digest(initial)
    torch.save({k:v.cpu() for k,v in initial.items()},out/'shared-initial-state.pt')
    save('dataset-manifest.json',dict(episodes=manifest,samples=len(x),input_dim=x.shape[1],keys=keys,force_history_samples=320,
        no_teacher_state_input=True,sealed_final_used=False,old_gap_development_used=True,old_normal_training_only=True,initial_state_sha256=initial_sha,
        trainable_parameters=sum(p.numel() for p in template.parameters()),protocol=protocol))
    torch.save(dict(x=x.cpu(),obs={k:v.cpu() for k,v in obs.items()},target_alpha=target_alpha.cpu(),target_q=target_q.cpu(),target_action=target_action.cpu(),episode_index=episode_index),out/'training-dataset.pt')
    cfg=protocol['optimization'];results={};batch_hashes={}
    def stats(net,selected):
        chunks=[];acts=[];qs=[];zs=[]
        with torch.no_grad():
            for ix in selected.split(64):
                alpha=net(x[ix]);a,q,z=predict_action(alpha,ix)
                chunks.append(alpha);acts.append(a);qs.append(q);zs.append(z)
        alpha=torch.cat(chunks);a=torch.cat(acts);q=torch.cat(qs);z=torch.cat(zs)
        metrics=dict(alpha_rmse=float((alpha-target_alpha[selected]).square().mean().sqrt()),alpha_max_abs=float(abs(alpha-target_alpha[selected]).max()),
            q_PD_rms_deg=float(torch.rad2deg((q-target_q[selected]).square().mean().sqrt())),q_PD_max_abs_deg=float(torch.rad2deg(abs(q-target_q[selected]).max())),
            action_rmse=float((a-target_action[selected]).square().mean().sqrt()),z_norm_max_error=float(abs(z.norm(dim=-1)-16).max()),alpha_min=float(alpha.min()),alpha_max=float(alpha.max()))
        return metrics,alpha,q
    started=time.monotonic()
    for phase,updates in [('full',cfg['updates'])]:
        selected=groups[0] if phase=='single' else allids
        for objective in ['A','B']:
            net=AlphaStudent(x.shape[1]).to(dev);net.load_state_dict(initial);net.train()
            assert state_digest(net.state_dict())==initial_sha
            optimizer=torch.optim.Adam(net.parameters(),lr=cfg['lr'])
            generator=torch.Generator(device=dev).manual_seed(cfg['batch_seed']);batch_hash=hashlib.sha256()
            before,_,_=stats(net,selected);trace=[];start=time.monotonic()
            with torch.enable_grad():
                for step in range(updates):
                    check_time()
                    ix=torch.cat([g[torch.randint(len(g),(count,),device=dev,generator=generator)] for g,count in strata])
                    batch_hash.update(ix.detach().cpu().numpy().tobytes());optimizer.zero_grad(set_to_none=True)
                    alpha=net(x[ix])
                    if objective=='A':loss=(alpha-target_alpha[ix]).square().mean()
                    else:
                        _,q,_=predict_action(alpha,ix);loss=((q-target_q[ix])/.1).square().mean()
                    assert torch.isfinite(loss)
                    loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['grad_clip'])
                    assert torch.isfinite(gn) and all(p.grad is None for p in model.parameters())
                    if step==0:assert float(gn)>0
                    optimizer.step()
                    if step==0 or (step+1)%100==0:
                        trace.append(dict(update=step+1,loss=float(loss),gradient_norm=float(gn),elapsed_s=time.monotonic()-start))
                        save(f'{phase}_{objective}_trace.json',trace);print('ALPHA_FIT',phase,objective,step+1,float(loss),flush=True)
                    if phase=='full' and (step+1)%500==0:
                        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},updates=step+1,objective=objective,keys=keys,protocol=protocol),out/f'alpha_{objective}_step{step+1:04d}.pt')
            after,alpha,q=stats(net,selected);batch_hashes[phase+'_'+objective]=batch_hash.hexdigest()
            detail=dict(updates=updates,batch_size=cfg['batch_size'],before=before,after=after,wall_seconds=time.monotonic()-start,
                initial_state_sha256=initial_sha,batch_sequence_sha256=batch_hash.hexdigest(),per_condition={})
            if phase=='full':
                for name,g in zip([e['name'] for e in protocol['training_sources']],groups):
                    met,aa,qq=stats(net,g);met['q_PD_step_max_deg']=float(torch.rad2deg(abs(torch.diff(qq,dim=0)).max()))
                    detail['per_condition'][name]=met
                detail['static_command_screen_passed']=all(v['q_PD_step_max_deg']<10 for v in detail['per_condition'].values())
                detail['offline_fit_descriptive_pass']=after['alpha_rmse']<.05 and after['q_PD_rms_deg']<1
                net.eval().requires_grad_(False)
                for p in net.parameters():p.grad=None
                checkpoint=out/protocol['final_checkpoints'][objective]
                torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},updates=updates,objective=objective,keys=keys,protocol=protocol,initial_state_sha256=initial_sha),checkpoint)
                detail['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                save(f'alpha_{objective}_offline_predictions.json',dict(alpha=alpha[:,0].tolist(),target_alpha=target_alpha[:,0].tolist(),q_PD=q.tolist()))
            results[phase+'_'+objective]=detail;save('paired-fit-result.json',results)
            print('ALPHA_FIT_COMPLETE',phase,objective,json.dumps(detail),flush=True)
        assert batch_hashes[phase+'_A']==batch_hashes[phase+'_B']
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    return dict(status='completed',fits=results,student_optimizer_updates=sum(r['updates'] for r in results.values()),
        full_training_updates_per_model=2000,single_diagnostic_updates_per_model=0,wall_seconds=time.monotonic()-started,
        bfm_updates=0,final_selection='fixed_step_2000',physics_not_run_in_this_stage=True)
