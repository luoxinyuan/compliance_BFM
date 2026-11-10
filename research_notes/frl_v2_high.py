"""Approved matched privileged high-level PPO; immutable checkpoints and child eval."""
import copy,gzip,hashlib,json,math,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch

from dual_track_task import CFG
from frl_v2_task import Task,manifests as parent_manifests
from dual_track_runtime import PrivilegedBranch,PPOContract,TrainingBranch,save_training_checkpoint,restore_training_checkpoint
from frl_v2_fit import update
from frl_v2_retention import Retention
from dual_track_returns import advantages,physical_discount,high_reward_from_native

ROOT=Path(__file__).resolve().parents[1]
BC=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/interactive-contact-pilot-20261108/runs/bc-fit-01/old_data_only-500.pt')
BC_SHA='a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e'


def write(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def build():
    assert sha(BC)==BC_SHA
    torch.manual_seed(CFG['seed'])
    net=PrivilegedBranch(torch.load(BC,map_location='cpu',weights_only=True)).cuda()
    net.requires_grad_(False);net.actor.requires_grad_(True);net.critic.requires_grad_(True)
    net.log_std.fill_(math.log(CFG['raw_std']))
    return net


def evaluate(e,m,contract,out,args):
    from native_transfer import phase_report
    from dual_track_quality import quality as phase_quality
    from dual_track_task import digest
    assert sha(args.checkpoint)==args.sha256
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    net=build();net.load_state_dict(ck['model'],strict=True);net.eval().requires_grad_(False)
    groups=parent_manifests()[args.split]
    if args.smoke:groups=[groups[i] for i in (0,8,16,24,28)]
    task=Task(e,m,contract,56017);rows=[];prefixes={};source_hashes={}
    for group in groups:
        items=[('main',group['spec'])]
        if 'continued_spec' in group:items.append(('continue',group['continued_spec']))
        for child,s in items:
            torch.manual_seed(56017);np.random.seed(56017)
            label=group['parent_id'][:12]+'-'+child
            try:task.reset(s,record=True,label=label)
            except RuntimeError as exc:
                rows.append(dict(parent_id=group['parent_id'],source=group['source'],family=group['family'],child=child,
                                 source_failure=str(exc),complete=False,boundary_scope='source_before_policy'))
                write(out/'summaries.json',rows);continue
            source_hash=task.prefix_data['audit']['sha256']
            source_hashes.setdefault(source_hash,[]).append(group['parent_id'])
            decisions=[];paired_stop=group['spec'].get('stop_time',-1)
            while not task.done:
                x=torch.cat([task.student_features(),task.features()],-1)
                raw=net.distribution(x).mean
                if float(e.tether.time)<paired_stop-1e-7:
                    decisions.append(dict(t=float(e.tether.time),obs=sha_tensor(x),raw=sha_tensor(raw),state=sha_tensor(torch.as_tensor(e.sim.data.qpos))))
                task.step(raw)
            rr=dict(parent_id=group['parent_id'],source=group['source'],family=group['family'],child=child,
                condition=s,source_state_sha256=source_hash,complete=task.boundary is None,boundary=task.boundary,
                quality=phase_quality(task),**phase_report(task))
            prefix_physics=digest([r for r in task.physics if r['force_time']<paired_stop-1e-7])
            if child=='main' and 'continued_spec' in group:prefixes[group['parent_id']]=(decisions,prefix_physics,source_hash)
            if child=='continue':
                # Compare only the period before its paired stop time, not the
                # whole continuing task. The manifest never enters actor inputs.
                assert prefixes[group['parent_id']]==(decisions,prefix_physics,source_hash),'Common-prefix leak or physical mismatch'
                rr['paired_prefix_decisions']=len(decisions);rr['paired_prefix_exact']=True
            rows.append(rr);write(out/'summaries.json',rows)
            if args.record:task.save(out)
            print('DUAL_DEV',args.split,args.iteration,group['source'],group['family'],child,rr['all']['error_cm'],rr['all']['force_rms'],flush=True)
    collisions={k:v for k,v in source_hashes.items() if len(set(v))>1}
    assert not collisions,'Distinct parent profiles produced duplicate actual source states'
    write(out/'evaluation.json',dict(runtime=args.runtime,model_sha256=args.sha256,split=args.split,
        parent_targets=len(groups),actual_source_groups=len(source_hashes),source_hashes=source_hashes,
        evaluation_rng_seed=56017,rows=len(rows),RL_decisions=0))
    return dict(status='completed',evaluation_rows=len(rows),formal_RL_decisions=0)


def sha_tensor(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def select(rows,initial):
    """Approved family-balanced physical objective with explicit tail guards."""
    by_initial={(r['parent_id'],r['child']):r for r in initial}
    costs={};bad=[];family_errors={};family_forces={}
    for r in rows:
        key=(r['parent_id'],r['child']);base=by_initial[key]
        if r.get('source_failure') or base.get('source_failure'):
            continue # Source failure is not a learned-policy success or a policy-caused failure.
        if r['boundary']:bad.append(dict(key=key,reason='physical_boundary'));continue
        a,b=r['all'],base['all'];family=r['family'];q=r['quality'];bq=base['quality']
        if a['error_cm'] is not None:
            family_errors.setdefault(family,[]).append((a['error_cm'],b['error_cm']))
            family_forces.setdefault(family,[]).append((a['force_rms'],b['force_rms']))
            value=a['force_rms']+.03*a['error_cm']+a['cap_fraction']
        else:value=.5*np.linalg.norm(a['root_delta_cm'])/100+.02*abs(a['pelvis_yaw_change_deg'])
        slip=max(v['slip_peak_mps'] for v in q.values());oldslip=max(v['slip_peak_mps'] for v in bq.values())
        if slip>oldslip+max(.2,.5*oldslip):bad.append(dict(key=key,reason='slip_tail'))
        yaw=abs(q.get('released',list(q.values())[-1])['yaw_deg_final'])
        oldyaw=abs(bq.get('released',list(bq.values())[-1])['yaw_deg_final'])
        if yaw>oldyaw+10:bad.append(dict(key=key,reason='yaw_tail'))
        costs.setdefault(family,[]).append(float(value))
    for family,pairs in family_errors.items():
        current,base=np.median(pairs,axis=0)
        if current>base+max(2.,.2*base):bad.append(dict(family=family,reason='family_error_tail'))
    for family,pairs in family_forces.items():
        current,base=np.median(pairs,axis=0)
        if current>base+.75:bad.append(dict(family=family,reason='family_force_tail'))
    score=float(np.mean([np.median(costs[k]) for k in ('near','mid','stop') if k in costs])) if any(k in costs for k in ('near','mid','stop')) else 1e9
    return dict(score=score,admissible=not bad,family_costs={k:float(np.median(v)) for k,v in costs.items()},rejections=bad)


def train(e,m,contract,out,args):
    net=build();opt=torch.optim.Adam([{'params':net.actor.parameters(),'lr':CFG['actor_lr']},
                                     {'params':net.critic.parameters(),'lr':CFG['critic_lr']}])
    statehash=hashlib.sha256()
    for key,value in sorted(net.state_dict().items()):
        statehash.update(key.encode());statehash.update(value.detach().cpu().numpy().tobytes())
    write(out/'initialization.json',dict(model_state_sha256=statehash.hexdigest(),BC500_sha256=BC_SHA,
        joint_KL=args.kl,std=CFG['raw_std'],fresh_Adam=True,fresh_critic=True,
        actor_parameters=sum(p.numel() for p in net.actor.parameters()),
        critic_parameters=sum(p.numel() for p in net.critic.parameters()),
        old_input_and_normalizer_retained=True,zero_privileged_output_branch=True))
    retention=Retention(net,args.arm=='B')
    task=Task(e,m,contract,CFG['seed']);source_failures=[];completed_episodes=[]
    def reset_source(label):
        for attempt in range(20):
            try:
                task.reset(record=True,label=label)
                with (out/'source-episodes.jsonl').open('a') as f:
                    f.write(json.dumps(dict(label=label,spec=task.spec,
                        actual_source_sha256=task.prefix_data['audit']['sha256'],
                        support=np.asarray(e.last_support).tolist()))+'\n')
                return
            except RuntimeError as exc:
                if not str(exc).startswith('native source prefix failed:'):raise
                source_failures.append(dict(spec=copy.deepcopy(task.spec),error=str(exc),attempt=attempt))
                write(out/'native-source-failures.json',source_failures)
        raise RuntimeError('Twenty unmaintainable source prefixes; no learning/qualification invented')
    reset_source('train0')
    first=torch.cat([task.student_features(),task.features()],-1)
    assert torch.equal(net.distribution(first).mean,net.base.distribution(first[:,:1639]).mean)
    rng=torch.Generator(device='cuda').manual_seed(CFG['seed']*997)
    guard=PPOContract(joint_kl_limit=args.kl,joint_kl_stop=.75*args.kl,joint_kl_assert=args.kl+.00001)
    runtime=args.runtime;logs=[];dev=[];decisions=0;episode=0;coverage={};episode_coverage={};initial_rows=None;best=None
    def save(i):
        path=out/f'candidate-{i}.pt';assert not path.exists()
        save_training_checkpoint(path,task,net,opt,[rng],dict(iteration=i,RL_decisions=decisions,episode=episode,
            runtime=runtime,initial_BC500_sha256=BC_SHA,joint_KL=args.kl,kind='privileged_Expert',arm=args.arm,retention=retention.state_dict(),coverage=coverage,
            episode_coverage=episode_coverage))
        return path
    def dev_eval(i,path):
        nonlocal initial_rows,best
        state=TrainingBranch(task,[rng]);dest=out/f'dev-{i}'
        cmd=[sys.executable,str(ROOT/'research_notes/frl_v2_stage.py'),'--mode','high_eval','--output',str(dest),
             '--checkpoint',str(path),'--sha256',sha(path),'--iteration',str(i),'--split','dev','--arm',args.arm,'--record']
        if args.smoke:cmd.append('--smoke')
        t=time.monotonic()
        with (out/f'dev-{i}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=2400)
        assert state.digest()==state.fingerprint
        # No restoring over accidental mutation: parent state and all RNG streams
        # must remain unchanged. Child process owns its simulator and sampling RNG.
        from dual_track_runtime import RNGState
        after=RNGState(task,[rng]);before=state.rng_all
        assert after.python==before.python
        assert after.numpy[0]==before.numpy[0] and np.array_equal(after.numpy[1],before.numpy[1]) and after.numpy[2:]==before.numpy[2:]
        assert torch.equal(after.cpu,before.cpu) and all(torch.equal(a,b) for a,b in zip(after.cuda,before.cuda))
        assert after.task_rng==before.task_rng and torch.equal(after.generators[0][1],before.generators[0][1])
        rows=json.loads((dest/'summaries.json').read_text())
        if initial_rows is None:initial_rows=rows
        choice=select(rows,initial_rows);choice.update(iteration=i,sha256=sha(path),checkpoint_path=str(path),eval_job_s=time.monotonic()-t)
        dev.append(choice);write(out/'development.json',dev)
        if i==0 or (choice['admissible'] and choice['score']<best['score']):best=copy.deepcopy(choice);write(out/'selection.json',best)
        write(out/f'dev-{i}-isolation.json',dict(physical_history_exact=True,torch_and_task_rng_exact=True,independent_process=True))
    first_iteration=1
    if args.resume:
        assert not args.smoke
        restored=restore_training_checkpoint(args.resume,task,net,opt,[rng])
        assert restored['runtime']==runtime and restored['joint_KL']==args.kl,'Use original pinned runtime and KL for resume'
        retention.load_state_dict(restored['retention'])
        decisions=restored['RL_decisions'];episode=restored['episode'];coverage=restored['coverage']
        episode_coverage=restored.get('episode_coverage',{});first_iteration=restored['iteration']+1
        parent=args.resume.parent
        if (parent/'dev-0/evaluation.json').exists():
            initial_rows=json.loads((parent/'dev-0/summaries.json').read_text())
            logs=[r for r in json.loads((parent/'learning.json').read_text()) if r['iteration']<first_iteration] if (parent/'learning.json').exists() else []
            dev=[r for r in json.loads((parent/'development.json').read_text()) if r['iteration']<first_iteration]
            best=copy.deepcopy(min([r for r in dev if r['iteration']==0 or r['admissible']],key=lambda r:r['score']))
            best.setdefault('checkpoint_path',str(parent/f"candidate-{best['iteration']}.pt"))
        write(out/'resume.json',dict(checkpoint=str(args.resume),sha256=sha(args.resume),counters=restored,
            physical_history_and_all_RNG_exact=True,curriculum_pair_memory_restored=True))
        dev_eval(restored['iteration'],args.resume)
    else:dev_eval(0,save(0))
    batches=1 if args.smoke else CFG['high_batches_per_arm']
    for iteration in range(first_iteration,batches+1):
        import shutil
        assert shutil.disk_usage(out).free>100*(1<<30),'100GiB shared free-space protection'
        if (out/'STOP_REQUEST.json').exists():
            if not (out/f'candidate-{iteration-1}.pt').exists():save(iteration-1)
            return dict(status='stopped_by_request',Expert_RL_decisions=decisions)
        arrays={k:[] for k in ('obs','action','logp','value','next_value','reward','discount','terminated','truncated','pre_mask')}
        stats=[];probes=[];control_records=[];physical_records=[];t0=time.monotonic()
        for j in range(CFG['batch']):
            if task.done:
                episode+=1;reset_source(f'train{episode}')
            x=torch.cat([task.student_features(),task.features()],-1).detach().clone()
            arrays['pre_mask'].append(task.precontact_eligible())
            decision_time=float(e.tether.time)
            dist=net.distribution(x);raw=dist.mean+dist.stddev*torch.randn(dist.mean.shape,device='cuda',generator=rng)
            if j%96==0:
                probes.append(dict(x=x.clone(),obs={k:v.clone() for k,v in e.obs.items()},previous=task.last_command.clone()))
            lp=dist.log_prob(raw).sum(-1);v=net.value(x)
            phys_start=len(task.physics);ctrl_start=len(task.controls)
            _,reward,done,st=task.step(raw)
            if done:
                completed_episodes.append(dict(parent_id=task.spec['parent_id'],family=task.spec['demand_family'],source=task.spec['source'],actual_decisions=len(task.stats),return_=task.episode_return,boundary=copy.deepcopy(task.boundary)))
                write(out/'completed-episodes.json',completed_episodes)
                if task.boundary is not None:
                    failure_dir=out/'physical-failures';failure_dir.mkdir(exist_ok=True);task.save(failure_dir)
            control_records.extend(dict(parent_id=task.spec['parent_id'],**{k:c[k] for k in ('time','executed_z','action','q_PD','pre_qpos','pre_qvel','actual_duration_s','reward_terms','reward_integral')}) for c in task.controls[ctrl_start:])
            physical_records.extend(dict(parent_id=task.spec['parent_id'],**{k:r[k] for k in ('time','force_time','active','force','error','qpos','qvel','support','feet_xyz','torque','joint_limit_excess_max_rad','stage_loaded_slip_mps','pelvis_matrix','torso_matrix')}) for r in task.physics[phys_start:])
            from transition_env import phase
            native_phase_counts={}
            for c in task.controls[-st['control_steps']:]:
                key=phase(task.spec,c['time']);native_phase_counts[key]=native_phase_counts.get(key,0)+1
            st.update(decision_time_s=decision_time,native_phase_counts=native_phase_counts)
            nx=torch.cat([task.student_features(),task.features()],-1)
            nv=torch.zeros(1,device='cuda') if st['terminated'] else net.value(nx)
            for k,val in dict(obs=x[0],action=raw[0],logp=lp[0],value=v[0],next_value=nv[0]).items():arrays[k].append(val.cpu().numpy())
            native_durations=[c['actual_duration_s'] for c in task.controls[-st['control_steps']:]]
            training_reward=high_reward_from_native(st['native_rewards'],native_durations)
            st['undiscounted_physical_reward']=reward;st['PPO_reward_native_discounted']=training_reward
            arrays['reward'].append(training_reward);arrays['discount'].append(physical_discount(st['actual_duration_s']))
            arrays['terminated'].append(st['terminated']);arrays['truncated'].append(st['truncated'])
            key=st['source']+'/'+st['family']+'/'+st.get('phase','unknown');coverage[key]=coverage.get(key,0)+1
            if done:episode_coverage[key]=episode_coverage.get(key,0)+1
            st.update(parent_id=task.spec['parent_id'],raw_mean_norm=float(dist.mean.norm()),raw_sample_norm=float(raw.norm()),std=float(dist.stddev.mean()))
            stats.append(st);decisions+=1
        cpu={k:np.asarray(v,dtype=bool if k in ('terminated','truncated','pre_mask') else np.float32) for k,v in arrays.items()}
        with gzip.open(out/f'native-controls-{iteration}.json.gz','wt') as f:json.dump(control_records,f)
        with gzip.open(out/f'physics-{iteration}.json.gz','wt') as f:json.dump(physical_records,f)
        np.savez_compressed(out/f'rollout-{iteration}.npz',**cpu);write(out/f'rollout-{iteration}-stats.json',stats)
        b={k:torch.from_numpy(v).cuda() for k,v in cpu.items()}
        adv,returns=advantages(b['reward'],b['value'],b['next_value'],b['discount'],b['terminated'],b['truncated'],CFG['gae'])
        b.update(returns=returns,advantages=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6),raw_adv_std=(returns-b['value']).std())
        if iteration==1:
            original={k:v.detach().clone() for k,v in net.actor.state_dict().items()}
            with torch.enable_grad():
                for _ in range(CFG['critic_value_only_first_batch']):
                    opt.zero_grad(set_to_none=True);loss=.5*((net.value(b['obs'])-b['returns'])/10).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(net.critic.parameters(),1.);opt.step()
            assert all(torch.equal(v,original[k]) for k,v in net.actor.state_dict().items())
        retention.seed_reference(b['obs'],b['pre_mask'])
        before=net.distribution(b['obs']).mean.detach().double()
        log=update(net,opt,b,epochs=CFG['epochs'],minibatch=CFG['minibatch'],
                   critic_shuffle_seed=CFG['seed']*10000+iteration,contract=guard,retention=retention)
        log['precontact_retention']=retention.after_batch(net,b['obs'],b['pre_mask'])
        after=net.distribution(b['obs']).mean.detach().double()
        from clean_policy import direct_command
        from locomotion_rollout import spherical
        action_changes=[];target_changes=[]
        for probe in probes:
            ix=len(action_changes)*96
            oldz=direct_command(before[ix:ix+1].float(),e.standing)
            newz=direct_command(after[ix:ix+1].float(),e.standing)
            oldactions=[];newactions=[]
            for f in (.2,.4,.6,.8,1.):
                oldactions.append(m.act(probe['obs'],spherical(probe['previous'],oldz,f),mean=True))
                newactions.append(m.act(probe['obs'],spherical(probe['previous'],newz,f),mean=True))
            da=torch.stack(newactions)-torch.stack(oldactions)
            action_changes.append(dict(rms=float(da.square().mean().sqrt()),maximum=float(da.abs().max())))
            # Original executor mapping uses the same action scale for both.
            scale=torch.as_tensor(contract['q_PD_scale'],device=da.device)
            bound=contract['clip']/contract['normalize_ratio']
            qdiff=(torch.stack(newactions).clamp(-bound,bound)-torch.stack(oldactions).clamp(-bound,bound))*scale
            target_changes.append(float(torch.rad2deg(qdiff).abs().max()))
        cosine=(before*after).sum(-1)/(before.norm(dim=-1)*after.norm(dim=-1)).clamp_min(1e-30)
        log.update(iteration=iteration,RL_decisions=decisions,episodes=episode,coverage=copy.deepcopy(coverage),
                   episode_coverage=copy.deepcopy(episode_coverage),wall_s=time.monotonic()-t0,
                   frozen_observation_five_substep_action_changes=action_changes,
                   frozen_observation_qpd_max_changes_deg=target_changes,
                   probe_contract='same BFM observation and actual previous-z; five interpolation fractions; not a physics rollout',
                   legal_z_mean_change_deg=float(torch.rad2deg(torch.acos(cosine.clamp(-1,1))).mean()))
        logs.append(log);write(out/'learning.json',logs)
        print('ENGINEERING_PPO' if args.smoke else 'FORMAL_PRIVILEGED_PPO',args.kl,iteration,decisions,log['optimizer_updates'],log['legal_z_mean_change_deg'],flush=True)
        checkpoint=save(iteration)
        if iteration in (16,32,64,96,128) or iteration==batches:dev_eval(iteration,checkpoint)
    return dict(status='completed',Expert_RL_decisions=0 if args.smoke else decisions,
        engineering_sample_decisions=decisions if args.smoke else 0,selected=best,joint_KL=args.kl,Student_RL_decisions=0)
