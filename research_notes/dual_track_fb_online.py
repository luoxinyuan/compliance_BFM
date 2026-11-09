"""Approved contact97 online FB probe; original six-update loop, fresh assets.

The separate original model executes source prefixes. The independent mutable
actor executes every controlled50Hz action. No archived high-level actor is used.
Versioned full packages rebuild the extended topology before loading weights.
"""
import copy,json,hashlib,os,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch.utils._pytree import tree_map
from dual_track_task import Task,CFG,global_rng,restore_global_rng,parent_manifests
from dual_track_native_data import NativeActor,NativeSequences
from dual_track_fb_extension import load_warm_started_agent,freeze_robot_normalizers,EXTENSION_VERSION
from dual_track_runtime import TrainingBranch
from dual_track_high import write,sha

P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
MODEL=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view/checkpoint/model'
BASE_SHA='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361'


def factory(compile=False):
    from humanoidverse.agents.nn_models import weight_init
    assert sha(MODEL/'model.safetensors')==BASE_SHA
    torch.manual_seed(5602)
    agent,manifest=load_warm_started_agent(MODEL,device='cuda')
    f=CFG['fb']
    train=agent.cfg.train.model_copy(update=dict(discount=CFG['gamma_native'],batch_size=f['batch'],
        reg_coeff=f['reg_coeff'],reg_coeff_aux=f['reg_coeff_aux'],scale_reg=f['scale_reg'],
        train_goal_ratio=f['train_goal_ratio'],expert_asm_ratio=f['expert_asm_ratio'],relabel_ratio=f['relabel_ratio']))
    agent.cfg=agent.cfg.model_copy(update=dict(train=train,aux_rewards=['contact_physical'],
        aux_rewards_scaling={'contact_physical':1.},compile=False))
    agent._model._aux_reward_normalizer=torch.nn.Identity()
    agent._model._aux_critic.apply(weight_init)
    agent._model._target_aux_critic.load_state_dict(agent._model._aux_critic.state_dict())
    freeze_robot_normalizers(agent._model)
    assert all(not x['state'] for x in agent.optimizer_dict.values())
    assert agent.z_buffer.empty()
    if compile:
        agent.cfg=agent.cfg.model_copy(update={'compile':True});agent.setup_compile()
    return agent,manifest


def stack(rows):
    first=rows[0]
    return {k:stack([r[k] for r in rows]) if isinstance(v,dict) else torch.cat([r[k] for r in rows]) for k,v in first.items()}


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def cpu(tree):return tree_map(lambda x:x.detach().cpu() if torch.is_tensor(x) else x,tree)


class Collector:
    def __init__(self,e,m,contract,agent,capacity):
        from humanoidverse.agents.buffers.transition import DictBuffer
        self.task=Task(e,m,contract,5602)
        self.proxy=NativeActor(self.task,m,agent._model,stochastic=True)
        self.task.m=self.proxy
        self.replay=DictBuffer(capacity=capacity,device='cpu')
        self.sequences=NativeSequences()
        self.decisions=0;self.transitions=0;self.physics=0;self.source_controls=0;self.source_physics=0
        self.episodes=0;self.coverage={};self.episode_rows=[];self.native_actions=[];self.source_failures=[]

    def reset(self,s=None):
        self.proxy.begin_source()
        prefill=self.task.total_prefill
        try:self.task.reset(s,record=True,label=f'fb-{self.episodes}')
        except RuntimeError as exc:
            self.source_failures.append(dict(spec=copy.deepcopy(self.task.spec),error=str(exc)))
            raise
        self.proxy.finish_pending()
        self.sequences.add(self.task.spec['source'],self.task.spec['source_parent_id'],self.proxy.source_pairs)
        self.source_controls+=len(self.proxy.source_pairs)
        self.source_physics+=self.task.total_prefill-prefill+len(self.task.prefix_data['physics'])
        self.proxy.begin_online();self.native_actions=[]

    def bootstrap_sources(self):
        # Three new native free prefixes for CPR, never PPO or train replay.
        from dual_track_task import sample_spec
        r=np.random.default_rng(560201)
        for source in CFG['source_probs']:
            self.reset(sample_spec(r,source,'none'))
        self.reset()

    def collect(self,count):
        start=time.monotonic();added=0
        while added<count:
            if self.task.done:
                self.episode_rows.append(dict(spec=copy.deepcopy(self.task.spec),return_=self.task.episode_return,
                    boundary=copy.deepcopy(self.task.boundary),decisions=len(self.task.decision_rows),controls=len(self.task.controls)))
                if self.task.boundary and hasattr(self,'output'):
                    dest=self.output/'failures';dest.mkdir(exist_ok=True)
                    self.task.save(dest)
                self.episodes+=1;self.reset()
            raw=self.task.native(self.task.spec['source'],float(self.task.e.tether.time))/16
            rows,done,stats=self.proxy.take_step(raw,native_limit=min(5,count-added))
            assert added+len(rows)<=count,'Collection/checkpoint boundary must align to actual high-level calls'
            for row in rows:
                dt=float(row['duration'])
                assert abs(dt-.02)<1e-6 or bool(row['next']['terminated']), 'Nonterminal discount-duration mismatch'
                self.native_actions.append(row['action'].tolist())
                self.physics+=round(dt/.005)
            self.replay.extend(stack(rows))
            added+=len(rows);self.transitions+=len(rows);self.decisions+=1
            key='/'.join((stats['source'],stats['family'],stats['phase']))
            self.coverage[key]=self.coverage.get(key,0)+len(rows)
        return dict(native_transitions=added,wall_s=time.monotonic()-start)

    @property
    def buffers(self):return dict(train=self.replay,expert_slicer=self.sequences)


def save_package(path,agent,c,updates,runtime):
    assert not path.exists();path.mkdir()
    begun=time.monotonic();torch.cuda.synchronize()
    state=TrainingBranch(c.task)
    torch.save(cpu(agent._model.state_dict()),path/'model.pt')
    torch.save(cpu(agent.optimizer_dict),path/'optimizers.pt')
    torch.save(dict(storage=cpu(agent.z_buffer._storage),idx=agent.z_buffer._idx,full=agent.z_buffer._is_full),path/'zbuffer.pt')
    torch.save(c.sequences,path/'native-sequences.pt')
    c.replay.save(path/'replay')
    write(path/'original-init-kwargs.json',json.loads((MODEL/'init_kwargs.json').read_text()))
    r=state.rng_all
    history=dict(spec=copy.deepcopy(c.task.spec),raw=[copy.deepcopy(d['raw']) for d in c.task.decision_rows],
        native_limits=[d['administrative_native_limit'] for d in c.task.stats],
        actual_native_actions=copy.deepcopy(c.native_actions),digest=state.fingerprint,
        physics_rng=copy.deepcopy(c.task.physics_rng),
        rng=dict(python=r.python,numpy=r.numpy,cpu=r.cpu,cuda=r.cuda,task=r.task_rng))
    torch.save(history,path/'physical-prefix.pt')
    counts=dict(native_transitions=c.transitions,high_level_calls=c.decisions,physics_ticks=c.physics,
        native_source_controls=c.source_controls,native_source_physics=c.source_physics,episodes=c.episodes,
        agent_updates=updates,coverage=c.coverage,source_failures=c.source_failures)
    write(path/'manifest.json',dict(version='DUAL_TRACK_FB_FULL_V1',extension=EXTENSION_VERSION,runtime=runtime,
        base_sha256=BASE_SHA,model_sha256=sha(path/'model.pt'),config=agent.cfg.model_dump(),
        replay_capacity=c.replay.capacity,replay_len=len(c.replay),counts=counts,
        original_normalizers_frozen=True,auxiliary_reward_normalizer='identity',
        action_contract=c.task.c,
        resume_contract='replay actual low-level actions through original physics; never recompute old actions using updated actor',
        completed_files={str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file()},save_s=time.monotonic()-begun))
    assert state.digest()==state.fingerprint
    return counts


def load_package(path,e,m,contract,*,physical,existing=None):
    from humanoidverse.agents.buffers.transition import DictBuffer
    meta=json.loads((path/'manifest.json').read_text())
    assert meta['version']=='DUAL_TRACK_FB_FULL_V1' and meta['extension']==EXTENSION_VERSION
    for name,h in meta['completed_files'].items():assert sha(path/name)==h
    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
    from humanoidverse.agents.envs.utils.gym_spaces import json_to_space
    from dual_track_fb_extension import extend_model_in_place
    cfg=FBcprAuxAgentConfig(**meta['config']).model_copy(update={'compile':False,'cudagraphs':False})
    kw=json.loads((path/'original-init-kwargs.json').read_text())
    agent=cfg.build(json_to_space(kw['obs_space']),kw['action_dim'])
    extend_model_in_place(agent._model);agent.obs_space=agent._model.obs_space
    agent.setup_training();agent._model._aux_reward_normalizer=torch.nn.Identity()
    agent._model.load_state_dict(torch.load(path/'model.pt',map_location='cuda',weights_only=True),strict=True)
    opts=torch.load(path/'optimizers.pt',map_location='cpu',weights_only=False)
    for name,sd in opts.items():getattr(agent,name).load_state_dict(sd)
    z=torch.load(path/'zbuffer.pt',map_location='cpu',weights_only=True)
    agent.z_buffer._storage.copy_(z['storage']);agent.z_buffer._idx=z['idx'];agent.z_buffer._is_full=z['full']
    freeze_robot_normalizers(agent._model)
    # Never install a second Task callback graph on a simulator already owned by
    # another Task. Reuse that exact collector for the in-process smoke resume.
    c=existing if existing is not None else Collector(e,m,contract,agent,meta['replay_capacity'])
    c.proxy.policy_model=agent._model
    c.sequences=torch.load(path/'native-sequences.pt',map_location='cpu',weights_only=False)
    if meta['replay_len']:c.replay=DictBuffer.load(path/'replay',device='cpu')
    v=meta['counts'];c.transitions=v['native_transitions'];c.decisions=v['high_level_calls'];c.physics=v['physics_ticks']
    c.source_controls=v['native_source_controls'];c.source_physics=v['native_source_physics'];c.episodes=v['episodes']
    c.coverage=v['coverage'];c.source_failures=v['source_failures']
    if physical:
        hist=torch.load(path/'physical-prefix.pt',map_location='cpu',weights_only=False)
        c.proxy.begin_source();c.task.reset(hist['spec'],record=True,label='restored-fb');c.proxy.finish_pending();c.proxy.begin_online()
        # Override ONLY with the exact recorded29D executed actions for physical
        # reconstruction. No such segment is added to online training data.
        original_act=c.proxy.act;actions=iter(hist['actual_native_actions']);played=[]
        def recorded(obs,z,mean=True):
            a=torch.tensor(next(actions),device=z.device,dtype=torch.float32)
            played.append(a);return a
        c.proxy.act=recorded
        try:
            for raw,limit in zip(hist['raw'],hist['native_limits']):
                c.task.step(torch.tensor(raw,device=e.standing.device,dtype=torch.float32)[None],native_limit=limit)
            try:next(actions);raise AssertionError('Unplayed archived action')
            except StopIteration:pass
        finally:c.proxy.act=original_act
        c.native_actions=hist['actual_native_actions'];c.task.physics_rng=hist['physics_rng']
        check=TrainingBranch(c.task)
        assert check.fingerprint==hist['digest'],'Mutable-actor physical/history resume mismatch'
        r=hist['rng'];restore_global_rng(dict(python=r['python'],numpy=r['numpy'],cpu=r['cpu'],cuda=r['cuda']))
        c.task.rng.bit_generator.state=copy.deepcopy(r['task'])
    return agent,c,meta


def run_updates(agent,c,first,count,compile=False):
    if compile and not agent.cfg.compile:
        agent.cfg=agent.cfg.model_copy(update={'compile':True});agent.setup_compile()
    times=[];last={}
    for i in range(first,first+count):
        t=time.monotonic()
        # m10 physics defaults grad mode off; training is explicitly enabled.
        with torch.enable_grad():metrics=agent.update(c.buffers,i)
        torch.cuda.synchronize();times.append(time.monotonic()-t)
        assert all(torch.isfinite(v).all() for v in metrics.values() if torch.is_tensor(v))
        freeze_robot_normalizers(agent._model)
        last={k:float(v) for k,v in metrics.items() if torch.is_tensor(v) and v.numel()==1}
    return dict(updates=count,wall_s=sum(times),per_update_s=times,last_metrics=last)


def actor_gradient_terms(agent,c):
    """Same-state64-row diagnostic only; does not update or change any RNG."""
    saved=global_rng();torch.manual_seed(560204)
    try:
        sample=c.replay.sample(64)
        obs={k:v.to('cuda') for k,v in sample['observation'].items()};z=sample['z'].to('cuda')
        obs=agent._model._obs_normalizer(obs)
        with torch.enable_grad(),torch.autocast(device_type='cuda',dtype=agent._model.amp_dtype,enabled=agent.cfg.model.amp):
            action=agent._model._actor(obs,z,agent._model.cfg.actor_std).sample(clip=agent.cfg.train.stddev_clip)
            qd=agent.get_targets_uncertainty(agent._model._critic(obs,z,action),agent.cfg.train.actor_pessimism_penalty)[2]
            qa=agent.get_targets_uncertainty(agent._model._aux_critic(obs,z,action),agent.cfg.train.actor_pessimism_penalty)[2]
            qf=agent.get_targets_uncertainty((agent._model._forward_map(obs,z,action)*z).sum(-1),agent.cfg.train.actor_pessimism_penalty)[2]
            w=qf.abs().mean().detach() if agent.cfg.train.scale_reg else 1.
            terms=dict(CPR=-qd.mean()*agent.cfg.train.reg_coeff*w,
                auxiliary=-qa.mean()*agent.cfg.train.reg_coeff_aux*w,FB=-qf.mean())
            params=tuple(agent._model._actor.parameters());norms={}
            for key,term in terms.items():
                grad=torch.autograd.grad(term,params,retain_graph=True)
                norms[key]=float(sum(v.double().square().sum() for v in grad).sqrt())
        return dict(gradient_norms=norms,loss_terms={k:float(v.detach()) for k,v in terms.items()},
            sample='fixed64 actual online states and their executed z; diagnostic, not the relabeled training batch',optimizer_steps=0)
    finally:restore_global_rng(saved)


def run(e,m,contract,out,args):
    if args.mode=='fb_eval':return evaluate(e,m,contract,out,args)
    assert args.mode in ('fb_smoke','fb_train')
    if args.resume:
        assert args.mode=='fb_train'
        agent,c,prior=load_package(args.resume,e,m,contract,physical=True)
        manifest=dict(resumed_from=str(args.resume),parent_model_sha256=prior['model_sha256'])
    else:
        agent,manifest=factory(compile=False)
        c=Collector(e,m,contract,agent,2048 if args.mode=='fb_smoke' else CFG['fb']['replay_capacity'])
        c.bootstrap_sources()
    write(out/'initialization.json',dict(migration=manifest,config=agent.cfg.model_dump(),
        source_counts=c.sequences.seen,formal_training=args.mode=='fb_train',fresh_replay=len(c.replay),fresh_Adam=True))
    if args.mode=='fb_smoke':
        collected=c.collect(256)
        probes={name:next(getattr(agent._model,name).parameters()).detach().clone() for name in
            ('_actor','_forward_map','_backward_map','_critic','_aux_critic','_discriminator')}
        updates=run_updates(agent,c,0,2)
        compiled=run_updates(agent,c,2,2,compile=True)
        assert all(len(v['state'])>0 for v in agent.optimizer_dict.values())
        changes={k:float((next(getattr(agent._model,k).parameters())-v).abs().max()) for k,v in probes.items()}
        assert all(v>0 for v in changes.values())
        gradient=actor_gradient_terms(agent,c)
        t=time.monotonic();save_package(out/'disposable-package',agent,c,4,args.runtime)
        expected_replay=c.replay
        restored,rc,meta=load_package(out/'disposable-package',e,m,contract,physical=True,existing=c)
        assert equal(agent._model.state_dict(),restored._model.state_dict())
        assert equal(agent.optimizer_dict,restored.optimizer_dict)
        assert equal(expected_replay.get_full_buffer(),rc.replay.get_full_buffer())
        torch.cuda.synchronize()
        result=dict(status='completed',engineering_online_transitions=256,formal_FB_transitions=0,
            collection=collected,eager_updates=updates,compiled_updates=compiled,parameter_changes=changes,save_and_read_s=time.monotonic()-t,
            actor_gradient_terms=gradient,
            model_and_six_optimizers_exact=True,replay_readback_exact=True,actual_action_physics_resume_exact=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write(out/'result.json',result);return result
    total=CFG['fb']['native_transitions'];updates=prior['counts']['agent_updates'] if args.resume else 0;logs=[]
    c.output=out
    save_package(out/(f'resume{c.transitions}' if args.resume else 'step0'),agent,c,updates,args.runtime)
    start=time.monotonic()
    while c.transitions<total:
        old_block=c.transitions//1024
        target=min(total,(old_block+1)*1024)
        # Exact0/250k/1M boundaries without falsely dropping or padding a control.
        if c.transitions<250000<target:target=250000
        remaining=target-c.transitions
        collected=c.collect(remaining)
        new_block=c.transitions//1024
        nupdates=max(0,new_block-max(1,old_block))*CFG['fb']['updates_per_collection']
        stats=run_updates(agent,c,updates,nupdates,compile=CFG['fb']['compile']) if nupdates else dict(updates=0,wall_s=0,per_update_s=[])
        updates+=nupdates
        row=dict(native_transitions=c.transitions,agent_updates=updates,collection=collected,update=stats,
            wall_s=time.monotonic()-start,coverage=copy.deepcopy(c.coverage),
            controlled_physics_ticks=c.physics,source_controls=c.source_controls,source_physics=c.source_physics,
            replay_len=len(c.replay),episodes=c.episodes)
        logs.append(row);write(out/'learning.json',logs)
        write(out/'episodes.json',c.episode_rows)
        print('FORMAL_FB',c.transitions,updates,collected['wall_s'],stats['wall_s'],flush=True)
        if c.transitions in (250000,total):
            write(out/f'actor-gradient-{c.transitions}.json',actor_gradient_terms(agent,c))
            save_package(out/f'step{c.transitions}',agent,c,updates,args.runtime)
        assert c.transitions<=total
        free=shutil.disk_usage(out).free
        assert free>=CFG['minimum_shared_free_bytes'],'Shared storage reserve reached; pause with evidence'
        used=sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())
        assert used<=CFG['engineering_output_ceiling_bytes'],'Approved stage output ceiling reached'
    assert c.transitions==1000000
    return dict(status='completed',actual_native_transitions=c.transitions,agent_updates=updates,
        source_controls=c.source_controls,source_physics=c.source_physics,Student_RL_decisions=0)


def evaluate(e,m,contract,out,args):
    from dual_track_quality import quality
    from native_transfer import phase_report
    agent,c,meta=load_package(args.checkpoint,e,m,contract,physical=False)
    task=c.task;c.proxy.stochastic=False
    rows=[]
    for group in parent_manifests()[args.split]:
        c.reset(group['spec'])
        while not task.done:
            rows_unused,_,_=c.proxy.take_step(task.native(task.spec['source'],float(e.tether.time))/16)
        rows.append(dict(parent_id=group['parent_id'],source=group['source'],family=group['family'],
            boundary=task.boundary,quality=quality(task),**phase_report(task)))
        write(out/'summaries.json',rows)
    return dict(status='completed',rows=len(rows),formal_native_training_transitions=0,model_sha256=meta['model_sha256'])
