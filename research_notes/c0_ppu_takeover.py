"""Issue47 backend-paired interventions; original NVIDIA mismatch retained separately."""
import os,copy,json,gzip,time,hashlib
from pathlib import Path
import transfer_student_validated as runner

def study(env,model,contract,out):
    import numpy as np,torch,mujoco
    from clean_policy import Expert,direct_command
    from geometry_policy import expand
    from transition_env import Task,install_frame
    from locomotion_rollout import spherical
    from m11_runtime_speed import install
    import c0_ppu_backend
    c0_ppu_backend.install(out)
    install_frame()
    import geometry_env,reach_contact
    geometry_env.install_contact=reach_contact.install
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    runner.write(out/'runtime-speed.json',install(env))
    cfg=json.loads(Path(runner.ARGS.resume).read_text());task=Task(env,model,contract,4301)
    nets={};identities={};baselines={};rows=[];queries=[]
    for name in ['I','C0']:
        identity=cfg['models'][name];assert runner.sha(identity['path'])==identity['sha256']
        ck=torch.load(identity['path'],map_location=env.standing.device,weights_only=True)
        net=Expert(1088,'direct').to(env.standing.device);expand(net,ck);nets[name]=net.eval().requires_grad_(False)
        identities[name]=dict(**identity,checkpoint_runtime=ck.get('runtime'),std=float(net.log_std.exp().mean()))
    runner.write(out/'identities.json',identities)
    def plain(v):
        if torch.is_tensor(v):return v.detach().cpu().tolist()
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)) or type(v).__name__=='deque':return [plain(x) for x in v]
        return v
    def digest(v):return hashlib.sha256(json.dumps(plain(v),sort_keys=True,allow_nan=False).encode()).hexdigest()
    def snapshot():
        d=env.sim.data;mask=mujoco.mjtState.mjSTATE_INTEGRATION
        state=np.empty(mujoco.mj_stateSize(env.sim.model,mask));mujoco.mj_getState(env.sim.model,d,state,mask)
        blocks=dict(integration=state,solver=[d.qacc_warmstart,d.qacc,d.qfrc_constraint,d.ctrl,d.xfrc_applied],
          bfm=env.obs,previous_action=env.previous_action,last_z=task.last_command,last_qpd=task.lastq,
          prevz=task.prevz,raw=task.raw,force_history=list(task.h.rows),contact_history=list(task.contact_history),
          input=task.features(),support=env.last_support,memory=[task.alpha,task.phase,task.phase_rate,task.tick,task.prepared],
          tether=dict(time=env.tether.time,force=env.tether.force,anchor=env.tether.anchor,active=env.tether.active,rows=env.tether.geometry_rows),
          physics_prefix=task.physics,control_prefix=task.controls,source_prefix=task.prefix_data)
        return {k:digest(v) for k,v in blocks.items()}
    def query(label,t):
        x=task.features();prev=task.last_command.clone();obs={k:v.clone() for k,v in env.obs.items()}
        raw={k:n.distribution(x).mean for k,n in nets.items()};z={k:direct_command(v,env.standing) for k,v in raw.items()}
        q={k:torch.stack([task.qof(model.act(obs,spherical(prev,v,(j+1)/5),mean=True))[0] for j in range(5)]) for k,v in z.items()}
        queries.append(dict(prefix=label,time=t,raw_rms_difference=float((raw['I']-raw['C0']).square().mean().sqrt()),
          z_angle_deg=float(torch.rad2deg(torch.acos(((z['I']*z['C0']).sum()/256).clamp(-1,1)))),
          qpd_five_substep_joint_difference_deg=torch.rad2deg(q['I']-q['C0']).cpu().tolist(),
          scope='same real state and previous z, fixed observed input; not five physical steps'))
    def run(condition,prefix,suffix=None,switch=None,fixed_from=None,force_off=False):
        if time.monotonic()-runner.START>runner.ARGS.max_seconds-45:raise TimeoutError('stage time budget')
        label=f'{condition}__{prefix}' if suffix is None else f'{condition}__{prefix}_to_{suffix}_{switch:.1f}'
        if fixed_from is not None:label+=f'_fixedz_{fixed_from:.1f}'
        if force_off:label+='_forceoff'
        spec=copy.deepcopy(cfg['diagnostic_specs'][condition])
        task.reset(spec,record=True,label=label)
        snaps={};intervention=None;marks=cfg['switch_times'][condition];fixed=None
        while not task.done:
            t=float(env.tether.time);tick=round((t-4)*10);x=task.features()
            marked=any(tick==round((v-4)*10) for v in marks)
            if marked:
                snaps[str(tick)]=snapshot()
                if suffix is None:query(label,t)
            if suffix is not None and tick==round((switch-4)*10):
                original=baselines[(condition,prefix)]['snapshots'][str(tick)]
                intervention=dict(time=t,sections_equal={k:snaps[str(tick)][k]==v for k,v in original.items()})
                assert all(intervention['sections_equal'].values()),intervention
            selected=suffix if suffix is not None and tick>=round((switch-4)*10) else prefix
            raw=nets[selected].distribution(x).mean
            if fixed_from is not None and t>=fixed_from-1e-7:
                if fixed is None:
                    original=baselines[(condition,prefix)]['snapshots'][str(tick)]
                    intervention=dict(time=t,sections_equal={k:snaps[str(tick)][k]==v for k,v in original.items()})
                    assert all(intervention['sections_equal'].values()),intervention
                    fixed=task.last_command.detach().clone()
                    if force_off:task.spec['physical_force_off']=True
                raw=fixed/16
            task.step(raw)
        from transition_metrics import extra
        row=task.save(out);row.update(extra(task));row.update(prefix=prefix,suffix=suffix,switch=switch,intervention=intervention,fixed_from=fixed_from,force_off=force_off)
        if suffix is not None:assert intervention is not None
        if suffix is None and fixed_from is None and not force_off:
            historical=Path(cfg['historical_physics'][condition][prefix]);old=json.load(gzip.open(historical,'rt'))
            parity={k:float(np.max(np.abs(np.array([r[k] for r in task.physics])-np.array([r[k] for r in old['physics']])))) for k in ['qpos','qvel','force']} if len(old['physics'])==len(task.physics) else {'length_mismatch':len(task.physics)-len(old['physics'])}
            row['historical_parity']=dict(path=str(historical),max_abs_difference=parity,exact=False,scope='cross-backend descriptive, not an intervention-prefix check')
            baselines[(condition,prefix)]=dict(snapshots=snaps,row=row)
        rows.append(row);runner.write(out/'summaries.json',rows)
        runner.write(out/'prefix-parity.json',{str(k):v['snapshots'] for k,v in baselines.items()});runner.write(out/'same-state.json',queries)
        print('C0_ENGAGEMENT',label,row['error_cm'],row['force_rms'],row['boundary'],flush=True)
        if row['boundary']:raise RuntimeError('physical boundary; preserve and stop '+label)
    conditions=list(cfg['diagnostic_specs'])
    if os.environ.get('C0_PPU_QUALIFY_ONLY')=='1':
        for name in ['long72','local6']:
            for prefix in ['C0','I']:run(name,prefix)
    else:
        for name in conditions:
            for prefix in ['C0','I']:run(name,prefix)
            for t in cfg['switch_times'][name]:run(name,'C0','I',t);run(name,'I','C0',t)
        # Four bounded causal controls: same C0 command held vs still-feedback BFM,
        # true physical contact vs no applied force. Not a passive robot baseline.
        for name in ['long72','local6']:
            run(name,'C0',fixed_from=6.)
            run(name,'C0',fixed_from=6.,force_off=True)
    return dict(status='completed',episodes=len(rows),rl_training_decisions=0,supervised_updates=0,elapsed_s=time.monotonic()-runner.START)

runner.study=study
if __name__=='__main__':runner.main()
