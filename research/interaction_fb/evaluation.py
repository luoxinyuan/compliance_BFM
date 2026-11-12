"""Independent-process deterministic evaluation; no optimizer or training Task."""
import math,json,gzip,copy
from pathlib import Path
import numpy as np
import torch
from native_transfer import phase_report
from retention_eval import phase_quality
from f1_sequence_search import semantics,install_post_velocity

def summarize(t):
    r=phase_report(t);r['phase_quality']=phase_quality(t)
    ph=t.physics
    assert all(len(p['f1_post_contacts'])==len(p['stage_foot_contacts'])==len(p['stage_foot_contact_details']) for p in ph)
    active=[p for p in ph if p['active']]
    loaded=[p['stage_loaded_slip_mps'] for p in ph]
    yaw=abs(math.degrees(math.atan2(ph[-1]['pelvis_matrix'][1][0],ph[-1]['pelvis_matrix'][0][0])))
    r.update(boundary=t.boundary,loaded_slip_peak=max(loaded),loaded_slip_p90=float(np.quantile(loaded,.9)),
        yaw_abs_final=yaw,slip_semantics=semantics(ph),parent_id=t.spec['parent_id'],family=t.spec['demand_family'])
    err=r['all']['error_cm'];force=r['all']['force_rms'];fam=t.spec['demand_family']
    r['J']=None if err is None else force+.03*err+r['all']['cap_fraction']
    required=4. if fam=='near' else 8.;fr=4. if fam=='near' else 7.
    desired=active[-1]['anchor'][0]-active[0]['anchor'][0] if active else 0.
    progress=active[-1]['wrist'][0]-active[0]['wrist'][0] if active else 0.
    r.update(partner_progress_m=desired,wrist_progress_m=progress,progress_pass=progress>=desired-required/100)
    good=err is not None and err<=required and force<=fr and max(loaded)<=.45 and yaw<=20 and t.boundary is None and r['progress_pass']
    if fam=='stop':good=good and 'holding' in r and r['holding']['error_cm']<=8 and r['holding']['force_rms']<=6.5
    r['clean_success']=bool(good) if fam not in ('zero','none') else None
    r['no_contact_force_zero']=all(np.array(p['force']).tolist()==[0,0,0] for p in ph) if fam=='none' else None
    # Exact material-point transport from qpos on an independent MjData; no dynamics.
    import mujoco
    d=mujoco.MjData(t.e.sim.model);poses=[];ids={n:mujoco.mj_name2id(t.e.sim.model,mujoco.mjtObj.mjOBJ_BODY,n) for n in ('left_ankle_roll_link','right_ankle_roll_link')}
    for p in ph:
        d.qpos[:]=p['qpos'];mujoco.mj_kinematics(t.e.sim.model,d)
        poses.append({n:(d.xpos[i].copy(),d.xmat[i].reshape(3,3).copy()) for n,i in ids.items()})
    travel={k:0. for k in ids}
    for j,p in enumerate(ph[:-1]):
        inc={}
        for c,detail in zip(p['stage_foot_contacts'],p['stage_foot_contact_details']):
            if c['normal_N']<=1e-6:continue
            n=c['body'];x0,R0=poses[j][n];x1,R1=poses[j+1][n];local=np.array(detail['body_local_point'])
            inc[n]=max(inc.get(n,0),float(np.linalg.norm(((x1+R1@local)-(x0+R0@local))[:2])))
        for n,v in inc.items():travel[n]+=v
    r['loaded_material_point_xy_path_m']=travel
    return r

def run(env,agent,out,specs):
    assert not any(s.get('split')=='sealed' for s in specs)
    env.specs=specs;env.proxy.stochastic=False;results=[];prefixes={};probe=[]
    from frl_v3_counterfactual import install_contact_details
    install_contact_details(env.task.e);install_post_velocity(env.task.e)
    for i,s in enumerate(specs):
        torch.manual_seed(626299);np.random.seed(626299)
        env.reset(i);env.proxy.agent=agent;row_start=len(env.rows)
        while not env.task.done:
            try:
                with torch.no_grad():env.collect_call()
            except RuntimeError as exc:
                if not str(exc).startswith('HARD_PHYSICS_BOUNDARY:'):raise
                assert env.task.boundary is not None
                break
        t=env.task;r=summarize(t);r['source_state_sha']=t.prefix_data['audit']['sha256'];r['spec']=s
        # Inputs/actions before the partner branch point must match exactly.
        if 'pair_branch_at' in s:
            pre=[c for c in t.controls if c['time']<s['pair_branch_at']-1e-8]
            selected=[{k:c[k] for k in ('bfm_obs','executed_z','action','q_PD','pre_qpos','pre_qvel')} for c in pre]
            key=s['parent_id'];value=json.dumps(selected,sort_keys=True)
            if key in prefixes:assert value==prefixes[key],'stop/continue prebranch differs'
            else:prefixes[key]=value
        if agent.arm=='C':
            from .schema import robot,interaction,to_device
            rr=env.rows[row_start:]
            for j in range(0,len(rr)-50,5):
                o=rr[j]['observation'];labels={}
                for steps in (25,50):
                    future=rr[j+steps]['observation']['contact_privileged_v1'][0]
                    on=bool(future[21]);ef=float(torch.linalg.norm(future[9:12])*.25);ff=float(torch.linalg.norm(future[6:9])*5)
                    labels[str(steps/50)]=dict(active=int(on),quality=int(ef<=.08 and ff<=7.) if on else None)
                with torch.no_grad():
                    dev=to_device(o);br=agent.base.backward_map(robot(dev)).cpu();bi=(32**.5*torch.nn.functional.normalize(agent.nets['bi'](interaction(dev)),dim=-1)).cpu()
                probe.append(dict(parent=s['parent_id'],family=s['demand_family'],current_active=int(o['contact_privileged_v1'][0,21]),
                    B_R=br[0].tolist(),B_I=bi[0].tolist(),current_contact=interaction(o)[0].tolist(),labels=labels))
        results.append(r)
        with gzip.open(Path(out)/f'actual-{i}.json.gz','wt') as f:json.dump(dict(spec=s,physics=t.physics,controls=t.controls),f)
        (Path(out)/'evaluation.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    with gzip.open(Path(out)/'probe-features.json.gz','wt') as f:json.dump(probe,f)
    return dict(status='evaluation_completed',evaluation_only=True,episodes=len(results),optimizer_updates=0,common_prefix_checks=len(prefixes),results=results)
