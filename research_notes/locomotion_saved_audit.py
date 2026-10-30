"""Fixed saved-data/CUDA inference audit; no policy updates or physics stepping."""
import gc, hashlib, itertools, json, math, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
OLD=P/'bridge-v2-owned/runs/issue21-reward-stage-20261030A'
MODEL=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view'
BUFFER=Path('/mnt/data-oss/users/yi.gao/compliance_BFM/native_reward_assets/62b4206d68e026de5e5dc7efb1529bccfb95164c/train')


def main():
    assert os.environ.get('BFM_STAGE_SUPERVISED')=='1'
    deadline=float(os.environ['BFM_STAGE_DEADLINE_MONOTONIC'])
    out=Path(os.environ['BFM_RECOVERY_OUTPUT'])
    save=lambda n,v:(out/n).write_text(json.dumps(v,indent=2,allow_nan=False))
    result={'status':'running','model_loaded':False,'physics_steps':0,'policy_updates':0}
    save('result.json',result)
    import numpy as np, torch, mujoco
    from humanoidverse.agents import load_utils
    from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim,get_idxs
    from frozen_bfm_placement import cpu_load_then_stage,tensor_digest,INFERENCE_MODULES
    from reward_stage_contract import load_buffer_cpu,reward_physical_successor
    from humanoidverse.envs.g1_env_helper.rewards import LocomotionReward,get_center_of_mass_linvel,rot2eul,get_xmat
    torch.set_num_threads(8);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    placement=[]
    with cpu_load_then_stage(load_utils,placement):
        model=load_utils.load_model_from_checkpoint_dir(MODEL/'checkpoint',device='cuda')
    frozen={n:tensor_digest(getattr(model,n)) for n in INFERENCE_MODULES}
    save('placement.json',placement);result['model_loaded']=True
    read=lambda f:json.loads(f.read_text())
    contract=read(OLD/'physics-attempt-01/outputs/action-contract.json')
    scale=np.asarray(contract['q_PD_scale'])[0];offset=np.asarray(contract['q_PD_offset'])[0]
    target=lambda a:offset+np.clip(a*5,-contract['clip'],contract['clip'])*scale/5
    keys=model.cfg.archi.actor.input_filter.key
    tensor_obs=lambda o:{k:torch.as_tensor(v,dtype=torch.float32,device='cuda') for k,v in o.items()}
    original_z=torch.tensor(read(OLD/'inference-attempt-01/outputs/move-ego-0-0.3-latent.json')['z'],device='cuda')
    quat=lambda q:math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
    m=mujoco.MjModel.from_xml_path(str(ROOT/'humanoidverse/data/robots/g1/scene_29dof_freebase_noadditional_actuators.xml'));d=mujoco.MjData(m)
    reward_fn=LocomotionReward(move_angle=0,move_speed=.3)
    traces={};replay={};all_normalized=[]
    for name in ['standing','raisearms-l-m','move-ego-0-0.3']:
        ep=read(OLD/'physics-attempt-01/outputs'/(name+'.json'));cs=ep['control'];rs=ep['physics'];traces[name]=ep
        errors=[];last_errors=[];norm=[]
        for i,c in enumerate(cs):
            obs=tensor_obs(c['bfm_obs']);z=torch.tensor(c['executed_z'],device='cuda')
            a=model.act(obs,z,mean=True).cpu().numpy()[0]
            errors.append(float(abs(a-np.asarray(c['action'])).max()))
            if i:last_errors.append(float(abs(np.asarray(c['bfm_obs']['last_action'])[0]-5*np.asarray(cs[i-1]['action'])).max()))
            no=model._normalize(obs);norm.append(torch.cat([no[k] for k in keys],-1).cpu().numpy()[0])
        assert max(errors)<2e-5,(name,max(errors))
        all_normalized.extend(norm)
        vel=[];native_reward=[];heading=[]
        for r in rs:
            d.qpos[:]=r['qpos'];d.qvel[:]=r['qvel'];mujoco.mj_forward(m,d)
            v=get_center_of_mass_linvel(m,d);yaw=rot2eul(get_xmat(m,d,'pelvis'))[-1]
            vel.append([v[0]*np.cos(yaw)+v[1]*np.sin(yaw),-v[0]*np.sin(yaw)+v[1]*np.cos(yaw),np.linalg.norm(v[:2])]);native_reward.append(float(reward_fn.compute(m,d)));heading.append(float(yaw))
        vel=np.asarray(vel);times=np.asarray([r['force_time']-3 for r in rs]);phases={}
        for label,(lo,hi) in {'enter':(0,.75),'execute':(.75,3.25),'exit':(3.25,4),'settle':(4,6)}.items():
            mask=(times>=lo-1e-8)&(times<hi-1e-8)
            phases[label]={'native_torso_subtree_velocity_forward_lateral_speed_mean_mps':vel[mask].mean(0).tolist(),
                'forward_std_mps':float(vel[mask,0].std()),'native_reward_mean':float(np.asarray(native_reward)[mask].mean()),
                'speed_within_native_027_033_fraction':float(((vel[mask,2]>=.27)&(vel[mask,2]<=.33)).mean()),
                'heading_minmax_deg':np.degrees([np.min(np.asarray(heading)[mask]),np.max(np.asarray(heading)[mask])]).tolist()}
        replay[name]={'actor_replay_max_error':max(errors),'last_action_scaled_history_max_error':max(last_errors),
            'actor_dt_minmax':np.asarray(np.diff([c['time'] for c in cs]))[[np.argmin(np.diff([c['time'] for c in cs])),np.argmax(np.diff([c['time'] for c in cs]))]].tolist(),
            'phases':phases,'normalized_abs_p99_max':[float(np.quantile(abs(norm),.99)),float(np.max(abs(norm)))]}
    save('saved_trace_replay.json',replay)
    print('SAVED_ACTOR_AND_NATIVE_REWARD_AUDIT_COMPLETE',flush=True)
    # Exact three-input-block Shapley differences on saved adjacent observations.
    # Hybrid inputs are algebraic diagnostics, not physically valid counterfactuals.
    cs=traces['move-ego-0-0.3']['control'];events=[]
    for label,lo,hi in [('entry',0,.75),('fixed_z',.75,3.25),('exit',3.25,4),('recovery',4,6)]:
        i=max([i for i,c in enumerate(cs) if i and lo-1e-8<=c['relative_time']<hi-1e-8],key=lambda i:cs[i]['q_PD_step_max_rad'])
        a,b=cs[i-1],cs[i];o0=tensor_obs(a['bfm_obs']);o1=tensor_obs(b['bfm_obs']);z=torch.tensor(b['executed_z'],device='cuda')
        vals={}
        for mask in range(8):
            obs=dict(o0)
            for j,k in enumerate(keys):obs[k]=o1[k] if mask&(1<<j) else o0[k]
            vals[mask]=target(model.act(obs,z,mean=True).cpu().numpy()[0])
        contributions={}
        for j,k in enumerate(keys):
            v=np.zeros(29)
            for mask in range(8):
                if mask&(1<<j):continue
                n=mask.bit_count();weight=math.factorial(n)*math.factorial(2-n)/6
                v+=weight*(vals[mask|(1<<j)]-vals[mask])
            contributions[k]=np.degrees(v).tolist()
        assert np.max(abs(sum(np.asarray(v) for v in contributions.values())-np.degrees(vals[7]-vals[0])))<1e-6
        joint=int(np.argmax(abs(np.asarray(b['q_PD'])-a['q_PD'])))
        events.append({'phase':label,'time':b['relative_time'],'joint':contract['dof_names'][joint],
            'actual_target_step_deg':math.degrees(b['q_PD'][joint]-a['q_PD'][joint]),
            'fixed_new_z_observation_contributions_deg':{k:v[joint] for k,v in contributions.items()},
            'fixed_old_obs_z_change_deg':float(np.degrees(vals[0][joint]-a['q_PD'][joint])),
            'current_joint_velocity_rad_s':b['pre_qvel'][joint+6],
            'all_joints_contributions_deg':contributions})
    save('jump_attribution.json',events)
    assert time.monotonic()<deadline
    print('LOADING_SAME_FIXED_REPLAY_BATCH',flush=True)
    dataset=load_buffer_cpu(BUFFER,TrajectoryDictBufferMultiDim)
    dataset.output_key_t=list(dict.fromkeys(dataset.output_key_t+['qpos','qvel']))
    indices=[]
    def record(**kw):
        v=get_idxs(**kw);indices.append(v.clone());return v
    dataset._get_idxs=record;torch.manual_seed(20261030);np.random.seed(20261030)
    data=dataset.sample(150000)
    assert np.array_equal(indices[0].numpy(),np.load(OLD/'inference-attempt-01/outputs/sample-indices.npy'))
    del dataset;gc.collect()
    reward=torch.tensor(np.load(OLD/'inference-attempt-01/outputs/move-ego-0-0.3-rewards.npy'),dtype=torch.float32)
    weights=torch.softmax(10*reward,dim=0);wr=weights*reward
    partial=[torch.zeros((1,256),device='cuda') for _ in range(8)];total=torch.zeros((1,256),device='cuda');weighted_norm=0.;normal=[]
    for start in range(0,150000,4096):
        assert time.monotonic()<deadline
        stop=min(start+4096,150000);obs={k:v[start:stop].to('cuda') for k,v in data['next']['observation'].items()}
        B=model.backward_map(obs);w=wr[start:stop].to('cuda');total+=w.T@B
        weighted_norm+=float((w[:,0]*B.norm(dim=-1)).sum())
        for k in range(8):
            mask=(torch.arange(start,stop,device='cuda')%8)==k;partial[k]+=w[mask].T@B[mask]
        if start==0:
            no=model._normalize(obs);normal=torch.cat([no[k] for k in keys],-1).cpu().numpy()
    z=model.project_z(total);angle=lambda a,b:float(torch.rad2deg(torch.acos(((a*b).sum()/(a.norm()*b.norm())).clamp(-1,1))))
    halves=[sum(partial[::2]),sum(partial[1::2])]
    sensitivity={'same_sample_indices':True,'original_z_reconstruction_max_abs':float(abs(z-original_z).max()),
        'original_z_reconstruction_angle_deg':angle(z,original_z),
        'unprojected_resultant_norm':float(total.norm()),'weighted_B_norm':weighted_norm,
        'resultant_over_weighted_norm':float(total.norm())/weighted_norm,
        'eight_fixed_interleaved_strata_angle_to_full_deg':[angle(v,total) for v in partial],
        'leave_one_stratum_out_angle_to_full_deg':[angle(total-v,total) for v in partial],
        'two_half_angle_deg':angle(*halves),'no_candidate_selected_or_physically_executed':True,
        'replay_normalized_abs_p99_max':[float(np.quantile(abs(normal),.99)),float(abs(normal).max())]}
    qpos,qvel=reward_physical_successor(data,'46eac53f603e45921d25504fcf7d3eea861f2a9f8ea503046855e701493b546b')
    selected=np.arange(0,150000,30);speeds=[];rewcheck=[]
    for i in selected:
        d.qpos[:]=qpos[i];d.qvel[:]=qvel[i];mujoco.mj_forward(m,d)
        v=get_center_of_mass_linvel(m,d);yaw=rot2eul(get_xmat(m,d,'pelvis'))[-1]
        speeds.append([v[0]*np.cos(yaw)+v[1]*np.sin(yaw),-v[0]*np.sin(yaw)+v[1]*np.cos(yaw),np.linalg.norm(v[:2])]);rewcheck.append(reward_fn.compute(m,d))
    speeds=np.asarray(speeds);ww=wr[selected,0].numpy();ww/=ww.sum()
    sensitivity['fixed_5000_record_reward_recompute_max_abs']=float(abs(np.asarray(rewcheck)-reward[selected,0].numpy()).max())
    sensitivity['fixed_5000_weighted_native_forward_lateral_speed_mps']=(ww[:,None]*speeds).sum(0).tolist()
    sensitivity['fixed_5000_weighted_speed_std_mps']=np.sqrt((ww[:,None]*(speeds-(ww[:,None]*speeds).sum(0))**2).sum(0)).tolist()
    save('inference_sensitivity.json',sensitivity)
    assert frozen=={n:tensor_digest(getattr(model,n)) for n in INFERENCE_MODULES}
    result.update(status='completed',original_bfm_unchanged=True,actor_replay=replay,inference_sensitivity=sensitivity)
    save('result.json',result);print('SAVED_DATA_AUDIT_COMPLETE',json.dumps(sensitivity),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        out=Path(os.environ['BFM_RECOVERY_OUTPUT']);(out/'failure.json').write_text(json.dumps({'type':type(e).__name__,'boundary':str(e)},indent=2));raise
