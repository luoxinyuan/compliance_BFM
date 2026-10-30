"""GPU-only native reward inference. Prepare now; run after owner GPU notice.

Requires an external owned-process supervisor with the protocol's RAM/VRAM/wall
budgets. The original actor/backward and reward definitions are not modified.
"""
import argparse,gc,hashlib,json,os,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
R=P/'bridge-v2-owned/runs'
MODEL=R/'issue19-recovery-20261028T045520Z/model-view'
BUFFER=Path('/mnt/data-oss/users/yi.gao/compliance_BFM/native_reward_assets/62b4206d68e026de5e5dc7efb1529bccfb95164c/train')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute-after-owner-gpu-notice',action='store_true')
    a=p.parse_args()
    if not a.execute_after_owner_gpu_notice:
        p.error('Execution paused until owner GPU availability notice; no CPU fallback')
    # Check the supervisor contract before importing or loading a model.
    assert os.environ.get('CUDA_VISIBLE_DEVICES'), 'Supervisor must select one GPU'
    deadline=float(os.environ['BFM_STAGE_DEADLINE_MONOTONIC'])
    assert time.monotonic()<deadline
    assert os.environ.get('BFM_STAGE_SUPERVISED')=='1'
    protocol=json.loads((ROOT/'research_notes/reward_skill_protocol.json').read_text())
    out=a.output;out.mkdir(exist_ok=False,parents=True)
    def save(name,obj):(out/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
    result={'status':'started','policy_updates':0,'tasks':[]}
    save('result.json',result);save('protocol.json',protocol)
    try:
        assert BUFFER.joinpath('buffer.hdf5').stat().st_size==protocol['buffer_bytes']
        assert sha(BUFFER/'buffer.hdf5')==protocol['buffer_sha256']
        assert sha(MODEL/'checkpoint/model/model.safetensors')==protocol['model_sha256']
        import numpy as np,torch,mujoco,joblib
        from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim,get_idxs
        from humanoidverse.envs.g1_env_helper.bench.reward_eval_hv import relabel
        from humanoidverse.envs.g1_env_helper.robot import make_from_name
        from humanoidverse.agents import load_utils
        from frozen_bfm_placement import cpu_load_then_stage,tensor_digest,INFERENCE_MODULES
        from reward_stage_contract import load_buffer_cpu,with_inference_batch,reward_physical_successor
        assert torch.cuda.is_available() and torch.cuda.device_count()==1, 'GPU unavailable; stop rather than use CPU'
        torch.set_num_threads(protocol['resources']['cpu_cores'])
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        print('LOADING_ORIGINAL_BUFFER',flush=True)
        dataset=load_buffer_cpu(BUFFER,TrajectoryDictBufferMultiDim)
        assert dataset._is_full and dataset._idx==1250 and dataset.capacity==5000
        assert 'B' not in dataset.storage, 'Do not use cached embeddings of uncertain model identity'
        # This legacy file stores successor physics in current-row qpos/qvel:
        # verified via source aliasing and read-only observation alignment audit.
        dataset.output_key_t=list(dict.fromkeys(dataset.output_key_t+['qpos','qvel']))
        sample_indices=[]
        def recorded_get_idxs(**kw):
            idx=get_idxs(**kw);sample_indices.append(idx.clone());return idx
        dataset._get_idxs=recorded_get_idxs
        torch.manual_seed(protocol['inference']['seed']);np.random.seed(protocol['inference']['seed'])
        data=dataset.sample(protocol['inference']['num_samples'])
        assert len(sample_indices)==1
        np.save(out/'sample-indices.npy',sample_indices[0].numpy(),allow_pickle=False)
        save('sampling.json',{'seed':protocol['inference']['seed'],'count':len(sample_indices[0]),
             'indices_sha256':sha(out/'sample-indices.npy'),'buffer_size':dataset.size(),
             'cursor':dataset._idx,'capacity':dataset.capacity,'next_index':'(t+1)%capacity, same environment; original valid-trajectory sampling',
             'physical_successor_mapping':'qpos/qvel from sampled row t; raw legacy info alias already stores successor; observations remain row t+1',
             'sampler_execution':'original eager get_idxs; no change of sampling formula'})
        del dataset;gc.collect()
        qpos,qvel=reward_physical_successor(data,protocol['buffer_sha256'])
        qpos=qpos.numpy();qvel=qvel.numpy();action=data['action'].numpy()
        assert qpos.shape==(150000,36) and qvel.shape==(150000,35) and action.shape==(150000,29)
        assert all(torch.isfinite(v).all() for v in data['next']['observation'].values())
        mj_model=mujoco.MjModel.from_xml_path(str(ROOT/'humanoidverse/data/robots/g1/scene_29dof_freebase_noadditional_actuators.xml'))
        reward_values={}
        for task in protocol['tasks']:
            assert time.monotonic()<deadline
            print('RELABEL',task,flush=True)
            reward_values[task]=relabel(mj_model,qpos,qvel,action,make_from_name(task),max_workers=4,process_executor=True)
            np.save(out/(task+'-rewards.npy'),reward_values[task],allow_pickle=False)
        placements=[]
        with cpu_load_then_stage(load_utils,placements):
            model=load_utils.load_model_from_checkpoint_dir(MODEL/'checkpoint',device='cuda')
        assert not model.cfg.amp and model.cfg.archi.z_dim==256 and model.cfg.archi.norm_z
        original_batch=model.cfg.inference_batch_size
        model.cfg=with_inference_batch(model.cfg,protocol['inference']['inference_batch_size'])
        save('placement.json',{'placement':placements,'original_batch_size':original_batch,'operational_batch_size':model.cfg.inference_batch_size})
        frozen={n:tensor_digest(getattr(model,n)) for n in INFERENCE_MODULES}
        z_dict={}
        for task in protocol['tasks']:
            assert time.monotonic()<deadline
            reward=torch.as_tensor(reward_values[task],dtype=torch.float32,device='cpu')
            assert reward.shape==(150000,1) and torch.isfinite(reward).all()
            with torch.inference_mode():z=model.reward_wr_inference(data['next']['observation'],reward)
            assert z.device.type=='cuda' and z.shape==(1,256) and torch.isfinite(z).all()
            assert abs(float(z.norm())-16)<1e-4
            weights=torch.softmax(10*reward,dim=0)
            row={'task':task,'reward_min':float(reward.min()),'reward_mean':float(reward.mean()),'reward_max':float(reward.max()),
                 'softmax_weight_ESS':float(1/weights.square().sum()),'z_norm':float(z.norm()),'z':z.cpu().tolist(),
                 'reward_file_sha256':sha(out/(task+'-rewards.npy'))}
            save(task+'-latent.json',row);z_dict[task]=[z.cpu()];result['tasks'].append(row);save('result.json',result)
            print('LATENT_GENERATED',task,flush=True)
        joblib.dump(z_dict,out/'reward_locomotion.pkl')
        assert frozen=={n:tensor_digest(getattr(model,n)) for n in INFERENCE_MODULES}
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        result.update(status='completed',reward_latent_sha256=sha(out/'reward_locomotion.pkl'),model_sha256=protocol['model_sha256'],buffer_sha256=protocol['buffer_sha256'],bfm_frozen=True,physical_qualification='not yet run')
        save('result.json',result)
    except Exception as e:
        result.update(status='failed',first_boundary={'type':type(e).__name__,'message':str(e)})
        save('result.json',result);raise


if __name__=='__main__':main()
