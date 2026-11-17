"""Original motion-reference CPR schema restricted to prelocked train ancestors.

References are kinematic motion data, not additional online transitions or
executed-action labels. B/discriminator read only state/privileged_state.
"""
import copy
import hashlib
import json
from pathlib import Path
import torch
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBuffer,get_idxs


class NativeReference(TrajectoryDictBuffer):
    def __init__(self,episodes):
        super().__init__(episodes=episodes,device='cpu',seq_length=8)
        # Fixed eager execution of the identical original integer sampler.
        self._get_idxs=get_idxs


def build_reference(env,stage,output):
    from humanoidverse.utils.helpers import get_backward_observation
    stage=Path(stage);output=Path(output)
    lock=stage/'locks/source-qualification-v1.json'
    plan=json.loads(lock.read_text());names=[r['ancestor'] for r in plan['cases']]
    assert len(set(names))==12
    held=json.loads((stage/'locks/selected-source-state-manifest-v1.json').read_text())
    assert not set(names).intersection(r['ancestor'] for r in held['states'])
    lib=env._motion_lib;keys=lib._motion_data_keys.tolist()
    old=lib._curr_motion_ids.cpu().tolist();assert len(old)==1
    episodes=[];provenance=[]
    try:
        for j,name in enumerate(names):
            lib.load_motions(random_sample=False,start_idx=keys.index(name),num_motions_to_load=1)
            obs,_=get_backward_observation(env,0,use_root_height_obs=bool(env.config.obs.get('root_height_obs',True)))
            assert isinstance(obs,dict)
            n=len(obs['state']);assert n>=9
            # Official training loader uses zero bogus last_action. It is not
            # passed to B or discriminator; no actor/executed label is forged.
            observation={k:obs[k].detach().cpu().clone() for k in ['state','privileged_state']}
            observation['last_action']=torch.zeros(n,29)
            truncated=torch.zeros(n,dtype=torch.bool);truncated[-1]=True
            episodes.append(dict(observation=observation,terminated=torch.zeros(n,dtype=torch.bool),
                truncated=truncated,motion_id=torch.full((n,),j,dtype=torch.long)))
            provenance.append(dict(ancestor=name,reference_rows=n,reference_duration_s=n*env.dt,
                training_reference_not_actual_physical_rows=True))
    finally:
        lib.load_motions(random_sample=False,start_idx=old[0],num_motions_to_load=1)
    output.mkdir(parents=True,exist_ok=False)
    torch.save(episodes,output/'episodes.pt')
    manifest=dict(protocol='CTFT_NATIVE_CPR_TRAIN12_V1',source_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
        cases=provenance,total_reference_rows=sum(r['reference_rows'] for r in provenance),
        schema='Official motion-loader robot state/privileged_state, zero unused last_action; no fake actor history or contact fields.',
        normalizer_allow_mismatching_keys=True,sequence_length=8,
        episode_sha256=hashlib.sha256((output/'episodes.pt').read_bytes()).hexdigest(),
        no_holdout_reference_training=True,no_external_download=True)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return NativeReference(episodes),manifest


def load_reference(path):
    path=Path(path);m=json.loads((path/'manifest.json').read_text())
    assert hashlib.sha256((path/'episodes.pt').read_bytes()).hexdigest()==m['episode_sha256']
    return NativeReference(torch.load(path/'episodes.pt',map_location='cpu',weights_only=False))
