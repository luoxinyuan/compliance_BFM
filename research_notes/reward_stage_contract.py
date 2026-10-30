"""Operational adapters only: keep source replay on CPU and bound GPU batches."""
import json
from pathlib import Path

LEGACY_BUFFER_SHA = '46eac53f603e45921d25504fcf7d3eea861f2a9f8ea503046855e701493b546b'


def load_buffer_cpu(folder, buffer_class):
    # Original DictBuffer.load accepts device= but does not apply the argument.
    # Keep exact original load_hdf5/sample methods; fix placement in this adapter.
    folder=Path(folder);cfg=json.loads((folder/'config.json').read_text())
    cfg.pop('__target__',None)
    cursor=cfg.pop('_idx');full=cfg.pop('_is_full')
    cfg['device']='cpu'
    buffer=buffer_class(**cfg)
    buffer._idx=cursor;buffer._is_full=full
    buffer.load_hdf5(folder/'buffer.hdf5')
    return buffer


def with_inference_batch(config,batch_size):
    # BaseConfig is frozen: copy one operational field, not weights/architecture.
    return config.model_copy(update={'inference_batch_size':batch_size})


def reward_physical_successor(data, buffer_sha):
    """Correct a proven legacy info-dictionary alias, only for this pinned file.

    Old env.step returned the same extras dictionary and replaced its qpos/qvel
    before train.py stored info. Thus raw row t has obs(t), action(t), but
    qpos/qvel(t+1). Original obs(t+1) sampling stays unchanged. Do not shift
    observations, actions, trajectory boundaries, or the HDF5 itself.
    """
    if buffer_sha != LEGACY_BUFFER_SHA:
        raise ValueError('Legacy qpos/qvel mapping is valid only for audited buffer SHA')
    return data['qpos'], data['qvel']
