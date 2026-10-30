"""CPU metadata/sample audit only. Never loads a BFM or advances simulation."""
import argparse,hashlib,json
from pathlib import Path
import h5py
import numpy as np


def audit(folder, output):
    folder=Path(folder);output=Path(output)
    cfg=json.loads((folder/'config.json').read_text())
    result={'config':cfg,'datasets':{},'model_forwards':0,'simulation_steps':0}
    with h5py.File(folder/'buffer.hdf5','r') as f:
        for name,d in f.items():
            assert isinstance(d,h5py.Dataset),name
            result['datasets'][name]={'shape':list(d.shape),'dtype':str(d.dtype),'compression':d.compression,'logical_bytes':d.size*d.dtype.itemsize}
        result['logical_bytes']=sum(d['logical_bytes'] for d in result['datasets'].values())
        expected={'observation-state':64,'observation-privileged_state':463,'observation-last_action':29,'observation-history_actor':372,'action':29,'qpos':36,'qvel':35}
        for key,n in expected.items():
            assert key in f,('Missing source contract field',key)
            assert f[key].ndim==3 and f[key].shape[-1]==n,(key,f[key].shape,n)
        for key in ['truncated','terminated']:
            assert key in f and f[key].shape[-1]==1,key
        assert cfg['n_dim']==2 and cfg['seq_length']==1
        cap,n_envs=f['qpos'].shape[:2]
        assert cap==cfg['capacity'] and cfg['_is_full'] is True
        assert 0<=cfg['_idx']<cap
        rng=np.random.default_rng(20261030)
        indices=np.column_stack((rng.integers(0,cap,32),rng.integers(0,n_envs,32)))
        arrays={key:np.stack([f[key][int(t),int(e)] for t,e in indices]) for key in expected}
        assert all(np.isfinite(x).all() for x in arrays.values())
        q=arrays['qpos'];v=arrays['qvel'];obs=arrays['observation-state']
        default=np.array([-.1,0,0,.3,-.2,0]*2+[0]*17)
        w,x,y,z=q[:,3:7].T
        gravity=-np.column_stack((2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)))
        error=lambda a,b:{'rms':float(np.sqrt(np.mean((a-b)**2))),'max_abs':float(np.max(abs(a-b)))}
        result['sample_checks']={
            'rows':32,'seed':20261030,'indices':indices.tolist(),
            'quaternion_norm_max_error':float(np.max(abs(np.linalg.norm(q[:,3:7],axis=1)-1))),
            'state_position_vs_qpos_minus_default':error(obs[:,:29],q[:,7:]-default),
            'state_velocity_vs_qvel':error(obs[:,29:58],v[:,6:]),
            'state_gravity_vs_wxyz_qpos':error(obs[:,58:61],gravity),
            'state_angular_velocity_vs_qvel_quarter':error(obs[:,61:64],v[:,3:6]*.25),
            'note':'Diagnostic residuals only: source training has observation noise and DR; do not relabel noisy records as clean observations or silently regenerate them.'}
        result['source_contract_shapes_passed']=True
    output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k!='sample_checks'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('output');a=p.parse_args()
    audit(a.folder,a.output)
