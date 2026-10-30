"""Synthetic tests of original sampler and reward aggregation; no real model."""
import types,unittest,json,tempfile,h5py
from pathlib import Path
import numpy as np
import torch
from humanoidverse.agents.buffers.trajectory import find_start_stop_traj,get_idxs,TrajectoryDictBufferMultiDim
from humanoidverse.agents.fb.model import FBModel,FBModelConfig
from humanoidverse.agents.buffers.transition import DictBuffer
from reward_stage_contract import load_buffer_cpu,with_inference_batch,reward_physical_successor,LEGACY_BUFFER_SHA


class MockModel:
    device='cpu'
    amp_dtype=torch.bfloat16
    cfg=types.SimpleNamespace(amp=False,inference_batch_size=7,archi=types.SimpleNamespace(norm_z=True))
    reward_inference=FBModel.reward_inference
    reward_wr_inference=FBModel.reward_wr_inference
    project_z=FBModel.project_z
    def backward_map(self,obs):return obs['B']


class ContractTests(unittest.TestCase):
    def test_real_loader_sampler_and_legacy_pair_on_synthetic_hdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg={'device':'cuda','capacity':9,'n_dim':2,'seq_length':1,
                 'nested_key_separator':'-','end_key':'truncated',
                 'output_key_t':['action','qpos','qvel'],
                 'output_key_tp1':['observation','terminated'],
                 '_idx':7,'_is_full':True,'__target__':'original'}
            Path(tmp,'config.json').write_text(json.dumps(cfg))
            t=np.broadcast_to(np.arange(9,dtype=np.float32)[:,None,None],(9,2,1))
            done=np.zeros((9,2,1),dtype=bool);done[2,0]=True;done[4,1]=True
            with h5py.File(Path(tmp,'buffer.hdf5'),'w') as f:
                for k,v in {'observation-state':t,'action':t,
                            'qpos':(t+1)%9,'qvel':10*((t+1)%9),
                            'truncated':done,'terminated':np.zeros_like(done)}.items():
                    f.create_dataset(k,data=v)
            buf=load_buffer_cpu(tmp,TrajectoryDictBufferMultiDim)
            buf._get_idxs=get_idxs
            torch.manual_seed(20261030);data=buf.sample(1000)
            q,v=reward_physical_successor(data,LEGACY_BUFFER_SHA)
            self.assertTrue(torch.equal(q,data['next']['observation']['state']))
            self.assertTrue(torch.equal(v,10*q))
            self.assertEqual(q.device.type,'cpu')
            self.assertEqual(buf._idx,7);self.assertTrue(buf._is_full)
    def test_legacy_info_alias_and_physical_successor_pair(self):
        extras={'qpos':torch.tensor([[1.]]),'qvel':torch.tensor([[10.]])}
        info=extras
        copied_info=dict(extras)
        # Exact old mutation pattern: next env.step returns persistent extras.
        new_info=extras
        new_info['qpos']=torch.tensor([[2.]])
        new_info['qvel']=torch.tensor([[20.]])
        self.assertEqual(float(info['qpos'][0,0]),2.)
        self.assertEqual(float(copied_info['qpos'][0,0]),1.)
        data={'qpos':info['qpos'],'qvel':info['qvel'],
              'next':{'qpos':torch.tensor([[3.]]),'qvel':torch.tensor([[30.]])}}
        q,v=reward_physical_successor(data,LEGACY_BUFFER_SHA)
        self.assertEqual(float(q[0,0]),2.);self.assertEqual(float(v[0,0]),20.)
        with self.assertRaises(ValueError):reward_physical_successor(data,'another-buffer')
    def test_frozen_config_copy_changes_only_batch(self):
        old=FBModelConfig();new=with_inference_batch(old,4096)
        a=old.model_dump();b=new.model_dump();a['inference_batch_size']=4096
        self.assertEqual(a,b);self.assertEqual(old.inference_batch_size,500000)
    def test_original_loader_device_argument_and_adapter(self):
        class NoAllocationBuffer:
            load=classmethod(DictBuffer.load.__func__)
            def __init__(self,**kw):self.__dict__.update(kw)
            def load_hdf5(self,p):self.loaded_path=str(p)
        with tempfile.TemporaryDirectory() as tmp:
            cfg={'device':'cuda','capacity':5000,'_idx':1250,'_is_full':True,'__target__':'original'}
            Path(tmp,'config.json').write_text(json.dumps(cfg))
            original=NoAllocationBuffer.load(tmp,device='cpu')
            fixed=load_buffer_cpu(tmp,NoAllocationBuffer)
            self.assertEqual(original.device,'cuda')
            self.assertEqual(fixed.device,'cpu')
            self.assertEqual(fixed._idx,1250);self.assertTrue(fixed._is_full)
            self.assertEqual(json.loads(Path(tmp,'config.json').read_text()),cfg)
    def test_ring_boundaries_and_repeatability(self):
        done=torch.zeros(9,2,dtype=torch.bool);done[2,0]=True;done[6,0]=True;done[4,1]=True
        start,stop,length=find_start_stop_traj(done,at_capacity=True,cursor=7)
        valid=length>=2
        def draw():
            torch.manual_seed(20261030)
            return get_idxs(1,1000,length[valid],start[valid],9,None,1)
        a=draw();self.assertTrue(torch.equal(a,draw()))
        marked=done.clone();marked[7]=True
        self.assertFalse(marked[a[:,0],a[:,1]].any())
        self.assertTrue(((a[:,0]>=0)&(a[:,0]<9)).all())
        self.assertTrue((a[:,1]<2).all())
    def test_global_weighting_and_chunk_equivalence(self):
        torch.manual_seed(123)
        B=torch.randn(23,256);r=torch.linspace(0,1,23)[:,None];m=MockModel()
        actual=m.reward_wr_inference({'B':B},r)
        expected=m.project_z((r*torch.softmax(10*r,dim=0)).T@B)
        self.assertTrue(torch.allclose(actual,expected,atol=2e-6,rtol=2e-6))
        self.assertAlmostEqual(float(actual.norm()),16.,places=5)
        m.cfg=types.SimpleNamespace(amp=False,inference_batch_size=23,archi=types.SimpleNamespace(norm_z=True))
        self.assertTrue(torch.allclose(actual,m.reward_wr_inference({'B':B},r),atol=2e-6,rtol=2e-6))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
