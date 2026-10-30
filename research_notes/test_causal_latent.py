import json,math,unittest
from pathlib import Path
import numpy as np
import torch
from contact_projector import constrained
from latent_reachability import gates_from_trace

class Sphere:
    def project_z(self,z):return 16*torch.nn.functional.normalize(z,dim=-1)

class Tests(unittest.TestCase):
    def test_physical_stop_is_not_reported_as_complete(self):
        from causal_run import final_status
        self.assertEqual(final_status(dict(physical_executed=True,episodes=[{'complete':False}])), 'physical-stopped')
        self.assertEqual(final_status(dict(physical_executed=True,episodes=[{'complete':True}])), 'completed')
        self.assertEqual(final_status(dict(physical_executed=False,episodes=[])), 'precheck-stopped')

    def test_gate_reconstructs_causal_release(self):
        ep=dict(initial={'force':[0,0,0]},decisions=[{'time':3+i*.1} for i in range(60)],
          control=[{'tether_time':3+i*.02} for i in range(300)],
          physics=[{'force':[1,0,0] if i<1001 else [0,0,0]} for i in range(1200)])
        gate,_=gates_from_trace(ep)
        np.testing.assert_allclose(gate[:5],0)
        np.testing.assert_allclose(gate[5:255],1)
        np.testing.assert_allclose(gate[255:260],[1,.9,.8,.7,.6],atol=1e-10)
        np.testing.assert_allclose(gate[270:],0,atol=1e-10)

    def test_exact_gate_zero_is_uncontrollable(self):
        torch.manual_seed(0);standing=Sphere().project_z(torch.randn(1,256));raw=torch.randn(1,256,requires_grad=True)
        z,_=constrained(Sphere(),standing,raw,0.)
        self.assertTrue(torch.equal(z,standing));z.sum().backward();self.assertEqual(float(raw.grad.abs().max()),0)

    def test_shared_raw_five_frames_preserves_bound_and_gradient(self):
        torch.manual_seed(1);standing=Sphere().project_z(torch.randn(1,256));raw=torch.randn(1,256,requires_grad=True)
        z,_=constrained(Sphere(),standing,raw.repeat_interleave(5,0),[1,1,.8,.4,0])
        np.testing.assert_allclose(z.detach().norm(dim=-1),16,atol=2e-6)
        self.assertTrue(torch.equal(z[0],z[1]));self.assertTrue(torch.equal(z[-1],standing[0]))
        angle=torch.rad2deg(torch.acos(((z*standing).sum(-1)/256).clamp(-1,1)))
        self.assertLessEqual(float(angle.max()),math.degrees(math.atan(.3))+.001)
        z[:,:20].sum().backward();self.assertGreater(float(raw.grad.norm()),0)

    def test_pinned_windows_do_not_overlap(self):
        p=json.loads((Path(__file__).parent/'causal_protocol.json').read_text())
        indices=[round((t-3)*10) for t in p['selected_times_s']]
        self.assertEqual(len(indices),8);self.assertEqual(len(set(indices)),8)
        self.assertTrue(all(0<=i<60 for i in indices));self.assertEqual(p['policy_updates'],0)

if __name__=='__main__':unittest.main()
