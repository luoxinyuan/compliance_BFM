import unittest
import numpy as np
import torch
from contact_projector import ContactGate,constrained,deploy_input,Projector
class Frozen:
    def project_z(self,z):return 16*z/z.norm(dim=-1,keepdim=True)
    def _normalize(self,o):return o
class Tests(unittest.TestCase):
    def test_gate_uses_past_force_and_50hz_release(self):
        g=ContactGate();g.update(1.1,[0,0,0]);self.assertEqual(g.value(1.2),0)
        g.update(1.2,[.01,0,0]);self.assertEqual(g.value(1.28),1)
        g.update(4.6,[0,0,0]);self.assertAlmostEqual(g.value(4.62),.9)
        self.assertAlmostEqual(g.value(4.7),.5);self.assertAlmostEqual(g.value(4.8),0)
        self.assertEqual(g.value(5.),0)
    def test_zero_gradient_and_constraints(self):
        torch.manual_seed(0);stand=Frozen().project_z(torch.randn(1,256));u=torch.zeros(1,256,requires_grad=True)
        z,t=constrained(Frozen(),stand,u,1.);(z*torch.randn_like(z)).sum().backward();self.assertGreater(float(u.grad.norm()),0)
        z,t=constrained(Frozen(),stand,torch.randn(8,256)*5,torch.linspace(0,1,8))
        self.assertTrue(torch.allclose(z.norm(dim=-1),torch.ones(8)*16,atol=1e-5))
        self.assertLess(float((t*stand).sum(-1).abs().max()),1e-4)
        self.assertLess(float(torch.rad2deg(torch.acos(((z*stand).sum(-1)/256).clamp(-1,1))).max()),16.7)
        self.assertTrue(torch.equal(z[:1],stand))
    def test_privileged_fields_excluded(self):
        obs={'state':torch.zeros(1,4),'anchor':torch.ones(1,3)*99}
        a=deploy_input(Frozen(),obs,np.ones((3,3)),['state']);obs['anchor']*=100
        self.assertTrue(torch.equal(a,deploy_input(Frozen(),obs,np.ones((3,3)),['state'])))
        self.assertEqual(tuple(a.shape),(1,13))
        with self.assertRaises(AssertionError):deploy_input(Frozen(),obs,np.ones((3,3)),['state','anchor'])
    def test_held_latent_five_actual_frames(self):
        r=torch.randn(60,256);chosen=torch.tensor([3,7]);held=r[chosen].repeat_interleave(5,0)
        indices=(chosen[:,None]*5+torch.arange(5)).flatten()
        self.assertEqual(indices.tolist(),list(range(15,20))+list(range(35,40)))
        self.assertTrue(torch.equal(held[:5],r[3].expand(5,-1)))
    def test_small_network_gradient_through_frozen_actor(self):
        torch.manual_seed(0);h=torch.randn(8,24);net=Projector(h);stand=Frozen().project_z(torch.randn(1,256))
        actor=torch.nn.Linear(256,29);actor.requires_grad_(False)
        z,_=constrained(Frozen(),stand,net(h),1.);loss=(actor(z)-torch.randn(8,29)).square().mean();loss.backward()
        self.assertGreater(float(net.net[-1].weight.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in actor.parameters()))
if __name__=='__main__':unittest.main()
