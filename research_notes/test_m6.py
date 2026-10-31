import unittest,json,sys
from pathlib import Path
import torch
from m6_learning import Projector,policy_latent
from contact_gate import ContactGate
class FakeModel:
 def project_z(self,z):return 16*torch.nn.functional.normalize(z,dim=-1)
class Tests(unittest.TestCase):
 def test_deployment_latent_gradient(self):
  torch.manual_seed(0);h=torch.randn(8,1425);net=Projector(h);ends=[16*torch.nn.functional.normalize(torch.randn(1,256),dim=-1) for _ in range(4)];z=policy_latent(FakeModel(),ends,net(h),torch.ones(8));self.assertLess(float(abs(z.norm(dim=-1)-16).max()),1e-4);loss=(z-ends[2]).square().mean();loss.backward();self.assertGreater(float(net.net[-1].weight.grad.abs().max()),0)
 def test_zero_contact_native_identity(self):
  a=16*torch.nn.functional.normalize(torch.randn(1,256),dim=-1);ends=[a,a,a,a];z=policy_latent(FakeModel(),ends,torch.zeros(1,257),0.);self.assertTrue(torch.allclose(a,z,atol=1e-5))
 def test_causal_teacher_and_release(self):
  cfg=json.loads(Path(__file__).with_name('contact_gate_protocol.json').read_text())['gate'];g=ContactGate(cfg)
  for _ in range(500):self.assertEqual(g.step([0.,0.,0.])['alpha'],0.)
  for _ in range(250):g.step([2.,0.,0.])
  self.assertGreater(g.alpha,.9)
  for _ in range(70):g.step([0.,0.,0.])
  self.assertEqual(g.alpha,0.)
 def test_negative_force_does_not_command_forward_goal(self):
  cfg=json.loads(Path(__file__).with_name('contact_gate_protocol.json').read_text())['gate'];g=ContactGate(cfg)
  for _ in range(500):self.assertEqual(g.step([-2.,.2,.1])['alpha'],0.)
 def test_force_rule_uses_signed_force_and_preserves_zero_contact(self):
  from m6_rule import command
  base=torch.zeros(1,256);base[0,0]=16.;template=torch.zeros_like(base);template[0,1]=.1;h=torch.zeros(1,1425)
  self.assertTrue(torch.equal(command(FakeModel(),base,h,1.,template,1.),base))
  h[:,-9:]=torch.tensor([[-.2,0,0]*3]);z=command(FakeModel(),base,h,1.,template,1.)
  self.assertGreater(float(z[0,1]),0);self.assertAlmostEqual(float(z.norm()),16,places=4)
  self.assertTrue(torch.equal(command(FakeModel(),base,h,0.,template,1.),base))
  h=-h;self.assertTrue(torch.equal(command(FakeModel(),base,h,1.,template,1.),base))
 def test_test_conditions_not_collected(self):
  p=json.loads(Path(__file__).with_name('m6_protocol.json').read_text());self.assertTrue(all(not c.startswith('test') for s in ['qualification','collect1','collect2'] for c,_ in p['schedule'][s]));self.assertEqual(p['training']['steps'],[3000,2000,2000])
if __name__=='__main__':unittest.main()
