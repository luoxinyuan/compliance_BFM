import unittest,torch
from m1_interface import base_latent,policy_latent,SharedProjector
class MockModel:
 def project_z(self,z):return 16*torch.nn.functional.normalize(z,dim=-1)
class Interface(unittest.TestCase):
 def test_corners_norm_and_gradients(self):
  torch.manual_seed(0);e=[16*torch.nn.functional.normalize(torch.randn(1,256),dim=-1) for _ in range(4)]
  for u,a,j in [(0,0,0),(1,0,1),(0,1,2),(1,1,3)]:self.assertLess(float(abs(base_latent(e,u,a,0)-e[j]).max()),2e-6)
  self.assertLess(float(abs(base_latent(e,.5,.4,1)-e[0]).max()),2e-6)
  raw=torch.randn(7,259,requires_grad=True);z=policy_latent(MockModel(),e,raw,torch.ones(7));self.assertLess(float(abs(z.norm(dim=-1)-16).max()),4e-6);z[:,0].sum().backward();self.assertTrue(torch.isfinite(raw.grad).all());self.assertGreater(float(raw.grad.abs().sum()),0)
 def test_same_initial_model(self):
  torch.manual_seed(0);a=SharedProjector(torch.randn(4,1425));self.assertEqual(a(torch.zeros(1,1425)).shape,(1,259))
if __name__=='__main__':unittest.main()
