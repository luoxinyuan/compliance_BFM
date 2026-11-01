"""Fixed existing bounded arm feedback; training-only action projection."""
import numpy as np, torch
from ccl02a_teacher import arm_ik,ReleaseFade
class ArmFeedback:
 def __init__(self,env,contract):
  import mujoco
  self.e=env;self.c=contract;self.fade=ReleaseFade();self.corr=np.zeros(7);self.joint=[mujoco.mj_name2id(env.sim.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in contract['dof_names']];self.adr=env.sim.model.jnt_dofadr[self.joint];self.scale=np.array(contract['q_PD_scale'])[0];self.offset=np.array(contract['q_PD_offset'])[0]
 def step(self,a,t,tick):
  import mujoco
  e=self.e;q=self.offset+(a[0].cpu().numpy()*5).clip(-self.c['clip'],self.c['clip'])*self.scale/5
  if (tick-1)%5==0:
   desired=np.zeros(7)
   if e.tether.active:
    jac=np.zeros((3,e.sim.model.nv));point=e.last_point.copy();mujoco.mj_jac(e.sim.model,e.sim.data,jac,None,point,e.tether.body);desired=arm_ik(jac[:,self.adr[22:]],e.tether.anchor-point)
    lim=e.sim.model.jnt_range[self.joint[22:]];desired=np.clip(q[22:]+desired,lim[:,0],lim[:,1])-q[22:]
    # If nominal itself is outside joint range, do not silently enlarge expert authority.
    desired=np.clip(desired,-.04,.04);desired*=min(1.,.08/max(np.linalg.norm(desired),1e-12))
   self.corr=self.fade.update(t,bool(e.tether.active),desired)
  if self.fade.release_start is not None:self.corr=self.fade.value(t)
  da=torch.zeros_like(a);da[0,22:]=torch.as_tensor(self.corr/self.scale[22:],device=a.device,dtype=a.dtype)
  return a+da,self.corr.tolist()
def project_action(model,obs,base,target,contract,previous=None):
 scale=torch.tensor(contract['q_PD_scale'],device=base.device);off=torch.tensor(contract['q_PD_offset'],device=base.device);qof=lambda a:off+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
 with torch.enable_grad():
  z=base.detach().clone().requires_grad_(True);opt=torch.optim.Adam([z],lr=.06);qt=qof(target).detach();weight=torch.ones_like(qt);weight[:,22:]=2
  for _ in range(16):
   opt.zero_grad();p=model.project_z(z);q=qof(model.act(obs,p,mean=True));loss=((q-qt).square()*weight).mean()+.00002*(p-base).square().mean();z.grad=torch.autograd.grad(loss,z)[0];opt.step()
  result=model.project_z(z).detach()
 q=qof(model.act(obs,result,mean=True));return result,dict(qpd_rms_deg=float(torch.rad2deg((q-qt).square().mean().sqrt())),arm_rms_deg=float(torch.rad2deg((q[:,22:]-qt[:,22:]).square().mean().sqrt())),nonarm_rms_deg=float(torch.rad2deg((q[:,:22]-qt[:,:22]).square().mean().sqrt())),z_angle_deg=float(torch.rad2deg(torch.acos(((result*base).sum()/256).clamp(-1,1)))))
