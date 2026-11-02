"""Frozen-policy diagnostic instrumentation, never used to update parameters."""
import numpy as np,torch,mujoco
from m11_policy import legal_command

def monitor_point_read(task):
 t=task.e.tether;m,d=t.model,t.data;kind=mujoco.mjtState.mjSTATE_INTEGRATION
 before=np.empty(mujoco.mj_stateSize(m,kind));after=np.empty_like(before);original=t.point_state
 result=dict(calls=0,integration_state_max_change=0.)
 def checked():
  mujoco.mj_getState(m,d,before,kind);out=original();mujoco.mj_getState(m,d,after,kind)
  result['calls']+=1;result['integration_state_max_change']=max(result['integration_state_max_change'],float(np.abs(before-after).max()));return out
 t.point_state=checked
 return result

def action_probe(task,net,obs):
 """Same observation, same epsilon; only raw sampling scale changes. No state step."""
 base=task.prepare();dist=net.distribution(obs);mean=dist.mean;g=torch.Generator(device=obs.device).manual_seed(821)
 eps=torch.randn((8,256),device=obs.device,generator=g);a0=task.m.act(task.e.obs,base,mean=True);q0=task.qof(a0);rows=[]
 for scale in [0.,.5,1.]:
  qs=[];angles=[];norms=[]
  for epsilon in eps[:1] if scale==0 else eps:
   raw=mean+scale*dist.stddev*epsilon[None];z,norm=legal_command(task.m,base,raw);q=task.qof(task.m.act(task.e.obs,z,mean=True));qs.append(q[0]);norms.append(float(norm));angles.append(float(torch.rad2deg(torch.acos(((z*base).sum()/256).clamp(-1,1)))))
  q=torch.stack(qs);delta=torch.rad2deg(q-q0)
  rows.append(dict(std_scale=scale,raw_tangent_norm_mean=float(np.mean(norms)),angle_deg_mean=float(np.mean(angles)),qPD_rms_delta_deg=float(delta.square().mean().sqrt()),arm_qPD_rms_delta_deg=float(delta[:,22:].square().mean().sqrt()),nonarm_qPD_rms_delta_deg=float(delta[:,:22].square().mean().sqrt()),qPD_sample_std_rms_deg=float(torch.rad2deg(q.std(0,unbiased=False)).square().mean().sqrt()),qPD_delta_max_deg=float(delta.abs().max())))
 return dict(time=float(task.e.tether.time),raw_mean_norm=float(mean.norm()),raw_std_mean=float(dist.stddev.mean()),feature_clip_fraction=float((obs[:,:490].abs()>=9.999).float().mean()),force=task.e.tether.force.tolist(),cases=rows)
