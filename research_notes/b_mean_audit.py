"""Original B only, offline diagnostic. Never installs centered inference."""
import gc,json,time
from pathlib import Path
import numpy as np
import torch
from dual_track_high import write,sha
from native_env import Task,spec
from wholebody_physics import goal_observation
from rolling_reward import BUFFER,BSHA,MSHA

def angles(z):
 u=z.double()/z.double().norm(dim=-1,keepdim=True)
 a=torch.rad2deg(torch.acos((u@u.T).clamp(-1,1)))
 v=a[torch.triu(torch.ones_like(a,dtype=torch.bool),diagonal=1)]
 return dict(matrix_deg=a.tolist(),min=float(v.min()),p25=float(v.quantile(.25)),median=float(v.median()),p75=float(v.quantile(.75)),max=float(v.max()))

def run(e,m,c,out,args):
 start=time.monotonic()
 from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim,get_idxs
 from reward_stage_contract import load_buffer_cpu,reward_physical_successor
 assert c['model_sha256']==MSHA
 # Full existing asset identity; no download, rewriting or new source asset.
 assert sha(BUFFER/'buffer.hdf5')==BSHA
 dataset=load_buffer_cpu(BUFFER,TrajectoryDictBufferMultiDim)
 dataset.output_key_t=list(dict.fromkeys(dataset.output_key_t+['qpos','qvel']));indices=[]
 def get(**kw):
  ix=get_idxs(**kw);indices.append(ix.clone());return ix
 dataset._get_idxs=get;torch.manual_seed(5803);np.random.seed(5803)
 data=dataset.sample(30000);del dataset;gc.collect()
 np.save(out/'sample-indices.npy',indices[0].numpy(),allow_pickle=False)
 qp,qv=reward_physical_successor(data,BSHA);qp=qp.numpy();qv=qv.numpy()
 w,x,y,z=qp[:,3:7].T
 R=np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(-1,3,3)
 v=np.einsum('nji,nj->ni',R,qv[:,:3]);tilt=np.arccos(np.clip(R[:,2,2],-1,1))
 velocity=lambda target:np.exp(-((v[:,0]-target)/.12)**2-(v[:,1]/.10)**2-(qv[:,5]/.25)**2)
 names=['forward015','backward015','standing','lower_height','arm_lift','waist_left','waist_right']
 rewards=[velocity(.15),velocity(-.15),np.exp(-np.sum(v*v,1)/.08**2-(qv[:,5]/.2)**2),
   np.exp(-((qp[:,2]-.65)/.08)**2-(tilt/.35)**2),np.exp(-((qp[:,7+22]+.8)/.25)**2-(tilt/.35)**2),
   np.exp(-((qp[:,7+12]-.26)/.15)**2-(tilt/.35)**2),np.exp(-((qp[:,7+12]+.26)/.15)**2-(tilt/.35)**2)]
 blocks=[]
 for begin in range(0,30000,2048):
  obs={k:v[begin:begin+2048].cuda() for k,v in data['next']['observation'].items()}
  blocks.append(m.backward_map(obs).cpu())
 B=torch.cat(blocks).double();mu=B.mean(0,keepdim=True);center=B-mu
 rw=torch.tensor(np.array(rewards).T,dtype=torch.float64)
 emb=rw.T@B;ce=rw.T@center;manual_z=m.project_z(emb.float().cuda()).cpu();zs=torch.cat([m.reward_inference(data['next']['observation'],rw[:,j:j+1].float()).cpu() for j in range(len(names))]);czs=m.project_z(ce.float().cuda()).cpu()
 task=Task(e,m,c,5803);task.reset(spec(0.,4.,source='standing',until=4.,enabled=False),record=True,label='goal-origin')
 q=e.sim.data.qpos.copy();nv=e.sim.model.nv
 poses={'standing_goal':q.copy()}
 for name,j,offset in [('waist_left_goal',12,.26),('waist_right_goal',12,-.26),('arm_lift_goal',22,-.6)]:
  p=q.copy();p[7+j]+=offset;poses[name]=p
 p=q.copy();p[2]-=.06;p[7+2]-=.12;p[7+3]+=.25;p[7+8]-=.12;p[7+9]+=.25;poses['lower_goal']=p
 goalB=[]
 for name,p in poses.items():goalB.append(m.backward_map(goal_observation(e,p,np.zeros(nv))).cpu().double())
 gb=torch.cat(goalB);gz=m.project_z(gb.float().cuda()).cpu();gcz=m.project_z((gb-mu).float().cuda()).cpu()
 allnames=names+list(poses);allz=torch.cat([zs,gz]);allcz=torch.cat([czs,gcz]);all_emb=torch.cat([emb,gb]);all_ce=torch.cat([ce,gb-mu])
 unitmu=mu/mu.norm();rows=[]
 for i,name in enumerate(allnames):
  ep=all_emb[i];cen=all_ce[i];projection=float((ep@unitmu[0])/ep.norm())
  common=rw[:,i].sum()*mu[0] if i<len(names) else mu[0]
  rows.append(dict(name=name,source='robot_state_replay_reward' if i<len(names) else 'single_geometric_goal_not_execution',
    z=allz[i:i+1].tolist(),centered_z_DIAGNOSTIC_ONLY=allcz[i:i+1].tolist(),
    embedding_norm=float(ep.norm()),centered_embedding_norm=float(cen.norm()),
    mean_direction_cosine=projection,mean_direction_squared_projection_fraction=projection**2,
    common_term_norm_ratio=float(common.norm()/ep.norm()),centering_angle_deg=float(torch.rad2deg(torch.acos((allz[i]@allcz[i]/256).double().clamp(-1,1)))),
    reward_mean=float(rw[:,i].mean()) if i<len(names) else None))
 halves=[B[:15000].mean(0),B[15000:].mean(0)]
 result=dict(status='completed',original_model_sha256=MSHA,replay_sha256=BSHA,states=30000,seed=5803,
  official_reward_inference_calls=len(names),official_vs_double_aggregation_max_abs=float((zs-manual_z).abs().max()),B_mean_norm=float(mu.norm()),B_individual_norm_mean=float(B.norm(dim=1).mean()),
  B_mean_to_individual_norm_ratio=float(mu.norm()/B.norm(dim=1).mean()),
  split_mean_angle_deg=float(torch.rad2deg(torch.acos(torch.nn.functional.cosine_similarity(*[x[None] for x in halves]).clamp(-1,1)))),
  names=allnames,rows=rows,reward_pairwise_original=angles(zs),reward_pairwise_centered=angles(czs),
  goal_pairwise_original=angles(gz),goal_pairwise_centered=angles(gcz),combined_original=angles(allz),combined_centered=angles(allcz),
  centering='diagnostic only: mu from same 30000 replay successor distribution; sum r*(B-mu); goal B(goal)-mu; original norm16 projection',
  limitations='distribution-dependent mean; geometric goals not physically qualified here; angle change does not establish prompting failure cause',
  formal_updates=0,wall_s=time.monotonic()-start)
 torch.save(dict(B=B.float(),mean=mu.float(),rewards=rw.float(),goals=gb.float(),z=allz,centered_z=allcz),out/'offline-tensors.pt')
 write(out/'b-mean-results.json',result);return result
