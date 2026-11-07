"""Three fixed robot-state prompts on the original audited replay, never contact reward.
Ordinary and softmax-weighted inference share the SAME150k successor sample.
"""
import argparse,gc,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
S=ROOT.parent
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
BUFFER=Path('/mnt/data-oss/users/yi.gao/compliance_BFM/native_reward_assets/62b4206d68e026de5e5dc7efb1529bccfb95164c/train')
MODEL=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view'
BSHA='46eac53f603e45921d25504fcf7d3eea861f2a9f8ea503046855e701493b546b'
MSHA='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361'
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(exist_ok=False);start=time.monotonic()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0';assert BUFFER.joinpath('buffer.hdf5').stat().st_size==26675212288
 print('VERIFY_EXISTING_REPLAY_SHA',flush=True);bs=sha(BUFFER/'buffer.hdf5');assert bs==BSHA
 assert sha(MODEL/'checkpoint/model/model.safetensors')==MSHA
 write(out/'identity.json',dict(buffer_sha256=bs,model_sha256=MSHA,buffer_bytes=26675212288,downloaded=False,raw_mutated=False))
 import numpy as np,torch,mujoco
 from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim,get_idxs
 from reward_stage_contract import load_buffer_cpu,reward_physical_successor,with_inference_batch
 from frozen_bfm_placement import cpu_load_then_stage,tensor_digest,INFERENCE_MODULES
 from humanoidverse.agents import load_utils
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available();dataset=load_buffer_cpu(BUFFER,TrajectoryDictBufferMultiDim)
 assert dataset._is_full and dataset._idx==1250 and dataset.capacity==5000 and 'B' not in dataset.storage
 dataset.output_key_t=list(dict.fromkeys(dataset.output_key_t+['qpos','qvel']));indices=[]
 def get(**kw):
  ix=get_idxs(**kw);indices.append(ix.clone());return ix
 dataset._get_idxs=get;torch.manual_seed(5001);np.random.seed(5001);data=dataset.sample(150000);del dataset;gc.collect()
 np.save(out/'sample-indices.npy',indices[0].numpy(),allow_pickle=False)
 qpos,qvel=reward_physical_successor(data,BSHA);qpos=qpos.numpy();qvel=qvel.numpy();assert qpos.shape==(150000,36) and qvel.shape==(150000,35)
 assert np.max(np.abs(np.linalg.norm(qpos[:,3:7],axis=1)-1))<.002
 # Raw file stores physical successor at rowt, paired with original obs(t+1).
 w,x,y,z=qpos[:,3:7].T
 R=np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(-1,3,3)
 body_v=np.einsum('nji,nj->ni',R,qvel[:,:3]);yaw_rate=qvel[:,5]
 def velocity(target,sigma):return np.exp(-((body_v[:,0]-target)/sigma)**2-(body_v[:,1]/.10)**2-(yaw_rate/.25)**2)
 rewards=[velocity(.15,.10),velocity(.22,.12)]
 model_mj=mujoco.MjModel.from_xml_path(str(ROOT/'humanoidverse/data/robots/g1/scene_29dof_freebase_noadditional_actuators.xml'));d=mujoco.MjData(model_mj)
 wrist=mujoco.mj_name2id(model_mj,mujoco.mjtObj.mjOBJ_BODY,'right_wrist_yaw_link');shoulder=mujoco.mj_name2id(model_mj,mujoco.mjtObj.mjOBJ_BODY,'right_shoulder_pitch_link');assert min(wrist,shoulder)>0
 dz=np.empty(len(qpos),dtype=np.float32)
 print('RELABEL_FIXED3_ROBOT_STATE_PROMPTS',flush=True)
 for i,q in enumerate(qpos):
  d.qpos[:]=q;d.qvel[:]=qvel[i];mujoco.mj_forward(model_mj,d);dz[i]=d.xpos[wrist,2]-d.xpos[shoulder,2]
 tilt=np.arccos(np.clip(R[:,2,2],-1,1));rewards.append(rewards[0]*np.exp(-((dz+.22)/.25)**2-(tilt/.35)**2))
 names=['forward015','forward022','forward015_arm'];rw=[torch.tensor(r[:,None],dtype=torch.float32) for r in rewards]
 for name,r in zip(names,rw):
  assert torch.isfinite(r).all();np.save(out/(name+'-reward.npy'),r.numpy(),allow_pickle=False)
 placement=[]
 with cpu_load_then_stage(load_utils,placement):model=load_utils.load_model_from_checkpoint_dir(MODEL/'checkpoint',device='cuda')
 assert model.cfg.archi.z_dim==256 and model.cfg.archi.norm_z and not model.cfg.amp
 model.cfg=with_inference_batch(model.cfg,4096);frozen={n:tensor_digest(getattr(model,n)) for n in INFERENCE_MODULES}
 weights=[torch.softmax(10*r,dim=0) for r in rw]
 embeddings=[[torch.zeros((1,256),device='cuda'),torch.zeros((1,256),device='cuda')] for r in rw]
 print('GPU_BACKWARD_MAP_SAMPLES150000',flush=True)
 for begin in range(0,150000,4096):
  stop=begin+4096;obs={k:v[begin:stop].cuda() for k,v in data['next']['observation'].items()};b=model.backward_map(obs)
  for j,r in enumerate(rw):
   embeddings[j][0]+=r[begin:stop].cuda().T@b
   embeddings[j][1]+=(r[begin:stop]*weights[j][begin:stop]).cuda().T@b
 result=[]
 for j,name in enumerate(names):
  for method,e in zip(['ordinary','weighted'],embeddings[j]):
   z=model.project_z(e);assert z.shape==(1,256) and abs(float(z.norm())-16)<1e-4
   result.append(dict(name=name+'_'+method,reward_config=name,method=method,z=z.cpu().tolist(),preprojection_embedding=e.cpu().tolist(),reward_mean=float(rw[j].mean()),reward_max=float(rw[j].max()),softmax_weight_ESS=float(1/weights[j].square().sum()),robot_state_only=True,no_contact_anchor_input=True))
 # Check the manual preserved embedding path against the unmodified API once.
 official=model.reward_inference(data['next']['observation'],rw[0]);manual=model.project_z(embeddings[0][0]);diff=float((official-manual).abs().max());assert diff<1e-4,diff
 official_wr=model.reward_wr_inference(data['next']['observation'],rw[0]);manual_wr=model.project_z(embeddings[0][1]);diff_wr=float((official_wr-manual_wr).abs().max());assert diff_wr<1e-4,diff_wr
 assert frozen=={n:tensor_digest(getattr(model,n)) for n in INFERENCE_MODULES}
 assert all(not p.requires_grad and p.grad is None for p in model.parameters())
 write(out/'reward-candidates.json',dict(status='inference_completed_physical_qualification_pending',candidates=result,buffer_sha256=BSHA,model_sha256=MSHA,sample_seed=5001,samples=150000,same_sample_for_all=True,indices_sha256=sha(out/'sample-indices.npy'),methods='ordinary sum(r*B); weighted sum(r*softmax(10r)*B), then original project_z; no normalized-z linear composition',official_API_max_abs_error=diff,official_weighted_API_max_abs_error=diff_wr,source='same-checkpoint original replay, original sampler and successor contract',new_high_actor=False))
 write(out/'result.json',dict(status='completed',wall_s=time.monotonic()-start,backward_samples=450000,relabel_states=150000,configs=3,latents=6,buffer_sha256=BSHA,model_sha256=MSHA,original_bfm_unchanged=True,rl_decisions=0,Student_updates=0))
 print('REWARD_INFERENCE_COMPLETED',json.dumps(dict(wall_s=time.monotonic()-start,candidates=len(result),API_error=diff,weighted_API_error=diff_wr)),flush=True)
if __name__=='__main__':main()
