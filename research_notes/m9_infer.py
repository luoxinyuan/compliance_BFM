"""Same-weight original reward inference, fixed legacy sample, bounded memory."""
import json,hashlib,time,gc
from pathlib import Path
import numpy as np,torch,mujoco,h5py
from contact_projector import check_deadline
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUN=P/'bridge-v2-owned/runs/m9-contact-locomotion-20261101A';ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def run(env,model,contract,out):
 from humanoidverse.envs.g1_env_helper.robot import make_from_name
 from reward_stage_contract import with_inference_batch
 plan=json.loads((ROOT/'research_notes/m9_protocol.json').read_text());old=P/'bridge-v2-owned/runs/issue21-reward-stage-20261030A/inference-attempt-01/outputs';buffer=Path('/mnt/data-oss/users/yi.gao/compliance_BFM/native_reward_assets/62b4206d68e026de5e5dc7efb1529bccfb95164c/train/buffer.hdf5');assert sha(buffer)=='46eac53f603e45921d25504fcf7d3eea861f2a9f8ea503046855e701493b546b';assert sha(old/'sample-indices.npy')=='ebc708e2694b273935e16a8d087a69c5e3f5d741300f01b92cd79ab414207d29'
 ix=np.load(old/'sample-indices.npy');f=h5py.File(buffer,'r')
 def extract(key,next_obs=False):
  ds=f[key];assert ds.chunks is None and ds.id.get_offset() is not None
  mm=np.memmap(buffer,dtype=ds.dtype,mode='r',offset=ds.id.get_offset(),shape=ds.shape);a=np.asarray(mm[(ix[:,0]+int(next_obs))%5000,ix[:,1]]).copy();del mm;gc.collect();return a
 q=extract('qpos');v=extract('qvel');a=extract('action');obs={k:torch.from_numpy(extract('observation-'+k,True)) for k in ['state','privileged_state']};f.close()
 tasks=plan['reward_tasks'];fs=[make_from_name(t) for t in tasks];m=mujoco.MjModel.from_xml_path(str(ROOT/'humanoidverse/data/robots/g1/scene_29dof_freebase_noadditional_actuators.xml'));d=mujoco.MjData(m);r=np.empty((len(ix),len(tasks)),np.float32);oldfn=make_from_name('move-ego-0-0.3');oldr=np.load(old/'move-ego-0-0.3-rewards.npy');audit=0.
 for i in range(len(ix)):
  d.qpos[:]=q[i];d.qvel[:]=v[i];d.ctrl[:]=a[i];mujoco.mj_forward(m,d)
  for j,fn in enumerate(fs):r[i,j]=fn.compute(m,d)
  if i<100:
   audit=max(audit,abs(oldfn.compute(m,d)-float(oldr[i,0])))
   for j,fn in enumerate(fs):assert abs(float(r[i,j])-fn(m,q[i],v[i],a[i]))<1e-6
  if i%10000==0:check_deadline();print('M9_REWARD_ROWS',i,flush=True)
 assert audit<1e-6;np.save(out/'reward-values.npy',r);model.cfg=with_inference_batch(model.cfg,4096);candidates={}
 for j,task in enumerate(tasks):
  check_deadline();z=model.reward_wr_inference(obs,torch.from_numpy(r[:,j:j+1].copy()));assert z.shape==(1,256) and abs(float(z.norm())-16)<1e-4
  weights=torch.softmax(10*torch.from_numpy(r[:,j]),0);candidates[task]=dict(z=z.tolist(),norm=float(z.norm()),reward_mean=float(r[:,j].mean()),reward_max=float(r[:,j].max()),ESS=float(1/weights.square().sum()),source='original reward_wr_inference; fixed audited150k transitions; same model')
  print('M9_LATENT',task,flush=True)
 candidates['move-ego-0-0.3']=dict(**json.loads((old/'move-ego-0-0.3-latent.json').read_text()),source='old already audited same-weight latent; not regenerated')
 (out/'reward-candidates.json').write_text(json.dumps(candidates,indent=2));(out/'sampling-contract.json').write_text(json.dumps(dict(buffer_sha256=sha(buffer),sample_sha256=sha(old/'sample-indices.npy'),old_reward_first100_error=audit,rows=len(ix),seed=20261030,notes='Memory-mapped read-only original HDF5 rows; physics(t) is already successor in this pinned legacybuffer, observations(t+1). No change to sampler/reward/weights, no newseed selection.'),indent=2))
 return dict(status='completed',candidate_count=len(candidates),student_optimizer_updates=0)
