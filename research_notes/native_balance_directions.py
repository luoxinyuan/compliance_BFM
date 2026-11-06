"""One pre-guided correction of source-count bias, not a latent/reward search.
Ordinary running arm never reads this asset; old basis retained byte-for-byte.
"""
import argparse,json,hashlib,gzip,time,os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();start=time.monotonic();root=a.root
assert not (root/'ppo-guided').exists(),'do not alter any started guided experiment'
old=root/'native-directions.npy';before=old.read_bytes();backup=root/'native-directions-unweighted-v1.npy';assert not backup.exists();backup.write_bytes(before)
assets=json.loads((root/'native-assets.json').read_text());standing=np.array(assets['static_z']['standing']).reshape(256)/16
sources=[]
for k in ['goal_arm5','goal_arm15','goal_lean30']:
 v=np.array(assets['static_z'][k]).reshape(256)/16;d=v-standing*np.dot(standing,v);sources.append((k,d))
ref=json.loads(Path(assets['motion_reference']['path']).read_text())['variants']['task']['straight'];v=np.array(ref['z']).mean(0)/16;v-=standing*np.dot(standing,v);sources.append(('motion_walk_mean',v))
rows=[]
for name,v in sources:
 q=v.copy()
 for z in rows:q-=np.dot(q,z)*z
 assert np.linalg.norm(q)>1e-5,(name,np.linalg.norm(q));rows.append(q/np.linalg.norm(q))
D=np.array(rows,dtype=np.float32);assert np.max(abs(D@D.T-np.eye(4)))<1e-5
candidate=root/'native-directions-balanced-v2.npy';np.save(candidate,D)
# Exact fitted real-input probability check, no BFMinference/physics/update.
import torch
from native_policy import Expert
assert torch.cuda.is_available();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(4808)
ck=torch.load(root/'native-initial.pt',map_location='cuda',weights_only=True);n=Expert(ck['input_dim']).cuda();n.load_state_dict(ck['state_dict']);n.set_covariance(torch.tensor(D,device='cuda'))
with gzip.open(root/'native-continuation-labels.json.gz','rt') as f:data=json.load(f)
x=torch.tensor([r['x'] for r in data],device='cuda');x=x[torch.arange(768,device='cuda')%len(x)]
with torch.no_grad():
 ds=[n.distribution(v[None]) for v in x];u=torch.cat([d.sample() for d in ds]);lp=torch.cat([d.log_prob(v[None]).sum(-1) for d,v in zip(ds,u)]);error=float((n.distribution(x).log_prob(u).sum(-1)-lp).abs().max());meanerr=float((n.to_raw(n.distribution(x).mean)-n.raw_mean(x)).abs().max())
 assert error<3e-4,(error,meanerr);assert meanerr<1e-5,meanerr
capture={name:float(np.linalg.norm(D.T@(D@v))**2/np.linalg.norm(v)**2) for name,v in sources}
manifest={'stage':'pre-guided-only source-balance correction','ordinary_unchanged':True,'guided_RL_decisions_before_change':0,'reason':'71motion frames vs3static rows in unweighted SVD, static tangent capture only9-13%; directly balance source families instead of scanning outputs','basis_v1_sha256':hashlib.sha256(before).hexdigest(),'basis_v2_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'sources':[k for k,v in sources],'tangent_span_capture':capture,'covariance_trace_unchanged_v1_to_v2':True,'rank':4,'axis_std':.15,'isotropic_floor':.008,'joint_logprob_real_input_max':error,'original_logprob_threshold':3e-4,'whitened_mean_inverse_max':meanerr,'old_high_model_or_final_data_used':False,'seconds':time.monotonic()-start,'ended':time.time()}
assert not (root/'ppo-guided').exists();tmp=old.with_suffix('.tmp');tmp.write_bytes(candidate.read_bytes());tmp.replace(old)
(root/'native-direction-versions.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
