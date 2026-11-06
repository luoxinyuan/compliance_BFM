"""Shared fresh native continuation initialization, never inherited high-level weights."""
import argparse,json,gzip,subprocess,hashlib
from pathlib import Path

def main():
 import torch,numpy as np
 from native_policy import Expert
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(4801);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 with gzip.open(a.root/'native-continuation-labels.json.gz','rt') as f:rows=json.load(f)
 x=torch.tensor([r['x'] for r in rows],device='cuda');y=torch.tensor([r['raw'] for r in rows],device='cuda');names=sorted({r['native'] for r in rows});indices={k:torch.tensor([i for i,r in enumerate(rows) if r['native']==k],device='cuda') for k in names}
 n=Expert(x.shape[-1]).cuda();n.center.copy_(x.mean(0));n.scale.copy_(x.std(0,unbiased=False).clamp_min(.1));n.actor[-1].bias.data.copy_(torch.tensor(json.loads((a.root/'native-assets.json').read_text())['static_z']['standing'][0],device='cuda')/16)
 opt=torch.optim.Adam(n.actor.parameters(),lr=3e-4);curve=[]
 for it in range(2000):
  ix=torch.cat([v[torch.randint(len(v),(32,),device='cuda')] for v in indices.values()]);pred=n.raw_mean(x[ix]);loss=(pred-y[ix]).square().mean();opt.zero_grad();loss.backward();g=float(torch.nn.utils.clip_grad_norm_(n.actor.parameters(),1.));opt.step()
  if it%100==0:curve.append(dict(update=it,raw_MSE=float(loss.detach()),gradient=g))
 # Native direction basis: static reach/body displacements + actual motion PCA;
 # candidate axes are exploration proposals, not labels for force→motion choices.
 assets=json.loads((a.root/'native-assets.json').read_text());stand=np.array(assets['static_z']['standing'][0])/16;vec=[]
 for k,v in assets['static_z'].items():
  if k!='standing':vec.append(np.array(v).reshape(256)/16-stand)
 walk=Path(assets['motion_reference']['path']);zs=np.array(json.loads(walk.read_text())['variants']['task']['straight']['z'])/16;vec.extend(zs-stand)
 _,sv,vh=np.linalg.svd(np.array(vec),full_matrices=False);rank=min(4,len(sv));dirs=torch.tensor(vh[:rank],dtype=torch.float32,device='cuda')
 # Recheck single-sampling versus batched probability on actual native inputs,
 # after fitting a nonzero mean; original assertion is not relaxed.
 audits=[]
 for arm in ['ordinary','guided']:
  n.set_covariance(dirs if arm=='guided' else None)
  with torch.no_grad():
   xx=x[torch.arange(768,device='cuda')%len(x)]
   ds=[n.distribution(v[None]) for v in xx];uu=torch.cat([d.sample() for d in ds]);old=torch.cat([d.log_prob(v[None]).sum(-1) for d,v in zip(ds,uu)])
   err=float((n.distribution(xx).log_prob(uu).sum(-1)-old).abs().max())
   audits.append({'arm':arm,'actual_native_obs_joint_logprob_max':err,'original_threshold':3e-4})
   (a.root/'native-real-probability.json').write_text(json.dumps(audits,indent=2))
   assert err<3e-4,(arm,err)
 n.set_covariance(None);path=a.root/'native-initial.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in n.state_dict().items()},input_dim=x.shape[-1],kind='native-direct',initialization='fresh actor/critic; native physical continuation only',updates=2000,old_actor_loaded=False,runtime=subprocess.check_output(['git','-C',str(Path(__file__).parents[1]),'rev-parse','HEAD'],text=True).strip()),path);np.save(a.root/'native-directions.npy',dirs.cpu().numpy())
 (a.root/'initialization.json').write_text(json.dumps(dict(samples=len(rows),independent_episodes=len({r['episode'] for r in rows}),per_source={k:len(v) for k,v in indices.items()},curve=curve,updates=2000,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),direction_singular_values=sv[:8].tolist(),rank=rank,old_high_actor_or_labels_used=False,scope='initialization prior; no contact selection success claimed'),indent=2))
 print('NATIVE_INITIALIZED',len(rows),names,flush=True)
if __name__=='__main__':main()
