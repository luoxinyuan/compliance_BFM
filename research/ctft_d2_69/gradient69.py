from pathlib import Path
import os,time,hashlib,json,torch
from agent import build
from packed_checkpoint import PackedTensorStore
from learning_checkpoint import digest
s=Path.cwd();stage=Path(os.environ['CTFT_PINNED_RUNTIME']).parent
t=time.monotonic()
u=stage/'runs/formal-U-v1a-01'
old=s.parent/'ctft_d1_68'
events=json.loads((old/'d0/training-events.json').read_bytes())
picks=sorted((r for r in events if r['tick']==0 and r['case'] in [0,1,2,4,5,6,7,8]),key=lambda r:r['row'])
obs={};zs=[];ids=[];blocks={}
for r in picks:
    i=r['row'];end=(i//1024+1)*1024;p=u/f'actual-rows-{end:06d}.pt'
    if end not in blocks:
        block=torch.load(p,map_location='cpu',weights_only=False);blocks[end]=hashlib.sha256(p.read_bytes()).hexdigest()
    for k,v in block['observation'].items():obs.setdefault(k,[]).append(v[i%1024].clone())
    zs.append(block['z'][i%1024].clone());ids.append([r['case'],r['phase'],i])
obs={k:torch.stack(v) for k,v in obs.items()};z=torch.stack(zs)
torch.save(dict(obs=obs,z=z,ids=ids,blocks=blocks),s/'gradient-batch.pt')
agent,_=build(aux_scale=json.loads((stage/'locks/common-aux-scale-v1.json').read_bytes())['value'],compile_execution=False)
store=PackedTensorStore(u/'store');meta=store.metadata('round-131072')['learner']['model'];model=agent._model
with torch.no_grad():
    for k,v in model.state_dict().items():
        if k.startswith(('_actor.','_critic.','_forward_map.','_aux_critic.','_obs_normalizer.')):v.copy_(store.unpack(meta[k]))
sha=digest({k:v.cpu().clone() for k,v in model._actor.state_dict().items()})
assert sha==json.loads((old/'forward/reference/model-identities.json').read_bytes())['U']
model.eval();z=z.cuda();obs=model._obs_normalizer({k:v.cuda() for k,v in obs.items()})
a=model._actor(obs,z,model.cfg.actor_std).mean
def q(v):return agent.get_targets_uncertainty(v,agent.cfg.train.actor_pessimism_penalty)[2]
fb=q((model._forward_map(obs,z,a)*z).sum(-1));weight=fb.abs().mean().detach() if agent.cfg.train.scale_reg else 1.
terms=[fb.mean(),q(model._critic(obs,z,a)).mean()*agent.cfg.train.reg_coeff*weight,q(model._aux_critic(obs,z,a)).mean()*agent.cfg.train.reg_coeff_aux*weight];gr=[];params=list(model._actor.parameters())
for term in terms:
    g=torch.autograd.grad(-term,[a]+params,retain_graph=True,allow_unused=True);gr.append([g[0].flatten(),torch.cat([v.flatten() for v in g[1:] if v is not None])])
def result(i):
    m=torch.stack([v[i] for v in gr]);unit=torch.nn.functional.normalize(m);return dict(norm=m.norm(dim=-1).tolist(),cosine=(unit@unit.T).tolist())
(s/'gradient.json').write_text(json.dumps(dict(domain='frozen U mean at archived historical training states',order=['FB','CPR','aux'],actor_sha=sha,n=len(ids),ids=ids,action=result(0),parameters=result(1),seconds=time.monotonic()-t),indent=2))
