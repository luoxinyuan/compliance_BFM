"""CPU-only final-snapshot Bellman and same-state operator audit; no step/optimizer."""
import runpy,time,json
from pathlib import Path
T=__import__('time').monotonic()
g=runpy.run_path(str(Path(__file__).with_name('audit.py')))
globals().update({k:v for k,v in g.items() if not k.startswith('__')})
from humanoidverse.agents import load_utils
from research.interaction_fb.agent import Agent
from research.interaction_fb.schema import robot
base=load_utils.load_model_from_checkpoint_dir(str(P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view/checkpoint'),device='cpu');base.eval().requires_grad_(False)
a=Agent.__new__(Agent);a.arm='B';a.base=base;a.nets=nn.ModuleDict({k:net(models['V1'],k) for k in ['u','adapter','q','disc']});a.targets=nn.ModuleDict({k:net(models['V1'],k,True) for k in ['u','adapter','q','disc']})
a.goal=models['V1']['goal'];a.scale=models['V1']['scale'];a.prompt=models['V1']['prompt'];a.exploration_std=.01;a.exploration_clip=.06;a.envelope=None
# Exact original B19 operator (no Envelope) for its actual learning target.
chosen=[]
for ph in ['precontact','active','stopping','holding','recovery']:
 ids=[i for i,r in enumerate(rows) if phase(r)==ph]
 chosen.extend([ids[int(i)] for i in np.linspace(0,len(ids)-1,min(12,len(ids))).astype(int)])
vals=[]
with torch.no_grad():
 for i in chosen:
  r=rows[i];no=r['next']['observation'];na,nz,ni=a.action(no,True)
  boot=a.targets['q'](a.inp(no,na,nz,ni));val=a.nets['q'](a.inp(r['observation'],r['action'],r['z'],r['zi']))
  d=.99**(r['duration']/.1)*(~r['next']['terminated']);rw=r['aux_rewards']['contact_physical'];target=rw+d*boot
  vals.append(dict(index=i,phase=phase(r),parent=pid(r),reward=float(rw),discount=float(d),target_next_Q=float(boot),Q=float(val),target=float(target),TD=float(val-target)))
write('target-policy-td.json',dict(contract='Original B19 no-Envelope final frozen-target policy, CPU retrospective stratified 12/phase. Not historical per-update losses; target is EMA not V1/Return continuation.',rows=vals,by_phase={ph:summary([abs(r['TD']) for r in vals if r['phase']==ph]) for ph in sorted(set(r['phase'] for r in vals))}))
# Same-state motion from all three adapters on both source visited domains, stored real z/base.
same=[]
for case in (2,3):
 for source in ['V1','return64']:
  tr=traces[(case,source)]
  for it in [0,25,75,149]:
   f=tr['actions'][it];o=f['observation'];z=f['z'];b=f['audit']['base'];x=torch.cat([deploy(o),z,torch.zeros(1,32)],-1);outs={}
   for name in models:
    delta=ad[name](x);ex,au=env(b,(b+delta).clamp(-1,1));outs[name]=dict(raw=(delta*sc).tolist()[0],action=ex.tolist()[0],qPD=au['qexec'].tolist()[0],Q=float(q(torch.cat([privileged(o),ex,z,torch.zeros(1,32)],-1))))
   same.append(dict(case=case,source=source,time=f['time'],slot=int(o['_slot']),models=outs))
write('same-state-cross-policy.json',same)
# Negative semi-definite reward => any genuine return <=0, irrespective of future policy.
with torch.no_grad():
 qlist=[]
 for r in rows:qlist.append(float(q(torch.cat([privileged(r['observation']),r['action'],r['z'],r['zi']],-1))))
write('Q-sign.json',dict(reward_max=max(float(r['aux_rewards']['contact_physical']) for r in rows),Q_positive=sum(v>0 for v in qlist),rows=len(rows),by_phase={ph:dict(n=sum(phase(r)==ph for r in rows),positive=sum(v>0 for r,v in zip(rows,qlist) if phase(r)==ph)) for ph in sorted(set(phase(r) for r in rows))},interpretation='Positive Q violates true return upper bound0; establishes calibration error, not its cause or relative action-ranking error.'))
write('target-receipt.json',dict(wall_s=__import__('time').monotonic()-T,new_physics=0,new_updates=0,cuda_initialized=__import__('torch').cuda.is_initialized()))
print('TARGET_AUDIT_COMPLETE',len(vals))
