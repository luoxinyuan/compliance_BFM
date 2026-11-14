"""D0: CPU archive arithmetic/autograd only; no environment construction or optimizer."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,io,zlib,hashlib,collections,csv
from pathlib import Path
import torch,numpy as np
from torch import nn
T=time.monotonic();torch.set_num_threads(2)
P=Path(os.environ['COMPLIANCE_PROJECT_ROOT']);S=P/'experiments/adapter-return-pilot-20261113';R=S/'runtime'
sys.path[:0]=[str(R),str(R/'research_notes')]
from research.interaction_fb.schema import deploy,privileged
from research.interaction_fb.envelope import Envelope
O=Path(__file__).resolve().parent/'data';O.mkdir(exist_ok=True)
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def write(n,v):(O/n).write_text(json.dumps(v,indent=2,allow_nan=False))
def compressed(p,rec):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==rec['sha256'];return torch.load(io.BytesIO(zlib.decompress(b)),map_location='cpu',weights_only=False)
path=P/'experiments/interaction-pilot-v3-20261112/runs/pilot-B-v3/checkpoints/round-19'
m=json.loads((path/'manifest.json').read_text());ck=compressed(path/'core.ptz',m['files']['core.ptz']);rows=[];native=[]
for seg in m['segments']:
 d=compressed(path.parent/seg['path'],seg);rows+=d['rows'];native+=d['native']
models={'V1':ck['agent']}
for name in ['control','return']:models[name+'64']=load(S/f'runs/fit-01/{name}/adapter-64.pt')['agent']
def net(state,key,targets=False):
 sd={k[len(key)+1:]:v for k,v in state['targets' if targets else 'nets'].items() if k.startswith(key+'.')}
 n=nn.Sequential(nn.Linear(sd['0.weight'].shape[1],128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,sd['4.weight'].shape[0]));n.load_state_dict(sd);n.eval().requires_grad_(False);return n
q=net(models['V1'],'q');targetq=net(models['V1'],'q',True)
ad={k:net(v,'adapter') for k,v in models.items()}
conf=models['return64']['envelope'];env=Envelope(conf,'cpu');sc=env.scale
names=[r['name'] for r in conf['mechanical_table']]
idx=json.loads((S/'data/archive-checks.json').read_text());specs=ck['task_specs']
def phase(r):
 s=specs[int(r['provenance']['spec_index'])];t=float(r['provenance']['time'])
 if not s['enabled']:return 'none'
 if t<s['onset']-1e-7:return 'precontact'
 if t>=s['release']-1e-7:return 'recovery'
 if 'stop_time' in s and t>=s['stop_time']-1e-7:
  return 'stopping' if t<s['stop_time']+s['deceleration']-1e-7 else 'holding'
 return 'holding' if t>=s['onset']+s['ramp']-1e-7 else 'active'
def pid(r):return bytes(r['provenance']['parent_sha256'].flatten().tolist()).hex()
def coverage(indices):
 out={}
 for i in indices:
  r=rows[i];key=(pid(r),phase(r));d=out.setdefault(key,dict(parent=key[0],phase=key[1],rows=0,seconds=0.,first=1e9,last=0.));d['rows']+=1;d['seconds']+=float(r['duration']);d['first']=min(d['first'],float(r['provenance']['time']));d['last']=max(d['last'],float(r['provenance']['time']))
 return list(out.values())
schedule=json.loads((S/'runs/fit-01/batch-schedule.json').read_text());draws=[idx['replay_indices'][i] for b in schedule for i in b['replay']]
old_draws=[i for a in ck['agent']['audit'] for i in a['replay_indices']]
write('coverage.json',dict(original=coverage(range(len(rows))),return_pool=coverage(idx['replay_indices']),return_draws=coverage(draws),return_unique_draws=coverage(sorted(set(draws))),Q_original_draws=coverage(old_draws),Q_original_unique=coverage(sorted(set(old_draws))),original_native_count=len(native),new_native_count=len(idx['native_indices']),new_Q_updates=0,original_Q_updates=ck['agent']['updates']))

def summary(x):
 a=np.asarray(x,float);return dict(n=len(a),mean=float(a.mean()),p50=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max()))
traces={};reward=[];jointrows=[];firsts=[];support_dist=[]
for case in (2,3):
 for name in models:
  folder={'V1':'V1','control64':'control','return64':'return'}[name]
  tr=load(S/f'runs/evaluate-case-{case:02d}-01/{folder}-mean/trace.pt');traces[(case,name)]=tr
  costs=collections.defaultdict(float);disc=1.;ret=0.;maxdiff=0.
  for c,r in zip(tr['controls'],tr['rows']):
   dt=float(r['duration']);rr=float(r['aux_rewards']['contact_physical']);rec=-c['actual_duration_s']*sum(c['reward_terms'].values());maxdiff=max(maxdiff,abs(rec-rr));ret+=disc*rr
   for k,v in c['reward_terms'].items():costs[k]+=disc*c['actual_duration_s']*v
   disc*=.99**(dt/.1)
  reward.append(dict(case=case,model=name,costs=dict(costs),return3s=ret,first20ms_reward=float(tr['rows'][0]['aux_rewards']['contact_physical']),tail_discount=disc,serialized_reward_max_abs_error=maxdiff,
    slip_cost_capped_controls=sum(c['reward_terms']['slip']>=.8-1e-9 for c in tr['controls']),controls=len(tr['controls'])))
  aus=[x['audit'] for x in tr['actions']];raw=torch.cat([a['adapter'] for a in aus]);exe=torch.cat([a['executed'] for a in aus]);bases=torch.cat([a['base'] for a in aus]);cor=torch.cat([a['qexec']-a['qbase'] for a in aus]);trig=torch.cat([a['triggered'] for a in aus])
  qpd=np.asarray([c['q_PD'] for c in tr['controls']]);qpos=np.asarray([c['pre_qpos'] for c in tr['controls']]);physq=np.asarray([r['qpos'] for r in tr['physics']]);physdq=np.asarray([r['qvel'] for r in tr['physics']])
  for j,n in enumerate(names):
   jointrows.append(dict(case=case,model=name,joint=n,index=j,raw_rad_rms=float((raw[:,j]*sc[0,j]).square().mean().sqrt()),exec_rad_rms=float(cor[:,j].square().mean().sqrt()),trigger_fraction=float(trig[:,j].float().mean()),qPD_min=float(qpd[:,j].min()),qPD_max=float(qpd[:,j].max()),qPD_step_max=float(abs(np.diff(qpd[:,j])).max()),qpos_index=j+7,actual_q_min=float(physq[:,j+7].min()),actual_q_max=float(physq[:,j+7].max()),actual_dq_abs_max=float(abs(physdq[:,j+6]).max()),qPD_q_rms=float(np.mean((qpd[:,j]-qpos[:,j+7])**2)**.5)))
 # First common state: stored base/z eliminate any CPU base-forward rounding discrepancy.
 f=traces[(case,'V1')]['actions'][0];o=f['observation'];z=f['z'];base=f['audit']['base'];inp=torch.cat([deploy(o),z,torch.zeros(1,32)],-1)
 group=[]
 for name in models:
  delta=ad[name](inp).detach().requires_grad_(True);raw=(base+delta).clamp(-1,1);a,au=env(base,raw)
  jac=torch.autograd.grad(a.sum(),delta,retain_graph=True)[0]
  qa=q(torch.cat([privileged(o),a,z,torch.zeros(1,32)],-1)).sum()
  g_exec=torch.autograd.grad(qa,a,retain_graph=True)[0];g_raw=torch.autograd.grad(qa,delta)[0]
  actual=traces[(case,name)]['actions'][0]['audit']
  err=float((a-actual['executed']).abs().max());rawerr=float((delta-actual['adapter']).abs().max())
  group.append(dict(model=name,q=float(qa),cpu_vs_saved_action_max_abs=err,cpu_vs_saved_adapter_max_abs=rawerr,trigger_count=int(au['triggered'].sum()),zero_jac_count=int((jac==0).sum()),joint=[dict(name=n,raw_adapter=float(delta[0,j]),raw_rad=float(delta[0,j]*sc[0,j]),action=float(a[0,j]),qPD=float(au['qexec'][0,j]),correction_rad=float((au['qexec']-au['qbase'])[0,j]),trigger=bool(au['triggered'][0,j]),jac_exec_raw=float(jac[0,j]),dQ_da=float(g_exec[0,j]),dQ_draw=float(g_raw[0,j])) for j,n in enumerate(names)]))
 firsts.append(dict(case=case,models=group))
 # Replay support: nearest whole normalized privileged vector and action distance;
 # nearest is diagnostic only, not claim state equality or causal coverage.
 X=torch.cat([privileged(r['observation']) for r in rows]);xo=privileged(o);std=X.std(0).clamp_min(.1);distance=((X-xo)/std).square().mean(-1).sqrt();ix=int(distance.argmin());mask=[i for i,r in enumerate(rows) if pid(r)==tr['task_spec']['parent_id']]
 support_dist.append(dict(case=case,nearest_global_index=ix,normalized_rms=float(distance[ix]),nearest_phase=phase(rows[ix]),nearest_parent=pid(rows[ix]),same_parent_min_distance=float(distance[mask].min()) if mask else None,
  action_distances=[dict(model=g['model'],rms_to_nearest_saved_action=float((torch.tensor([v['action'] for v in g['joint']])[None]-rows[ix]['action']).square().mean().sqrt())) for g in group]))
write('reward.json',reward);write('first-state-jacobian.json',firsts);write('joint-trajectory.json',jointrows);write('support-distance.json',support_dist)
with (O/'joint-trajectory.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(jointrows[0]),lineterminator="\n");w.writeheader();w.writerows(jointrows)
# Behavior successor residual: saved next actual action; not target-policy TD error.
res={};qvals={}
with torch.no_grad():
 for i,r in enumerate(rows[:-1]):
  nr=rows[i+1]
  if pid(r)!=pid(nr) or abs(float(nr['provenance']['time'])-float(r['provenance']['time'])-.02)>1e-6:continue
  no=r['next']['observation']
  if not all(torch.equal(no[k],nr['observation'][k]) for k in no):continue
  qa=q(torch.cat([privileged(r['observation']),r['action'],r['z'],r['zi']],-1))
  qb=targetq(torch.cat([privileged(no),nr['action'],nr['z'],nr['zi']],-1))
  y=r['aux_rewards']['contact_physical']+(.99**(r['duration']/.1))*(~r['next']['terminated'])*qb
  res.setdefault(phase(r),[]).append(float((qa-y).abs()));qvals.setdefault(phase(r),[]).append(float(qa))
write('behavior-td.json',dict(contract='Retrospective final-Q vs saved behavior successor targetQ. Not original target-policy TD, not historical optimizer residual.',abs_residual={k:summary(v) for k,v in res.items()},q={k:summary(v) for k,v in qvals.items()}))
# Earliest divergence; no intervention, only matched saved closed loops.
div=[]
for case in (2,3):
 v=traces[(case,'V1')]
 for name in ['control64','return64']:
  tr=traces[(case,name)];ap=np.array([c['q_PD'] for c in tr['controls']]);vp=np.array([c['q_PD'] for c in v['controls']]);qq=np.array([c['pre_qpos'] for c in tr['controls']]);vq=np.array([c['pre_qpos'] for c in v['controls']]);phys=tr['physics'];pv=v['physics']
  def onset(vals,threshold):
   good=np.array(vals)>threshold;ix=np.flatnonzero(good);return float(phys[int(ix[0])]['time']) if len(ix) else None
  div.append(dict(case=case,model=name,first_qPD_difference_max_rad=float(abs(ap[0]-vp[0]).max()),first_physics_qpos_difference_time=onset([max(abs(np.array(a['qpos'])-b['qpos'])) for a,b in zip(phys,pv)],0),first_root_1cm_difference_time=onset([np.linalg.norm(np.array(a['root'])-b['root']) for a,b in zip(phys,pv)],.01),first_slip_over_045_time=onset([a['stage_loaded_slip_mps'] for a in phys],.45),first_support_difference_time=next((float(a['time']) for a,b in zip(phys,pv) if a['support']!=b['support']),None),first_qpos_equal=bool(np.array_equal(qq[0],vq[0]))))
write('divergence.json',div)
assert not torch.cuda.is_initialized()
write('receipt.json',dict(wall_s=time.monotonic()-T,new_physics=0,new_updates=0,cuda_initialized=False,replay_rows=len(rows),source_runtime='f6bd9eb606975202b34e693968784308bad3745c',analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
print(json.dumps(dict(reward=reward,firsts=[dict(case=g['case'],models=[{k:v for k,v in a.items() if k!='joint'} for a in g['models']]) for g in firsts],divergence=div,wall_s=time.monotonic()-T)))
