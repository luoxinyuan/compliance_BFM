"""Fixed models; actual shared partner prefix and real native/moving source states."""
import json,copy,time,hashlib,gzip
from pathlib import Path
import transfer_student_validated as runner

def conditions():
 p=json.loads(Path(__file__).with_name('adapter_protocol.json').read_text());dev=json.loads(Path(__file__).with_name('prior_training.json').read_text())['dev'];base=dev['walk'];s=p['new_conditions'];out={}
 for d,v in zip(s['distances_cm'],s['peak_speeds']):
  ramp=1.5*d/100/v;o=s['onset'];out['new'+str(d)]=dict(base,distance=d/100,onset=o,ramp=ramp,release=o+ramp+2,end=o+ramp+6)
 pair=s['common_prefix'];out['continue']=dict(base,**{k:v for k,v in pair.items() if k not in ['branch','deceleration']});out['stop']=dict(out['continue'],stop_time=pair['branch'],deceleration=pair['deceleration'])
 out['armready22']=dict(out['new22'],onset=6.85,release=out['new22']['release']+2,end=out['new22']['end']+2,ready_progress=.7)
 out['moving65']=dict(out['new65'],learned_handoff=6.5)
 out['reg5N_8']=dict(dev['arm5'],distance=.08,total_cap=5.)
 return out

def setup(env,model,contract):
 import torch
 from transition_env import Task,install_frame
 from m11_runtime_speed import install
 install_frame();import geometry_env,reach_contact;geometry_env.install_contact=reach_contact.install
 torch.set_num_threads(1);torch.set_grad_enabled(False);install(env)
 cfg=json.loads(Path(__file__).with_name('recovery_protocol.json').read_text());from adapter_policy import load
 ident=cfg['models']['I'];assert runner.sha(ident['path'])==ident['sha256'];I=load(torch.load(ident['path'],map_location=env.standing.device,weights_only=True),env.standing.device)
 native=json.loads((Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/latent-prior-audit-20261105')/'qualify-02/native-sources.json').read_text());goal=torch.tensor(native['goals']['0.05'],device=env.standing.device)
 return Task(env,model,contract,0),I,goal

def episode(task,I,goal,net,spec,label,out,student=False,collect=False):
 import torch,numpy as np
 from clean_policy import direct_command
 from locomotion_rollout import spherical
 import reach_contact
 task.reset(spec,record=True,label=label);env=task.e;records=[];prefix=[];changes=[]
 while not task.done:
  t=float(env.tether.time);tx=task.features();sx=task.student_features();raw=net.distribution(sx if student else tx).mean
  if 'ready_progress' in spec and t<6.-1e-7:
   u,_=reach_contact.smooth(t,4.,1.5);raw=spherical(env.standing,goal,spec['ready_progress']*u)/16
  elif t<spec.get('learned_handoff',0)-1e-7:raw=I.distribution(tx).mean
  if t<7.05-1e-8 and label.endswith(('__stop','__continue')):
   prefix.append(dict(t=t,x=tx[0].tolist(),raw=raw[0].tolist(),q=env.sim.data.qpos.tolist(),dq=env.sim.data.qvel.tolist(),previous_z=task.last_command.tolist()))
  if collect:
   records.append(dict(x=sx[0].tolist(),z=direct_command(raw,env.standing)[0].tolist(),time=t,previous_z=task.last_command[0].tolist(),bfm_obs={k:v.tolist() for k,v in env.obs.items() if k in env.keys}))
  iraw=I.distribution(tx).mean;zz=direct_command(raw,env.standing);iz=direct_command(iraw,env.standing)
  changes.append(dict(time=t,raw_norm=float(raw.norm()),delta_raw_norm=float((raw-iraw).norm()),z_angle_from_I_deg=float(torch.rad2deg(torch.acos((zz*iz).sum().div(256).clamp(-1,1))))))
  task.step(raw)
 row=task.save(out)
 if prefix:
  phys=[r for r in task.physics if r['force_time']<7.05-1e-8];payload=json.dumps(dict(decisions=prefix,physics=phys),sort_keys=True).encode();row['common_prefix_sha256']=hashlib.sha256(payload).hexdigest();row['prefix_decisions']=len(prefix);row['prefix_physics']=len(phys)
 runner.write(out/(label+'-correction.json'),changes)
 if collect:
  with gzip.open(out/(label+'-labels.json.gz'),'wt') as f:json.dump(records,f)
 print('ADAPTER_PHYSICS',label,row['error_cm'],row['force_rms'],row['boundary'],flush=True)
 return row,records

def study(env,model,contract,out):
 import torch
 from adapter_policy import load
 task,I,goal=setup(env,model,contract);manifest=json.loads(Path(runner.ARGS.resume).read_text());nets={}
 for k,d in manifest.items():
  assert runner.sha(d['path'])==d['sha256'];nets[k]=load(torch.load(d['path'],map_location=env.standing.device,weights_only=True),env.standing.device)
 rows=[]
 for name,spec in conditions().items():
  for k,n in nets.items():
   if time.monotonic()-runner.START>runner.ARGS.max_seconds-45:raise TimeoutError('bounded final')
   r,_=episode(task,I,goal,n,spec,k+'__'+name,out);rows.append(r);runner.write(out/'summaries.json',rows)
 checks=[]
 for k in nets:
  pair=[r for r in rows if r['label'] in [k+'__stop',k+'__continue']];assert len(pair)==2;assert pair[0]['common_prefix_sha256']==pair[1]['common_prefix_sha256'],'Future condition leaked into prefix';checks.append(dict(model=k,sha256=pair[0]['common_prefix_sha256'],identical=True))
 runner.write(out/'prefix-parity.json',checks);runner.write(out/'models.json',manifest)
 return dict(status='completed',episodes=len(rows),updates=0,elapsed_s=time.monotonic()-runner.START)
runner.study=study
if __name__=='__main__':runner.main()
