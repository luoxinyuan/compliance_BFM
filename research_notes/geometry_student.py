"""Single shared ideal-force contact-slot Student. No partner, active, time or desired-yaw input."""
from pathlib import Path
import json,os,time,copy,gzip
import transfer_student_validated as runner
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/contact-geometry-20261103'

def mapped_student(ck,n,device):
 import torch
 from clean_policy import Expert
 old=ck['state_dict'];net=Expert(n,'direct').to(device);state=net.state_dict();mapping={i:i for i in range(465)}
 for i in range(100):
  for j in range(3):mapping[465+3*i+j]=721+9*i+j
 for i in range(256):mapping[765+i]=465+i
 for k,v in old.items():
  if k in ['center','scale']:
   for a,b in mapping.items():state[k][b]=v[a]
  elif k in ['actor.0.weight','critic.0.weight']:
   state[k].zero_()
   for a,b in mapping.items():state[k][:,b]=v[:,a]
  elif state[k].shape==v.shape:state[k].copy_(v)
  else:raise ValueError(k)
 for network in ['actor','critic']:
  constant=(torch.tensor([1.,0,0,0],device=device)-old['center'][1021:])/old['scale'][1021:]
  state[network+'.0.bias']+=old[network+'.0.weight'][:,1021:]@constant.clamp(-10,10)
 net.load_state_dict(state);return net

def study(env,model,contract,out):
 import torch,numpy as np
 from geometry_env import Task,PLAN
 from geometry_policy import expand
 from clean_policy import Expert,direct_command
 from m11_runtime_speed import install
 runner.write(out/'runtime-speed.json',install(env));torch.set_num_threads(1);torch.set_grad_enabled(False);torch.manual_seed(1307)
 selection=json.loads((R/'student-teacher-selection.json').read_text());task=Task(env,model,contract,1307);task.reset(PLAN["single"]);device=env.standing.device
 s0p=P/'bridge-v2-owned/runs/direct-transfer-20261103/A-prepare-01/job/ppo15-student-r0.pt';assert runner.sha(s0p)=='0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12';ck=torch.load(s0p,map_location=device,weights_only=True);student=mapped_student(ck,task.student_features().shape[-1],device);student.eval().requires_grad_(False);original=Expert(1025,'direct').to(device);original.load_state_dict(ck['state_dict']);original.eval().requires_grad_(False)
 identity=selection['teacher'];assert runner.sha(identity['path'])==identity['sha256'];tc=torch.load(identity['path'],map_location=device,weights_only=True);teacher=Expert(task.features().shape[-1],'direct').to(device);expand(teacher,tc);teacher.eval().requires_grad_(False)
 specs=copy.deepcopy(PLAN['dev']);legacy=json.loads(Path(__file__).with_name('heading_protocol.json').read_text());specs['local']=dict(legacy['regression']['local'],geometry='wrist');specs['straight']=dict(specs['straight'],end=20.05);specs['no_contact']=dict(specs['no_contact'],end=20.05);specs['early']=dict(specs['straight'],stop_time=7.2,deceleration=.8,release=10.,end=15.);specs['late']=dict(specs['straight'],stop_time=9.8,deceleration=.6,release=12.4,end=17.4)
 specs['pair_no_contact']=dict(specs['pair_translate'],enabled=False,distance=0.,rotation_deg=0.);summary=[];dataset=[];accepted=[]
 def run(name,spec,controller,collect=False,takeover=None):
  task.reset(spec,record=True,label=name);samples=[];parity=[]
  while not task.done:
   sx=task.student_features();tx=task.features();t=float(env.tether.time)
   if controller=='S0':
    RR=env.sim.data.xmat[1].reshape(3,3);hh=task.h.features(RR)[-100:,:3].reshape(-1);ox=torch.cat([task.pro.clamp(-10,10),torch.tensor(hh,device=device,dtype=torch.float32)[None],task.last_command/16,torch.tensor([[1.,0,0,0]],device=device)],-1);raw=original.distribution(ox).mean
    if spec.get('geometry')=='wrist':parity.append(float((raw-student.distribution(sx).mean).abs().max()))
   elif controller=='teacher':raw=teacher.distribution(tx).mean
   elif controller=='standing':raw=env.standing/16
   else:raw=student.distribution(sx).mean
   query=teacher.distribution(tx).mean if spec.get('geometry')=='torso_pair' and controller!='S0' and (collect or takeover is not None) else raw
   if takeover is not None and t>=takeover:raw=query
   if collect:samples.append(dict(x=sx[0].cpu().tolist(),z=direct_command(query,env.standing)[0].cpu().tolist(),time=t,obs={k:v.cpu().tolist() for k,v in env.obs.items() if k in env.keys},previous_z=task.last_command[0].cpu().tolist(),episode=name))
   task.step(raw)
  row=task.save(out);summary.append(row);runner.write(out/'summaries.json',summary)
  if parity:row['mapped_initial_parity_max']=max(parity)
  if samples:
   with gzip.open(out/(name+'-queries.json.gz'),'wt') as f:json.dump(samples,f)
  print('SHARED_CONTACT',name,row['error_cm'],row['sum_force_norm_rms'],row['boundary'],flush=True);return samples,row
 def fit(data,steps,name):
  x=torch.tensor([r['x'] for r in data],device=device);z=torch.tensor([r['z'] for r in data],device=device);groups=[torch.tensor([i for i,r in enumerate(data) if r['episode']==e],device=device) for e in sorted(set(r['episode'] for r in data))]
  student.requires_grad_(False);student.actor.requires_grad_(True);opt=torch.optim.Adam(student.actor.parameters(),lr=1e-4);curve=[]
  with torch.enable_grad():
   for step in range(steps):
    g=groups[step%len(groups)];ix=g[torch.randint(len(g),(128,),device=device)];zs=direct_command(student.distribution(x[ix]).mean,env.standing);loss=(zs-z[ix]).square().mean();opt.zero_grad();loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(student.actor.parameters(),1));opt.step()
    if step%200==0:curve.append(dict(step=step,loss=float(loss.detach()),gradient=grad))
  student.eval().requires_grad_(False)
  checks=[]
  for r in data[::max(1,len(data)//128)]:
   xx=torch.tensor(r['x'],device=device)[None];zt=torch.tensor(r['z'],device=device)[None];zs=direct_command(student.distribution(xx).mean,env.standing);ob={k:torch.tensor(v,device=device) for k,v in r['obs'].items()};qt=task.qof(model.act(ob,zt,mean=True));qs=task.qof(model.act(ob,zs,mean=True));checks.append(dict(z_angle_deg=float(torch.rad2deg(torch.acos(((zt*zs).sum()/256).clamp(-1,1)))),qpd_rms_deg=float(torch.rad2deg(qt-qs).square().mean().sqrt())))
  runner.write(out/(name+'-alignment.json'),checks)
  p=out/(name+'.pt');torch.save(dict(state_dict={k:v.cpu() for k,v in student.state_dict().items()},input_dim=x.shape[-1],kind='shared-contact-ideal-force-student',teacher=identity,updates=steps,seed=1307),p);runner.write(out/(name+'-training.json'),dict(updates=steps,samples=len(data),episodes=len(groups),sha256=runner.sha(p),curve=curve,extra_privileged_inputs=False))
 # Actual BFM permission audit reused, exact class assertions retained.
 assert set(env.keys)=={'state','last_action','history_actor'} and task.pro.shape[-1]==465
 runner.write(out/'observation-audit.json',dict(proprio='same465 audited encoder/IMU/action history',own_latent_memory=256,contact='100 real force samples x3 permanent slots x3 XYZ, transformed to current pelvis frame; per-slot mount identity/local position',mask='known instrumented mount, never active/contact intent',excluded=['anchor','partner velocity','time','target yaw','active truth','task ID','teacher history'],hardware='ideal force and known mounts in simulation; G1 sensing not established'))
 for name in ['straight','early','late','local','no_contact','pair_no_contact']:
  data,row=run('S0_source_'+name,specs[name],'S0',True);assert row.get('mapped_initial_parity_max',0)<1e-5
  if row['boundary'] is None:dataset+=data
 for name in selection['qualified_dev']:
  data,row=run('Teacher_source_'+name,specs[name],'teacher',True)
  if row['boundary'] is None:dataset+=data
 assert dataset
 fit(dataset,4000,'shared-r0')
 for name in ['straight','early','late','local','no_contact','pair_no_contact']+selection['qualified_dev']:run('Student_dev_'+name,specs[name],'student')
 # This bounded transfer has no automatic blind-label aggregation: physical teacher
 # takeover checks below characterize whether a later aggregation is justified.
 for name in selection['qualified_dev'][:2]:
  data,row=run('Student_visit_'+name,specs[name],'student',True)
  for t in [7.,11.]:run('Teacher_takeover_'+name+'_'+str(t),specs[name],'student',takeover=t)
 for name,spec in PLAN['final'].items():
  if spec.get('geometry')!='torso_pair':continue
  if not selection.get('rotation_qualified') and abs(spec.get('rotation_deg',0))>0 and spec['distance']==0:continue
  run('Teacher_final_'+name,spec,'teacher');run('Student_final_'+name,spec,'student')
 novel=dict(specs['pair_translate'],distance=.24,onset=4.45,ramp=8.05,release=14.5,end=19.5)
 run('Teacher_final_translation',novel,'teacher');run('Student_final_translation',novel,'student');run('Standing_final_translation',novel,'standing')
 result=dict(status='completed',decisions=0,student_optimizer_updates=4000,elapsed_s=time.monotonic()-runner.START,teacher_selection=selection);runner.write(out/'result.json',result);return result
runner.study=study
if __name__=='__main__':runner.main()
