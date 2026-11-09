"""Bounded physical batch + disposable FB updates; no formal policy training."""
import os,sys,json,time,hashlib,random,argparse,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def study(e,m,contract,out):
 import torch,numpy as np
 from p4000_env import Task
 from native_env import install,spec
 from retention_eval import MODELS
 from clean_policy import Expert
 from dual_track_runtime import PrivilegedBranch,TrainingBranch
 from dual_track_ppo import update
 from humanoidverse.force_compliance.parallel import merge_rollouts
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 install();torch.set_num_threads(1);torch.set_grad_enabled(False)
 write(out/'speed.json',speed(e));write(out/'backend.json',backend(e,m,P/'experiments/native-rolling-h100-20261107/compatibility-prefix'))
 path,digest=MODELS['BC500'];assert sha(path)==digest
 ck=torch.load(path,map_location='cpu',weights_only=True);net=PrivilegedBranch(ck).cuda();old=net.base
 task=Task(e,m,contract,5501);task.reset(record=True,label='benchmark')
 def obs():return torch.cat([task.student_features(),task.features()],-1)
 x=obs();assert x.shape==(1,2727)
 assert torch.equal(net.distribution(x).mean,old.distribution(x[:,:1639]).mean)
 for k in ['center','scale']:assert torch.equal(getattr(old,k).cpu(),ck['state_dict'][k])
 # Zero LAST layer only; prove task gradient reaches it and reaches earlier layer after one disposable step.
 with torch.enable_grad():
  xx=x.clone();oo=torch.optim.Adam(net.branch.parameters(),lr=1e-4)
  loss=(net.distribution(xx).mean-old.distribution(xx[:,:1639]).mean-.001).square().sum();loss.backward()
  g0=float(net.branch[-1].weight.grad.norm());assert g0>0;oo.step();oo.zero_grad()
  net.distribution(xx).mean.square().sum().backward();g1=float(net.branch[0].weight.grad.norm());assert g1>0
 net=PrivilegedBranch(ck).cuda();old=net.base
 rng=torch.Generator(device='cuda').manual_seed(5501)
 # Cross-reset snapshot physical/history/RNG replay, including task-owned generator.
 ss=TrainingBranch(task,[rng]);xx=obs().clone()
 def draws():return (random.random(),float(np.random.rand()),float(task.rng.random()),torch.rand(3).cpu(),torch.rand(3,device='cuda').cpu(),torch.rand(3,device='cuda',generator=rng).cpu())
 a=draws();raw=net.distribution(xx).mean;task.step(raw);q=task.e.sim.data.qpos.copy();zz=task.last_command.clone();f=obs().clone()
 task.reset(spec(.08,1.6,5.2),record=True,label='other_reset');ss.restore();b=draws();assert all(torch.equal(i,j) if torch.is_tensor(i) else i==j for i,j in zip(a,b));assert torch.equal(obs(),xx)
 task.step(raw);assert np.array_equal(q,e.sim.data.qpos) and torch.equal(zz,task.last_command) and torch.equal(f,obs());ss.restore()
 # Independent child evaluation; no training Task or any RNG touched by child.
 before=TrainingBranch(task,[rng]);dd=draws();before.restore()
 cp=out/'initial-branch.pt';torch.save({'state_dict':net.state_dict(),'base_sha':digest},cp)
 cmd=[sys.executable,str(ROOT/'research_notes/retention_eval.py'),'--roles','BC500','--output',str(out/'independent-eval')]
 t=time.monotonic()
 with open(out/'independent-eval.log','w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=400)
 assert before.digest()==before.fingerprint
 ee=draws();assert all(torch.equal(i,j) if torch.is_tensor(i) else i==j for i,j in zip(dd,ee));before.restore()
 write(out/'correctness.json',dict(zero_branch_exact=True,normalizer_exact=True,last_layer_gradient=g0,earlier_layer_gradient=g1,cross_reset_next_physics_exact=True,all_rng_exact=True,independent_evaluation_exact=True,eval_wall_s=time.monotonic()-t))
 opt=torch.optim.Adam([{'params':net.actor.parameters(),'lr':3e-5},{'params':net.critic.parameters(),'lr':3e-4}]);batch={k:[] for k in ['obs','next_obs','action','logp','value','next_value','reward','discount','done']};pool=[];reset_seconds=0
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.monotonic()
 for j in range(768):
  if task.done:
   rt=time.monotonic();task.reset(record=True,label='benchmark_'+str(j));reset_seconds+=time.monotonic()-rt
  x=obs().clone();d=net.distribution(x);raw=d.mean+d.stddev*torch.randn(d.mean.shape,device='cuda',generator=rng);lp=d.log_prob(raw).sum(-1);v=net.value(x)
  # Actual fifty-Hz obs/action transitions; each control record also carries full physical state.
  _,r,done,st=task.step(raw);nx=obs().clone();nv=net.value(nx)
  for k,vv in dict(obs=x[0],next_obs=nx[0],action=raw[0],logp=lp[0],value=v[0],next_value=nv[0]).items():batch[k].append(vv.cpu().numpy())
  batch['reward'].append(r);batch['discount'].append(.99**(st['control_steps']/5));batch['done'].append(done)
  for c in task.controls[-st['control_steps']:]:
   pool.append(c)
  if j%128==127:print('COLLECT',j+1,flush=True)
 torch.cuda.synchronize();sample_s=time.monotonic()-t
 bb={k:np.asarray(v,dtype=bool if k=='done' else np.float32) for k,v in batch.items()}
 t=time.monotonic();np.savez_compressed(out/'new-rollout.npz',**bb);torch.save(pool,out/'new-controls.pt');save_data=time.monotonic()-t
 # Complete original PPO batch; discarded benchmark update, not promoted model.
 merged={k:v.cuda() for k,v in merge_rollouts([dict(batch=bb)],.95).items()};merged['raw_adv_std']=(merged['returns']-merged['value']).std()
 torch.cuda.synchronize();t=time.monotonic();log=update(net,opt,merged,epochs=6,minibatch=256);torch.cuda.synchronize();update_s=time.monotonic()-t
 t=time.monotonic();torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict()),out/'disposable-high.pt');save_s=time.monotonic()-t
 t=time.monotonic();back=torch.load(out/'disposable-high.pt',map_location='cuda',weights_only=True);read_s=time.monotonic()-t;assert all(torch.equal(v,back['model'][k]) for k,v in net.state_dict().items())
 write(out/'high-benchmark.json',dict(decisions=768,controls=len(pool),sample_s=sample_s,reset_s=reset_seconds,update_s=update_s,data_save_s=save_data,checkpoint_save_s=save_s,checkpoint_read_s=read_s,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),update=log,compile='none; canonical row eager FP32',formal_training=False))
 return dict(status='completed',student_optimizer_updates=log['optimizer_updates'],benchmark_only=True,RL_decisions=768,formal_RL_decisions=0)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
 os.environ.update(CLEAN_KIND='direct',P4000_CANONICAL_ROW_ACTOR='1',BFM_RECOVERY_OUTPUT=str(out));os.chdir(out)
 import m10_stage,m10_physics
 m10_physics.OUT=out;m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
