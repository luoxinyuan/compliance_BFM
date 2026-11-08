"""Conditional Issue53 pilot: restricted Student's own true raw-Gaussian PPO, no expert overrides."""
import argparse,copy,json,os,sys,time,hashlib,gzip
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
S=ROOT.parent;P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def study(e,m,contract,out):
 if (S/'recovery-validity-erratum.json').exists():raise RuntimeError('Archived mixed dataset protocol affected; corrected labels require a separately authorized training budget')
 import torch,numpy as np
 from clean_policy import Expert,direct_command
 from native_env import spec,install
 from p4000_env import Task
 from p4000_critic import update
 from humanoidverse.force_compliance.parallel import merge_rollouts
 from native_transfer import phase_report
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 install();torch.set_num_threads(1);torch.set_grad_enabled(False);write(out/'speed.json',speed(e));write(out/'backend.json',backend(e,m,P/'experiments/native-rolling-h100-20261107/compatibility-prefix'))
 gate=json.loads((S/'runs/bc-eval-01/ppo-gate.json').read_text());assert gate['PPO_allowed']
 path=Path(gate['selected']['path']);assert sha(path)==gate['selected']['sha256'];ck=torch.load(path,map_location='cuda',weights_only=True)
 torch.manual_seed(ARGS.seed);net=Expert(1639,'direct').cuda();net.load_state_dict(ck['state_dict']);torch.manual_seed(ARGS.seed+17);fresh=Expert(1639,'direct').cuda();net.critic.load_state_dict(fresh.critic.state_dict());del fresh
 net.requires_grad_(False);net.actor.requires_grad_(True);net.critic.requires_grad_(True);net.log_std.fill_(np.log(.008))
 opt=torch.optim.Adam([{'params':net.actor.parameters(),'lr':3e-5},{'params':net.critic.parameters(),'lr':3e-4}]);task=Task(e,m,contract,ARGS.seed);task.reset(record=True,label='train_0');rng=torch.Generator(device='cuda').manual_seed(ARGS.seed*997)
 summary=[];logs=[];decisions=0;phase_counts={};complete=0;prefill=task.total_prefill;source=task.source_physics_total
 specs={'short08':spec(.08,1.6,5.2),'normal72':spec(.72,4.8,5.),'stop66':spec(.66,4.4,5.4,stop=8.1),'none':spec(0,3,5.2,enabled=False)}
 def save(i):
  p=out/('candidate-'+str(i)+'.pt');torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},optimizer=opt.state_dict(),input_dim=1639,kind='direct',role='restricted ideal-force Student PPO',seed=ARGS.seed,iteration=i,RL_decisions=decisions,start_sha256=sha(path),runtime=RUNTIME,critic_permission='same restricted1639; no additional privileged input; physical reward privileged',actor_permission='restricted1639 audit',no_online_teacher=True),p);return p
 def evaluate(i):
  for name,ss in specs.items():
   task.reset(ss,record=True,label=f'dev{i}_{name}')
   while not task.done:task.step(net.distribution(task.student_features()).mean)
   r=task.save(out);r.update(phase_report(task));r['iteration']=i;r['seed']=ARGS.seed;summary.append(r);write(out/'summaries.json',summary);print('STUDENT_PPO_DEV',i,name,r['all']['error_cm'],r['all']['force_rms'],r['boundary'],flush=True)
 save(0);evaluate(0);task.reset(record=True,label='train_0')
 for iteration in range(1,9):
  b={k:[] for k in ['obs','next_obs','action','logp','value','next_value','reward','discount','done']};stats=[];beg=time.monotonic()
  for j in range(768):
   if task.done:task.reset(record=True,label=f'train_{complete}')
   x=task.student_features().detach().clone();d=net.distribution(x);raw=d.mean+d.stddev*torch.randn(d.mean.shape,device='cuda',generator=rng);lp=d.log_prob(raw).sum(-1);value=net.value(x);_,r,done,st=task.step(raw);nx=task.student_features().detach().clone();nv=net.value(nx)
   for k,v in dict(obs=x[0],next_obs=nx[0],action=raw[0],logp=lp[0],value=value[0],next_value=nv[0]).items():b[k].append(v.cpu().numpy())
   b['reward'].append(r);b['discount'].append(.99**(st['control_steps']/5));b['done'].append(done)
   zm=direct_command(d.mean,e.standing);zs=direct_command(raw,e.standing);st.update(raw_mean_norm=float(d.mean.norm()),raw_sample_norm=float(raw.norm()),z_sample_mean_angle_deg=float(torch.rad2deg(torch.acos(((zm*zs).sum()/256).clamp(-1,1)))),executed_std=float(d.stddev.mean()),Student_actor_observation_dim=1639)
   stats.append(st);decisions+=1
   key=st.get('phase','unknown');phase_counts[key]=phase_counts.get(key,0)+1
   if done:
    complete+=1
    if task.boundary:task.save(out)
  batch0={k:np.asarray(v,dtype=bool if k=='done' else np.float32) for k,v in b.items()};np.savez_compressed(out/f'rollout-{iteration}.npz',**batch0);write(out/f'rollout-{iteration}-stats.json',stats)
  batch={k:v.cuda() for k,v in merge_rollouts([dict(batch=batch0)],.95).items()}
  # Fresh value prefit uses actual on-policy batch; actor is frozen, then original KL/Adam guard.
  if iteration==1:
   before={k:v.detach().clone() for k,v in net.actor.state_dict().items()}
   with torch.enable_grad():
    for _ in range(100):
     opt.zero_grad(set_to_none=True);loss=.5*((net.value(batch['obs'])-batch['returns'])/10).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(net.critic.parameters(),1);opt.step()
   assert all(torch.equal(v,before[k]) for k,v in net.actor.state_dict().items())
  log=update(net,opt,batch,epochs=6,minibatch=256,critic_shuffle_seed=ARGS.seed*10000+iteration);log.update(iteration=iteration,RL_decisions=decisions,rollout_seconds=time.monotonic()-beg,complete_episodes=complete,phase_counts=copy.deepcopy(phase_counts));logs.append(log);write(out/'learning.json',logs);save(iteration);print('STUDENT_PPO',ARGS.seed,iteration,decisions,log['optimizer_updates'],log['updated_value_EV'],flush=True)
  if iteration in [4,8]:evaluate(iteration);task.reset(record=True,label=f'train_afterdev{iteration}')
 # Fixed8 is main endpoint; dev4 retained, no final-driven selection.
 return dict(status='completed',student_optimizer_updates=0,RL_decisions=decisions,seed=ARGS.seed,actor_permission='restricted only',critic_permission='restricted only; privileged physical reward',phase_counts=phase_counts,complete_episodes=complete,reset_prefill_physics_steps=task.total_prefill-prefill,source_physics_steps=task.source_physics_total-source)
def main():
 global ARGS,RUNTIME
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,choices=[5301,5302],required=True);ARGS=p.parse_args();out=ARGS.output;out.mkdir(exist_ok=False);os.environ['P4000_CANONICAL_ROW_ACTOR']='1'
 import subprocess
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain']);RUNTIME=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();write(out/'launch.json',dict(runtime=RUNTIME,seed=ARGS.seed,source_sha256=sha(__file__),scope='conditional Student PPO only; no Expert PPO',max_decisions=6144))
 os.environ['BFM_RECOVERY_OUTPUT']=str(out);os.chdir(out)
 import m10_stage,m10_physics
 m10_physics.OUT=out;m10_physics.MODEL=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view';m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
