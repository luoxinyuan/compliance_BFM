"""Uninterrupted Student vs cross-reset no-intervention replay; no PPO/BC/search."""
import os,sys,json,gzip,hashlib,copy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
S=ROOT.parent;P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def study(e,m,contract,out):
 import torch,numpy as np
 from native_env import Task,spec,install
 from clean_policy import Expert
 from rolling_snapshot import BranchState
 from interactive_snapshot import CrossResetState
 from native_transfer import phase_report
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 torch.set_num_threads(1);torch.set_grad_enabled(False);install();write(out/'speed.json',speed(e));write(out/'backend.json',backend(e,m,P/'experiments/native-rolling-h100-20261107/compatibility-prefix'))
 ckpath=P/'experiments/native-transfer-20261107/runs/bc-fit-01/S-4000.pt';assert hashlib.sha256(ckpath.read_bytes()).hexdigest()=='8f5e7287e974deedba1a8d841dd65d7692e944e278bd4a091f1ec224f167a479';ck=torch.load(ckpath,map_location='cuda',weights_only=True);net=Expert(1639,'direct').cuda();net.load_state_dict(ck['state_dict']);net.eval().requires_grad_(False)
 task=Task(e,m,contract,5301);records=[];counts=dict(new_searches=0,new_RL_decisions=0,Student_supervised_updates=0,actual_validation_physics_steps=0,reset_physics_steps=0)
 def step(n=1):
  for _ in range(n):
   _,_,_,st=task.step(net.distribution(task.student_features()).mean);counts['actual_validation_physics_steps']+=st['physics_steps']
 def reset(ss):
  task.reset(ss,record=True,label='validation');counts['reset_physics_steps']+=600+len(task.prefix_data['physics'])
 def segment(n0,c0):return copy.deepcopy(task.physics[n0:]),copy.deepcopy(task.controls[c0:])
 def compare(a,b):
  ar,ac=a;br,bc=b;assert len(ar)==len(br) and len(ac)==len(bc)
  fields=['qpos','qvel','force','anchor','feet_xyz','foot_normal_force_n','torque'];ans={k:float(np.max(abs(np.array([r[k] for r in ar])-np.array([r[k] for r in br])))) for k in fields}
  ans['qpd']=float(np.max(abs(np.array([c['q_PD'] for c in ac])-np.array([c['q_PD'] for c in bc]))));ans['all_physics_rows_equal']=ar==br;ans['all_control_rows_equal']=ac==bc;return ans
 for name,ss,t in [('normal_pre',spec(.72,4.8,5.),4.7),('short_post',spec(.08,1.6,5.2),5.4),('normal_post',spec(.72,4.8,5.),6.)]:
  reset(ss)
  while e.tether.time<t-1e-7:step()
  old=BranchState(task);fixed=CrossResetState(task);n0=len(task.physics);c0=len(task.controls)
  step(20);reference=segment(n0,c0)
  # A different actual episode, including contact; not a qpos-only perturbation.
  reset(spec(.24,3,5.3))
  while e.tether.time<6.-1e-7:step()
  old.restore();step(20);broken=segment(n0,c0);old_cmp=compare(reference,broken)
  fixed.restore();step(20);correct=segment(n0,c0);fixed_cmp=compare(reference,correct)
  assert fixed_cmp['all_physics_rows_equal'] and fixed_cmp['all_control_rows_equal'],fixed_cmp
  row=dict(case=name,time=t,legacy_cross_reset=old_cmp,fixed_cross_reset=fixed_cmp,proof='uninterrupted actual Student vs exact cross-reset continuation; callback/spec/origin graph restored',no_new_Teacher_label=True)
  records.append(row);write(out/'validation.json',records);print('EXACT_CROSS_RESET',name,old_cmp,fixed_cmp,flush=True)
 write(out/'counts.json',counts);return dict(status='completed',counts=counts,student_optimizer_updates=0,all_fixed_exact=True,legacy_cross_reset_unsafe=True)
def main():
 import argparse,subprocess
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 write(a.output/'launch.json',dict(runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='repair validation only, no extra BC/PPO/search'))
 os.environ['BFM_RECOVERY_OUTPUT']=str(a.output);os.chdir(a.output)
 import m10_stage,m10_physics
 m10_physics.OUT=a.output;m10_physics.MODEL=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view';m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
