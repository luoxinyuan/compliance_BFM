"""Verify early actual Teacher labels under fresh continuous correct Task, not cross-reset reuse."""
import os,sys,json,gzip,hashlib,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
S=ROOT.parent;P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def study(e,m,contract,out):
 import torch,numpy as np
 from native_env import Task,install
 from clean_policy import Expert
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 from native_transfer import phase_report
 torch.set_num_threads(1);torch.set_grad_enabled(False);install();write(out/'speed.json',speed(e));write(out/'backend.json',backend(e,m,P/'experiments/native-rolling-h100-20261107/compatibility-prefix'))
 D=S/'runs/diagnose-01';path=D/'Teacher_replan_2s_normal72_4.7.json.gz'
 with gzip.open(path,'rt') as f:ep=json.load(f)
 qs=json.loads((D/'searches.json').read_text());chosen={round(q['time'],6):q for q in qs if q['label'] in ['shadow_normal72_4.7','Teacher_replan_2s_normal72_4.7']}
 ckpath=P/'experiments/native-transfer-20261107/runs/bc-fit-01/S-4000.pt';assert hashlib.sha256(ckpath.read_bytes()).hexdigest()=='8f5e7287e974deedba1a8d841dd65d7692e944e278bd4a091f1ec224f167a479';ck=torch.load(ckpath,map_location='cuda',weights_only=True);net=Expert(1639,'direct').cuda();net.load_state_dict(ck['state_dict']);net.eval().requires_grad_(False)
 task=Task(e,m,contract,5301);task.reset(ep['condition'],record=True,label='continuous_early_label_replay');errors={k:0. for k in ['privileged','restricted','previous_z','qpos','qvel','qpd','force','active_anchor']};j=0;n=0
 while not task.done:
  t=float(e.tether.time);key=round(t,6)
  if key in chosen:
   q=chosen[key];n+=1;errors['privileged']=max(errors['privileged'],float((task.features()-torch.tensor(q['privileged_input'],device='cuda')).abs().max()));errors['restricted']=max(errors['restricted'],float((task.student_features()-torch.tensor(q['restricted_input'],device='cuda')).abs().max()));errors['previous_z']=max(errors['previous_z'],float((task.last_command-torch.tensor(q['previous_z'],device='cuda')).abs().max()))
  c=ep['control'][5*j];errors['qpos']=max(errors['qpos'],float(abs(e.sim.data.qpos-np.array(c['pre_qpos'])).max()));errors['qvel']=max(errors['qvel'],float(abs(e.sim.data.qvel-np.array(c['pre_qvel'])).max()))
  raw=torch.tensor(ep['decisions'][j]['raw'],device='cuda',dtype=torch.float32)[None] if 4.7-1e-7<=t<6.7-1e-7 else net.distribution(task.student_features()).mean
  _,_,_,st=task.step(raw);cs=ep['control'][5*j:5*j+st['control_steps']];errors['qpd']=max(errors['qpd'],float(abs(np.array([c['q_PD'] for c in cs])-np.array([c['q_PD'] for c in task.controls[-st['control_steps']:]])).max()))
  rs=ep['physics'][20*j:20*j+st['physics_steps']]
  for a,b in zip(rs,task.physics[-st['physics_steps']:]):
   errors['force']=max(errors['force'],float(abs(np.array(a['force'])-b['force']).max()))
   if a['active']:errors['active_anchor']=max(errors['active_anchor'],float(abs(np.array(a['anchor'])-b['anchor']).max()))
  j+=1
 assert n==20 and max(errors.values())==0.,(n,errors)
 row=task.save(out);row.update(phase_report(task));write(out/'verification.json',dict(valid_early_labels=20,errors=errors,actual_physics_steps=len(task.physics),reset_physics_steps=600+len(task.prefix_data['physics']),actual_Student_after6p7=True,scope='first2s command reproduction with identical original Teacher inputs; after6.7 actually recompute frozen Student; no new search/labels/training',metrics=row));return dict(status='completed',student_optimizer_updates=0,new_queries=0,all_errors_zero=True)
def main():
 import argparse,subprocess
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain']);write(a.output/'launch.json',dict(runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()))
 os.environ['BFM_RECOVERY_OUTPUT']=str(a.output);os.chdir(a.output)
 import m10_stage,m10_physics
 m10_physics.OUT=a.output;m10_physics.MODEL=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view';m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
