"""Complete only unaffected frozen old-data BC control; no affected-model final exposure."""
import os,sys,json,hashlib,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
S=ROOT.parent;P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def study(e,m,contract,out):
 import torch
 from native_env import Task,spec,install
 from clean_policy import Expert
 from native_transfer import phase_report
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 torch.set_num_threads(1);torch.set_grad_enabled(False);install();write(out/'speed.json',speed(e));write(out/'backend.json',backend(e,m,P/'experiments/native-rolling-h100-20261107/compatibility-prefix'))
 selected=json.loads((S/'runs/bc-eval-01/selection.json').read_text())['old_data_only']['selected'];assert selected['step']==500
 paths={'S4000':P/'experiments/native-transfer-20261107/runs/bc-fit-01/S-4000.pt','old_BC500':Path(selected['model']['path'])};assert sha(paths['old_BC500'])==selected['model']['sha256'];write(out/'frozen-before-final.json',dict(selection='pre-existing3dev selected old-only500; no final consulted',models={k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()},affected_interactive_model_excluded=True))
 plan=json.loads((ROOT/'research_notes/interactive_protocol.json').read_text());cases={'mid24':spec(.24,3,5.3),'none':spec(0,3,5.2,enabled=False)}
 for name,c in plan['sealed_cases'].items():
  ss=spec(c['distance'],c['ramp'],c['onset'])
  if 'stop_time' in c:ss.update(stop_time=c['stop_time'],deceleration=c['deceleration'],release=c['stop_time']+c['deceleration']+2,end=c['stop_time']+c['deceleration']+6)
  cases['FINAL_'+name]=ss
 task=Task(e,m,contract,5301);rows=[];physical=reset=0
 for role,path in paths.items():
  ck=torch.load(path,map_location='cuda',weights_only=True);net=Expert(1639,'direct').cuda();net.load_state_dict(ck['state_dict']);net.eval().requires_grad_(False)
  for name,s in cases.items():
   task.reset(s,record=True,label=role+'_'+name);reset+=600+len(task.prefix_data['physics'])
   while not task.done:
    _,_,_,st=task.step(net.distribution(task.student_features()).mean);physical+=st['physics_steps']
   row=copy.deepcopy(task.save(out));row.update(phase_report(task));row.update(model_sha256=sha(path),role=role,development_or_final='sealed final' if name.startswith('FINAL_') else 'regression',not_new_training=True);rows.append(row);write(out/'summaries.json',rows);print('VALID_CONTROL',role,name,row['all']['error_cm'],row['all']['force_rms'],row['boundary'],flush=True)
 return dict(status='completed',student_optimizer_updates=0,new_searches=0,new_RL_decisions=0,actual_evaluation_physics_steps=physical,reset_physics_steps=reset,models=2,cases=len(cases),no_reselection_after_final=True)
def main():
 import argparse,subprocess
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain']);write(a.output/'launch.json',dict(runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),scope='unaffected frozen controls only; no extra BC/PPO'))
 os.environ['BFM_RECOVERY_OUTPUT']=str(a.output);os.chdir(a.output)
 import m10_stage,m10_physics
 m10_physics.OUT=a.output;m10_physics.MODEL=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view';m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
