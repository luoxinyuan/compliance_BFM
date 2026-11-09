"""Exact fresh-process replay validation of saved M1 full-state assets; zero queries."""
import os,sys,json,hashlib,argparse,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');MAIN=P/'experiments/source-coverage-m1-20261109'
def write(p,x):Path(p).write_text(json.dumps(x,indent=2))
def study(e,m,contract,out):
 import numpy as np,torch,random,mujoco
 from native_env import Task,install
 from interactive_snapshot import CrossResetState
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 torch.set_num_threads(1);torch.set_grad_enabled(False);install();speed(e);backend(e,m,P/'experiments/native-rolling-h100-20261107/compatibility-prefix');task=Task(e,m,contract,6101);checks=[];steps=0
 for source in ['standing','goal_arm5','motion_walk_task']:
  qdir=MAIN/'runs'/(source+'-r2')/'query-001';state=torch.load(qdir/'state.pt',map_location='cpu',weights_only=False);plan=torch.load(qdir/'plans.pt',map_location='cpu',weights_only=False)
  torch.manual_seed(6101);np.random.seed(6101);random.seed(6101);task.rng=np.random.default_rng(6101);task.reset(state['spec'],record=True,label='exact_replay_'+source)
  for d in state['actual_prefix_decisions']:
   _,_,done,st=task.step(torch.tensor(d['raw'],device=e.standing.device,dtype=torch.float32)[None]);steps+=st['physics_steps'];assert not done
  errors={}
  integ=np.empty_like(state['integration']);mujoco.mj_getState(e.sim.model,e.sim.data,integ,mujoco.mjtState.mjSTATE_INTEGRATION);errors['integration']=float(np.max(abs(integ-state['integration'])))
  for k in ['qpos','qvel','qacc','qacc_warmstart','qfrc_constraint']:errors[k]=float(np.max(abs(getattr(e.sim.data,k)-state['solver_arrays'][k])))
  errors['previous_z']=float((task.last_command.cpu()-plan['previous_z']).abs().max());errors['restricted']=float((task.student_features().cpu()-plan['restricted']).abs().max());errors['privileged']=float((task.features().cpu()-plan['privileged']).abs().max())
  for k in e.keys:errors['obs_'+k]=float((e.obs[k].cpu()-plan['BFM_obs'][k]).abs().max())
  ss=CrossResetState(task);match=ss.fingerprint==state['fingerprint'];checks.append(dict(source=source,asset=str(qdir),max_errors=errors,exact_digest=match,steps=len(state['actual_prefix_decisions']),no_new_search=True));write(out/'checks.json',checks)
  assert max(errors.values())==0 and match,(source,errors,match)
 return dict(status='completed',checks=checks,actual_replay_physics=steps,queries=0,training=0)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);write(a.output/'launch.json',dict(runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()));os.environ['CLEAN_KIND']='direct';os.environ['P4000_CANONICAL_ROW_ACTOR']='1';os.environ['BFM_RECOVERY_OUTPUT']=str(a.output);os.chdir(a.output)
 import m10_stage,m10_physics
 m10_physics.OUT=a.output;m10_physics.MODEL=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view';m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
