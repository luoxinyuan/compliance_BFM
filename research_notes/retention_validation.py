"""Frozen paired evaluation only. Original Task/control/physics implementations unchanged."""
import argparse,os,sys,json,hashlib,subprocess,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
from retention_eval import phase_quality
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=ROOT.parent
PLAN=json.loads(Path(__file__).with_name('retention_validation_protocol.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def study(e,m,contract,out):
 import torch,numpy as np
 from native_env import Task,spec,install
 from clean_policy import Expert
 from native_transfer import phase_report
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 install();torch.set_num_threads(1);torch.set_grad_enabled(False)
 write(out/'speed.json',speed(e));write(out/'backend.json',backend(e,m,P/'experiments/native-rolling-h100-20261107/compatibility-prefix'))
 assert contract['model_sha256']==PLAN['BFM_sha256']
 ident=PLAN['models'][ARGS.role];path=P/ident['path'];assert sha(path)==ident['sha256'];ck=torch.load(path,map_location='cuda',weights_only=True);assert ck['input_dim']==1639
 net=Expert(1639,'direct').cuda();net.load_state_dict(ck['state_dict'],strict=True);net.eval().requires_grad_(False)
 assert all(torch.equal(getattr(net,k),ck['state_dict'][k]) for k in ['center','scale'])
 task=Task(e,m,contract,PLAN['eval_seed']);rows=[];initials={};physics=0
 for item in PLAN['conditions']:
  seed=PLAN['eval_seed'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed);task.rng=np.random.default_rng(seed)
  ss=spec(**item['args']);task.reset(ss,record=True,label=ARGS.role+'_'+item['name'])
  initial=dict(audit=task.prefix_data['audit'],restricted=task.student_features().cpu().tolist(),root=e.sim.data.qpos[:3].tolist(),wrist=task.initial.get('wrist'),source_physics_steps=len(task.prefix_data['physics']))
  initial['restricted_sha']=hashlib.sha256(json.dumps(initial['restricted']).encode()).hexdigest();initials[item['name']]=initial;write(out/'initials.json',initials)
  while not task.done:
   raw=net.distribution(task.student_features()).mean
   _,_,_,st=task.step(raw);physics+=st['physics_steps']
  r=task.save(out);r.update(phase_report(task));r.update(role=ARGS.role,case=item['name'],family=item['family'],model_sha256=ident['sha256'],model_path=str(path),quality=phase_quality(task),evaluation_role='fixed new conditions, not used for candidate selection',runtime=RUNTIME)
  rows.append(r);write(out/'summaries.json',rows);print('NEW_CONDITION',ARGS.role,item['name'],r['all']['error_cm'],r['all']['force_rms'],r['boundary'],flush=True)
 assert sha(path)==ident['sha256']
 return dict(status='completed',episodes=len(rows),physics_steps=physics,RL_decisions=0,Teacher_queries=0,supervised_updates=0,model_unchanged=True,protocol_sha256=sha(Path(__file__).with_name('retention_validation_protocol.json')))
def main():
 global ARGS,RUNTIME
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--role',choices=list(PLAN['models']),required=True);ARGS=p.parse_args();out=ARGS.output;out.mkdir(exist_ok=False)
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain']);RUNTIME=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 write(out/'launch.json',dict(runtime=RUNTIME,source_sha256=sha(__file__),role=ARGS.role,model=PLAN['models'][ARGS.role],protocol_sha256=sha(Path(__file__).with_name('retention_validation_protocol.json')),seed=PLAN['eval_seed'],independent_process=True))
 os.environ['CLEAN_KIND']='direct';os.environ['P4000_CANONICAL_ROW_ACTOR']='1';os.environ['BFM_RECOVERY_OUTPUT']=str(out);os.chdir(out)
 import m10_stage,m10_physics
 m10_physics.OUT=out;m10_physics.MODEL=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view';m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
