"""Matched retrospective physical evaluation, no learning or Teacher search."""
import argparse,os,sys,json,hashlib,subprocess,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=ROOT.parent;OLD=P/'experiments/interactive-increment-20261108';PILOT=P/'experiments/interactive-contact-pilot-20261108'
MODELS={
 'BC500':(PILOT/'runs/bc-fit-01/old_data_only-500.pt','a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e'),
 's5401b4':(OLD/'runs/ppo-clean-5401/candidate-4.pt','40f91ceb142665b57a9e424cbccf3b2bc2e2a1c78d631d17c34c8d5ad6babb68'),
 's5401b7':(OLD/'runs/ppo-clean-5401/candidate-7.pt','882e32696fbad797d301c93cd9229d61feae20531f8f520983d935f077732954'),
 's5402b4':(OLD/'runs/ppo-clean-5402/candidate-4.pt','6cd5797ff629544da22dd8944e8cd8ddf44ac3455a1b1c613c6989951cb4351c')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def phase_quality(task):
 import numpy as np
 s=task.spec;stop=s.get('stop_time',s['onset']+s['ramp']);hold=stop+s.get('deceleration',0)
 spans=[('precontact',4,s['onset']),('startup',s['onset'],min(s['onset']+1,stop)),('sustained',min(s['onset']+1,stop),stop),('deceleration',stop,hold),('holding',hold,s['release']),('released',s['release'],s['end'])];res={}
 for name,a,b in spans:
  rr=[r for r in task.physics if a-1e-8<=r['force_time']<b-1e-8];cc=[r for r in task.controls if a-1e-8<=r['time']<b-1e-8]
  if not rr:continue
  active=[r for r in rr if r['active']];f=np.array([r['force'] for r in active]);err=np.array([np.array(r['contacts'][0]['anchor'])-r['contacts'][0]['point'] for r in active]);root=np.array([r['root'] for r in rr]);w=np.array([np.array(r['wrist'])-r['shoulder_xyz'] for r in rr]);qp=np.array([c['q_PD'] for c in cc]);q=np.array([r['qpos'][7:] for r in rr]);feet=np.array([r['feet_xyz'] for r in rr]);yaw=np.unwrap([np.arctan2(r['pelvis_matrix'][1][0],r['pelvis_matrix'][0][0]) for r in rr]);slip=[r.get('contact_tangent_speed_max_mps',0) for r in rr]
  res[name]=dict(start=a,end=b,physics=len(rr),contact_physics=len(active),force_xyz_rms=np.sqrt((f*f).mean(0)).tolist() if len(active) else None,error_xyz_rms_cm=(100*np.sqrt((err*err).mean(0))).tolist() if len(active) else None,error_xyz_signed_cm=(100*err.mean(0)).tolist() if len(active) else None,root_delta_cm=(100*(root[-1]-root[0])).tolist(),root_velocity_xyz_mean=np.mean([r['qvel'][:3] for r in rr],0).tolist(),wrist_relative_shoulder_delta_cm=(100*(w[-1]-w[0])).tolist(),yaw_deg_initial=float(np.degrees(yaw[0])),yaw_deg_final=float(np.degrees(yaw[-1])),yaw_excursion_deg=float(np.degrees(max(abs(yaw-yaw[0])))),slip_peak_mps=float(max(slip)),slip_p95_mps=float(np.quantile(slip,.95)),feet_path_cm=(100*np.linalg.norm(np.diff(feet,axis=0),axis=-1).sum(0)).tolist(),qpd_step_peak_deg=float(np.degrees(abs(np.diff(qp,axis=0)).max())) if len(qp)>1 else 0,actual_joint_5ms_peak_deg=float(np.degrees(abs(np.diff(q,axis=0)).max())) if len(q)>1 else 0)
 return res

def study(e,m,contract,out):
 import torch,numpy as np,random
 from native_env import Task,spec,install
 from clean_policy import Expert
 from native_transfer import phase_report
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 install();torch.set_num_threads(1);torch.set_grad_enabled(False);write(out/'speed.json',speed(e));write(out/'backend.json',backend(e,m,P/'experiments/native-rolling-h100-20261107/compatibility-prefix'))
 assert contract['model_sha256']=='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361'
 task=Task(e,m,contract,5417);specs={'short08':spec(.08,1.6,5.2),'normal72':spec(.72,4.8,5.),'stop66':spec(.66,4.4,5.4,stop=8.1),'none':spec(0,3,5.2,enabled=False)};rows=[];initials={};phys=0
 roles=ARGS.roles.split(',') if not ARGS.checkpoint else ['s5403b'+str(ARGS.iteration)]
 for role in roles:
  path,digest=MODELS[role] if not ARGS.checkpoint else (ARGS.checkpoint,ARGS.sha256)
  assert sha(path)==digest;ck=torch.load(path,map_location='cuda',weights_only=True);assert ck['input_dim']==1639
  net=Expert(1639,'direct').cuda();net.load_state_dict(ck['state_dict'],strict=True);net.eval().requires_grad_(False)
  assert torch.equal(net.center,ck['state_dict']['center']) and torch.equal(net.scale,ck['state_dict']['scale'])
  decisions={}
  for case,ss in specs.items():
   torch.manual_seed(5417);np.random.seed(5417);random.seed(5417);task.rng=np.random.default_rng(5417);task.reset(ss,record=True,label=role+'_'+case)
   initials[role+'_'+case]=task.prefix_data['audit'];dd=[]
   while not task.done:
    x=task.student_features();raw=net.distribution(x).mean
    dd.append(dict(time=float(e.tether.time),restricted=x[0].cpu().tolist(),last_z=task.last_command[0].cpu().tolist(),raw=raw[0].cpu().tolist()))
    _,_,_,st=task.step(raw);phys+=st['physics_steps']
   r=task.save(out);r.update(phase_report(task));r.update(role=role,case=case,model_sha256=digest,model_path=str(path),retrospective=ARGS.checkpoint is None,quality=phase_quality(task));rows.append(r);write(out/'summaries.json',rows);decisions[case]=dd
   print('RETENTION_EVAL',role,case,r['all']['error_cm'],r['all']['force_rms'],r['boundary'],flush=True)
  import gzip
  with gzip.open(out/(role+'-restricted-decisions.json.gz'),'wt') as f:json.dump(decisions,f)
 write(out/'initials.json',initials)
 return dict(status='completed',episodes=len(rows),physics_steps=phys,RL_decisions=0,Teacher_queries=0,retrospective=ARGS.checkpoint is None)
def main():
 global ARGS
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--roles',default='BC500,s5401b4,s5401b7,s5402b4');p.add_argument('--checkpoint',type=Path);p.add_argument('--sha256');p.add_argument('--iteration',type=int);ARGS=p.parse_args();out=ARGS.output;out.mkdir(exist_ok=False)
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain']);write(out/'launch.json',dict(runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),roles=ARGS.roles,checkpoint=str(ARGS.checkpoint),probability_contract='unchanged canonical-row',eval_seed=5417))
 os.environ['CLEAN_KIND']='direct';os.environ['P4000_CANONICAL_ROW_ACTOR']='1';os.environ['BFM_RECOVERY_OUTPUT']=str(out);os.chdir(out)
 import m10_stage,m10_physics
 m10_physics.OUT=out;m10_physics.MODEL=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view';m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
