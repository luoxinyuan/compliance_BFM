"""Query the existing expert on saved real states using separate MuJoCo data.
No simulation, no model evaluation. Base BFM action is the recorded corresponding command.
"""
import json
from pathlib import Path
import numpy as np,mujoco
from ccl02a_teacher import arm_ik,ReleaseFade
from accumulated_controller import AccumulatedCorrection,executable_bounds
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUN=P/'bridge-v2-owned/runs/m1-state-response-20261031A'
ROOT=P/'bridge-v2-owned/trees/m1-state-response/repo'
def replay_query(ep,model,contract):
 d=mujoco.MjData(model);ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in contract['dof_names']];vadr=model.jnt_dofadr[ids];limits=model.jnt_range[ids];arm=np.arange(22,29);body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'right_wrist_yaw_link');sc=np.array(contract['q_PD_scale'])[0];off=np.array(contract['q_PD_offset'])[0]
 fade=ReleaseFade();accum=AccumulatedCorrection([.2]*7);corr=np.zeros(7);targets=[];point_error=0.;acc=ep['controller']=='accumulated_goal'
 for i,c in enumerate(ep['control']):
  d.qpos[:]=c['pre_qpos'];d.qvel[:]=c['pre_qvel'];mujoco.mj_forward(model,d);point=d.xpos[body];point_error=max(point_error,float(abs(point-np.array(c['pre_wrist'])).max()));error=np.array(c['pre_anchor'])-point;active=c['active'];now=c['time'];qbase=np.array(c['base_q_PD']);bounds=executable_bounds(qbase[arm],limits[arm],sc[arm],off[arm],contract['clip']) if acc else None
  if i%5==0:
   desired=np.zeros(7)
   if active:
    jac=np.zeros((3,model.nv));mujoco.mj_jac(model,d,jac,None,point,body);desired=arm_ik(jac[:,vadr[arm]],error)
    if not acc:desired=np.clip(qbase[arm]+desired,limits[arm,0],limits[arm,1])-qbase[arm]
   if acc:accum.propose(desired,active,*bounds)
   else:corr=fade.update(now,active,desired)
  if acc:corr,_=accum.tick(now,active,*bounds)
  elif fade.release_start is not None:corr=fade.value(now)
  da=np.zeros(29,dtype=np.float32);da[arm]=corr/sc[arm];a=np.array(c['base_action'],dtype=np.float32)+da;q=off+np.clip(a*5,-contract['clip'],contract['clip'])*sc/5
  targets.append(dict(time=now,action=a.tolist(),q_PD=q.tolist(),correction=corr.tolist()))
 return targets,point_error

def main():
 m=mujoco.MjModel.from_xml_path(str(ROOT/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'));results=[]
 for run in json.loads((RUN/'compute-ledger.json').read_text())['attempts']:
  if run['stage'] not in ['teachers','repair_teacher','visits']:continue
  out=Path(run['path'])/'outputs';contract=json.loads((out/'action-contract.json').read_text())
  for f in sorted(out.glob('*__*.json')):
   ep=json.loads(f.read_text())
   if not isinstance(ep,dict) or not ep.get('control'):continue
   if run['stage']=='visits':
    if ep['controller'] not in ['alignment','distill']:continue
    ep['controller']='teacher_goal'
    for c in ep['control']:
     for field in ['action','q_PD','base_action','base_q_PD']:c[field]=c['oracle_'+field]
     c['teacher_correction']=c['oracle_correction']
   targets,pe=replay_query(ep,m,contract);a=float(abs(np.array([c['action'] for c in ep['control']])-np.array([c['action'] for c in targets])).max());q=float(abs(np.array([c['q_PD'] for c in ep['control']])-np.array([c['q_PD'] for c in targets])).max());corr=float(abs(np.array([c['teacher_correction'] for c in ep['control']])-np.array([c['correction'] for c in targets])).max());assert pe<1e-10 and a<1e-6 and q<1e-6 and corr<1e-10,(pe,a,q,corr)
   results.append(dict(file=str(f),point_parity_m=pe,action_parity=a,q_PD_parity_rad=q,correction_parity_rad=corr,release_included=True))
 (RUN/'analysis/query-parity.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
