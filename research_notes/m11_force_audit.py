"""Reconstruct actual spring/damper force from saved pre-step physical states.
MuJoCo kinematics only, no rollout, policy, asset download, or force-cap change.
"""
import argparse,json,gzip
from pathlib import Path
import numpy as np,mujoco

def main():
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--xml',type=Path,required=True);a=p.parse_args()
 m=mujoco.MjModel.from_xml_path(str(a.xml));d=mujoco.MjData(m);body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_wrist_yaw_link');jac=np.zeros((3,m.nv));out=[]
 for f in sorted(a.folder.glob('*.json.gz')):
  e=json.loads(gzip.decompress(f.read_bytes()));previous=e['initial'];raws=[];springs=[];dampers=[];forces=[];errors=[];work=[];reconstruct=[];times=[]
  for row in e['physics']:
   if row['active']:
    d.qpos[:]=previous['qpos'];d.qvel[:]=previous['qvel'];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);point=d.xpos[body].copy();mujoco.mj_jac(m,d,jac,None,point,body);velocity=jac@d.qvel
    spring=80*(np.array(row['anchor'])-point);damper=8*(np.array(row['anchor_velocity'])-velocity);raw=spring+damper;expected=raw*min(1.,5/max(np.linalg.norm(raw),1e-12));force=np.array(row['force'])
    reconstruct.append(float(np.max(np.abs(expected-force))));raws.append(raw);springs.append(spring);dampers.append(damper);forces.append(force);errors.append(np.linalg.norm(spring)/80);work.append(np.dot(force,velocity)*.005);times.append(row['force_time'])
   previous=row
  if not raws:continue
  F=np.array(forces);raw=np.array(raws);S=np.array(springs);D=np.array(dampers);norm=np.linalg.norm(raw,axis=1);actualnorm=np.linalg.norm(F,axis=1);err=np.array(errors);sat=actualnorm>=4.999
  rms=lambda x:np.sqrt(np.mean(np.square(x),axis=0)).tolist()
  row=dict(label=e['label'],exact_reconstruction_max_abs_error_N=max(reconstruct),active_seconds=len(F)/200,uncapped_norm_rms_N=float(np.sqrt(np.mean(norm**2))),uncapped_norm_quantiles_N=np.percentile(norm,[0,25,50,75,95,100]).tolist(),actual_xyz_rms_N=rms(F),raw_xyz_rms_N=rms(raw),spring_xyz_rms_N=rms(S),damper_xyz_rms_N=rms(D),actual_force_xyz_mean_N=F.mean(0).tolist(),actual_force_rms_N=float(np.sqrt(np.mean(actualnorm**2))),saturation_fraction=float(sat.mean()),yz_exceeds_x_fraction=float((np.linalg.norm(F[:,1:],axis=1)>np.abs(F[:,0])).mean()),force_cost_active_mean=float(np.mean(1.5*actualnorm**2/25)),force_cost_saturated_std=float(np.std(1.5*actualnorm[sat]**2/25)) if sat.any() else None,tracking_cost_active_mean=float(np.mean(.15*np.minimum(err**2/.04,16))),raw_demand_cost_active_mean=float(np.mean(.15*np.minimum(np.log1p(norm**2/25),8))),net_external_work_J=float(sum(work)),physical_cap_N=5,static_spring_error_at_cap_cm=6.25)
  out.append(row)
 result=dict(method='Reconstructed before each actual physics step from preceding qpos/qvel; same XML, wrist link origin, world-frame K80/D8 and vector-norm cap5. No simulation or changed interaction.',robot_mass_kg=float(m.body_mass.sum()),cap_as_fraction_of_weight=float(5/(m.body_mass.sum()*9.81)),rows=out)
 (a.folder/'force-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
