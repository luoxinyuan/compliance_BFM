"""Freeze limits from archived B and actual G1 joint margins; CPU-only, no rollout."""
import hashlib,json
from pathlib import Path
import numpy as np
import mujoco

P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
SOURCE=P/'bridge-v2-owned/runs/issue21-goalcalibration-20261030A/attempt-01/outputs'
ROOT=Path(__file__).resolve().parents[1]

def prepare():
    ep=json.loads((SOURCE/'anchored.json').read_text())
    contract=json.loads((SOURCE/'action-contract.json').read_text())
    m=mujoco.MjModel.from_xml_path(str(ROOT/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'))
    q=np.array([r['qpos'][7:] for r in ep['physics']]);qp=np.array([r['q_PD'] for r in ep['control']])
    ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in contract['dof_names']]
    limits=m.jnt_range[ids];rows=[];caps=[]
    for i in contract['arm_indices']:
        margin=min(float(np.minimum(q[:,i]-limits[i,0],limits[i,1]-q[:,i]).min()),
                   float(np.minimum(qp[:,i]-limits[i,0],limits[i,1]-qp[:,i]).min()))
        cap=min(.20,.5*margin);caps.append(cap)
        rows.append(dict(joint=contract['dof_names'][i],limits_rad=limits[i].tolist(),
                         observed_actual_range_rad=[float(q[:,i].min()),float(q[:,i].max())],
                         observed_command_range_rad=[float(qp[:,i].min()),float(qp[:,i].max())],
                         min_actual_and_command_margin_rad=margin,total_cap_rad=cap,
                         observed_command_step_max_deg=float(np.degrees(abs(np.diff(qp[:,i])).max()))))
    pins={f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in ['anchored.json','anchored-initial-state.json','goal-path.json','geometry.json','action-contract.json']}
    return dict(base='a3f0aaad8d605ef0407ca2b2846c33e65b235951',
      question='Can bounded accumulated wrist-error correction remove persistent error on fixed B goal path?',
      method_change='Privileged teacher algorithm modification; not an engineering bug fix or latent-only policy',
      baseline_source=str(SOURCE),pins=pins,source_teacher_sha256=hashlib.sha256((ROOT/'research_notes/ccl02a_teacher.py').read_bytes()).hexdigest(),
      labels=['accumulated'],seed=0,onset_s=3.,ramp_s=4.,release_s=8.,horizon_s=9.,K=80,D=8,max_force_n=5,
      increment=dict(hz=10,per_joint_rad=.04,norm_rad=.08,damping=.02),
      accumulated=dict(per_joint_caps_rad=caps,norm_cap_rad=.40,per_joint_velocity_rad_s=.40,norm_velocity_rad_s=.80,hz=50,
        release_duration_s=.80,release_shape='1-smoothstep; snapshot last applied at first inactive 50Hz sample; then zero',
        anti_windup='Integrate from applied correction, project into total/joint limits, discard rejected increment; no hidden accumulator; loss of contact stops all integration',
        online_joint_margin_rad=.02),
      bound_rationale='Original10Hz .04/.08 steps imply .4/.8 rad/s; cubic.8s release permits <=.2133/.4267rad, rounded down to.20/.40. Per-joint caps additionally <=half observed actual/command clearance. No joint-specific20deg target.',
      margins=rows,goal_path='Exact archived B latent at every50Hz sample including unchanged original.2s latent release',
      success='Original hold3D RMS<1cm AND complete stable rollout; report ramp XYZ, force and release independently. Do not tune after observing this run.',
      physical_stop='Original failure or root<.65m or pelvis tilt>20deg or nonfootground or actual joint excess>.02rad; stop on nonfinite/invalid command or conflicting correction bounds',
      attribution='Log BFM goal q_PD, same-state standing q_PD and applied joint correction separately. RMS correction / RMS BFM-goal-minus-standing is descriptive command attribution, not motion causality. If ratio>=1 mark direct correction as dominant in that joint group.',
      branch_if_failed='Stop current manual goal/IK calibration line; do not raise caps or train',
      policy_updates=0,resources=dict(cpu=4,rss_gib=12,gpu_mib=4096,cumulative_wall_s=2400))

if __name__=='__main__':
    p=ROOT/'research_notes/accumulated_protocol.json'
    assert not p.exists(),'Do not overwrite sealed protocol'
    value=prepare();p.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
