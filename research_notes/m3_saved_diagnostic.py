"""Post-final diagnostic only: fixed saved Student states, temporary z; never refit Student."""
import gzip,json
from pathlib import Path
import numpy as np,torch
from m3_learning import project,support,write
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m3-continuous-body-20261031A')
def run(env,model,contract,out):
 ep=json.loads(gzip.decompress((R/'final-attempt-01/outputs/u625__m3_student.json.gz').read_bytes()));cs=ep['control'];dev,qof=support(env,model,contract);records=[]
 for t in [13.4,15.15,17.95,18.35,24.75]:
  i=int(np.argmin(abs(np.array([c['time'] for c in cs])-t)));b=cs[i:i+5];assert all(c['oracle_query_valid'] for c in b)
  obs={k:torch.tensor([c['bfm_obs'][k][0] for c in b],device=dev) for k in env.keys};base=torch.tensor([c['oracle_z'][0] for c in b],device=dev);gate=torch.tensor([c['local_gate'] for c in b],device=dev);target=torch.tensor([c['oracle_q_PD'] for c in b],device=dev);current=torch.tensor([c['q_PD'] for c in b],device=dev);delta=target-torch.tensor([c['oracle_base_q_PD'] for c in b],device=dev)
  z,e=project(model,obs,base,gate,target,qof);ref=float(delta[:,22:].square().mean().sqrt());arm=float(e[:,22:].square().mean().sqrt());nonarm=float(e[:,:22].square().mean().sqrt())
  row=dict(requested_time=t,actual_times=[c['time'] for c in b],gate=gate.tolist(),student_coordinates=[c['native_coords'] for c in b],teacher_increment_arm_rms_rad=ref,student_vs_query_arm_rms_rad=float((current-target)[:,22:].square().mean().sqrt()),student_vs_query_nonarm_rms_rad=float((current-target)[:,:22].square().mean().sqrt()),temporary_projection_arm_rms_rad=arm,temporary_projection_nonarm_rms_rad=nonarm,passed=arm<=max(.5*ref,.001) and nonarm<=.005,norm_error=float(abs(z.norm(dim=-1)-16).max()));records.append(row);print('M3_SAVED_DIAGNOSTIC',row,flush=True)
 result=dict(blocks=records,student_optimizer_updates=0,temporary_z_steps=160,source='Frozen final Student visited states; qualified original goal-base teacher queried on those states.',scope='Post-final finite expression diagnosis, no deployment of optimized z, no test data training or checkpoint selection. Does not prove projected actions improve physics.')
 write(out,'saved-student-diagnostic.json',result);return result
