"""Actual global-z/action audit. Context swaps are sensitivity, not label-conflict proof."""
import gzip,json,hashlib
from pathlib import Path
import numpy as np,torch
from m1_interface import endpoints,base_latent
from m1_learning import load_net,completed
from m2_rule import infer_coordinate
from m3_learning import support
from contact_gate import smooth
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs')
M3=R/'m3-continuous-body-20261031A'
def read(p):return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
def angle(a,b):
 a=np.asarray(a).reshape(-1);b=np.asarray(b).reshape(-1)
 return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))
def run(env,model,contract,out):
 dev,qof=support(env,model,contract);ends=endpoints(env);ep=read(M3/'final-attempt-01/outputs/u500__m3_student.json.gz');cs=ep['control'];refs=read(M3/'final-attempt-01/outputs/references.json')['proprio_refs'];rec=[]
 for t in [12.4,13.2,13.6,14.4,16.,19.4,24.]:
  c=min(cs,key=lambda c:abs(c['time']-t));obs={k:torch.tensor(c['bfm_obs'][k],device=dev) for k in env.keys};contexts=[]
  for u in [.25,.5,.75]:
   for mode in ['goal','pose']:
    a=smooth((t-12.4)/4) if mode=='goal' else 0.;r=smooth((t-17.4)/.75) if t>=17.4 else 0.;z=base_latent(ends,u,a,r);q=qof(model.act(obs,z,mean=True));contexts.append(dict(u=u,mode=mode,z=z[0].tolist(),q=q[0].tolist()))
  original=torch.tensor(c['oracle_z'],device=dev);oq=qof(model.act(obs,original,mean=True));pose=next(x for x in contexts if x['u']==.5 and x['mode']=='pose');goal=next(x for x in contexts if x['u']==.5 and x['mode']=='goal')
  rec.append(dict(time=c['time'],force=c['measured_force_xyz'],current_proprio_context=infer_coordinate(c['pre_qpos'][29:36],refs),context_goal_pose_z_angle_deg=angle(pose['z'],goal['z']),context_goal_pose_arm_q_rms_deg=float(np.degrees(np.sqrt(np.mean((np.asarray(pose['q'][22:])-goal['q'][22:])**2)))),original_base_q_max_error=float(abs(oq[0]-torch.tensor(c['oracle_base_q_PD'],device=dev)).max()),oracle_correction=c['oracle_correction'],contexts=contexts))
 ready=min(cs,key=lambda c:abs(c['time']-10.4));ready_inference=infer_coordinate(ready['pre_qpos'][29:36],refs)
 # Actual dataset labels, using global projected z and q_PD, never raw redundant u/a/r.
 d=torch.load(M3/'train-attempt-01/outputs/dataset.pt',map_location=dev,weights_only=True);z=torch.load(M3/'train-attempt-01/outputs/projected-global-z.pt',map_location=dev,weights_only=True);net=load_net(M3/'train-attempt-01/outputs/projected-z-final.pt',dev);h=(d['h']-net.center)/net.scale;label_pairs=[]
 for g in d['group'].unique():
  ix=(d['group']==g).nonzero().flatten();other=(d['group']!=g).nonzero().flatten();best=[]
  for start in range(0,len(ix),64):
   ii=ix[start:start+64];dist=torch.cdist(h[ii],h[other])/h.shape[1]**.5;vals,cols=dist.min(1)
   best.extend((float(v),int(i),int(other[j])) for v,i,j in zip(vals,ii,cols))
  for distance,i,j in sorted(best)[:3]:label_pairs.append(dict(feature_rms=distance,i=i,j=j,source_i=d['metadata'][i],source_j=d['metadata'][j],global_z_angle_deg=angle(z[i].cpu(),z[j].cpu()),q_PD_arm_rms_deg=float(torch.rad2deg((d['q'][i,22:]-d['q'][j,22:]).square().mean().sqrt())),q_PD_nonarm_rms_deg=float(torch.rad2deg((d['q'][i,:22]-d['q'][j,:22]).square().mean().sqrt()))))
 result=dict(status='completed',student_optimizer_updates=0,ready_context_from_deployable_proprio=ready_inference,same_observation_context_sensitivity=rec,nearest_cross_source_actual_labels=sorted(label_pairs,key=lambda x:x['feature_rms']),teacher_structure=dict(global_command='Initial pose/goal u plus preset time; M3 selects pose-base only for u500 Student labels, goal-base elsewhere.',feedback='Original frozen BFM receives actual state/history at50Hz. Bounded7jointJacobian feedback receives actual anchor-wrist/Jacobian at10Hz, .2s release at50Hz.',rule='Deployable initialproprio inferred once at ready, remembered. Original force-history gate changes progress; does not recompute reachability/wholebody endpoint from ongoing state.',memory='Initial intended pose can be retained from ready proprio/previous executed command. Default postcontact return is shared standing. Current MLP has6.4s force history but only native observation/action history; no persistent intended-pose register. Rawu/a/r are not labels and their nonuniqueness is not conflict.',limits='Actual near-input differences and counterfactual context swaps do not alone prove nonidentifiability. No teacher/Student/memory changes. Physical recovery separately required.'),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [M3/'train-attempt-01/outputs/dataset.pt',M3/'train-attempt-01/outputs/projected-global-z.pt']})
 (out/'label-memory-audit.json').write_text(json.dumps(result,indent=2));print('M4_AUDIT',json.dumps({k:result[k] for k in ['ready_context_from_deployable_proprio','teacher_structure']}),flush=True);return result
