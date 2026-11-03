"""Saved real-physics analysis; no policy loading, simulation, or model selection."""
from pathlib import Path
import json,gzip
import numpy as np
import mujoco
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM'); R=P/'bridge-v2-owned/runs/t30-transfer-20261103'; out=R/'analysis';out.mkdir(exist_ok=True)
contract=json.loads((R/'A-prepare-01/job/action-contract.json').read_text()); names=contract['dof_names'];m=mujoco.MjModel.from_xml_path(str(P/'bridge-v2-owned/trees/t30-transfer/repo/humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'));ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]; qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids]
def read(f):
 with gzip.open(f,'rt') as g:return json.load(g)
def quality(e):
 c=e['control'];pp=e['physics']; t=np.array([x['time'] for x in c]);tar=np.array([x['q_PD'] for x in c]);q=np.array([x['pre_qpos'] for x in c])[:,qa];v=np.array([x['pre_qvel'] for x in c])[:,va];jump=np.rad2deg(np.diff(tar,axis=0));act=np.rad2deg(np.diff(q,axis=0));i,j=np.unravel_index(np.abs(jump).argmax(),jump.shape);i+=1
 event=dict(time=float(t[i]),joint=names[j],target_jump_deg=float(jump[i-1,j]),preceding_actual_deg=float(act[i-1,j]),following_actual_deg=float(act[i,j]) if i<len(act) else None,actual_velocity_rad_s=float(v[i,j]),phase=next((n for n,(a,b) in e['phases'].items() if a<=t[i]<b),'other'))
 joint=[dict(name=n,target_max_deg=float(np.abs(jump[:,j]).max()),target_p99_deg=float(np.quantile(np.abs(jump[:,j]),.99)),actual_max_deg=float(np.abs(act[:,j]).max()),target_actual_gap_max_deg=float(np.rad2deg(np.abs(tar[:,j]-q[:,j])).max())) for j,n in enumerate(names)]
 phases={}
 for n,(a,b) in e['phases'].items():
  p=[r for r in pp if a<=r['time']<b];idx=np.where((t>=a)&(t<b))[0];idx=idx[idx>0]
  if len(p)<3:continue
  tt=np.array([x['time'] for x in p]);root=np.array([x['root'] for x in p]);w=np.array([x['wrist'] for x in p]);f=np.array([x['force'] for x in p]);err=np.array([x['anchor'] for x in p])-w;dt=np.diff(tt);rv=np.diff(root,axis=0)/dt[:,None];wv=np.diff(w,axis=0)/dt[:,None];ra=np.diff(rv,axis=0)/dt[1:,None];wa=np.diff(wv,axis=0)/dt[1:,None];support=np.array([x['support'] for x in p],bool)
  phases[n]=dict(samples=len(p),error_cm=float(100*np.sqrt(np.mean(np.sum(err*err,axis=1)))),signed_anchor_minus_wrist_cm=(100*err.mean(0)).tolist(),force_rms=float(np.sqrt(np.mean(np.sum(f*f,axis=1)))),force_xyz_rms=np.sqrt(np.mean(f*f,axis=0)).tolist(),force_xyz_mean=f.mean(0).tolist(),cap_fraction=float(np.mean(np.linalg.norm(f,axis=1)>=4.999)),root_delta=(root[-1]-root[0]).tolist(),wrist_delta=(w[-1]-w[0]).tolist(),root_path=float(np.linalg.norm(np.diff(root,axis=0),axis=1).sum()),root_speed_rms=float(np.sqrt(np.mean(np.sum(rv*rv,axis=1)))),wrist_speed_rms=float(np.sqrt(np.mean(np.sum(wv*wv,axis=1)))),root_acc_p99=float(np.quantile(np.linalg.norm(ra,axis=1),.99)),wrist_acc_p99=float(np.quantile(np.linalg.norm(wa,axis=1),.99)),torque_abs_peak=float(np.abs([x['torque'] for x in p]).max()),torque_sat_peak=max(x['torque_saturation'] for x in p),slip_peak=max(x['contact_tangent_speed_max_mps'] for x in p),no_support_fraction=float(np.mean(~support.any(1))),support_boolean_transitions=int(np.count_nonzero(np.diff(support.astype(int),axis=0))),tilt_max=max(x['tilt_deg'] for x in p),qpd_jump_max_deg=float(np.abs(jump[idx-1]).max()) if len(idx) else 0,actual_step_max_deg=float(np.abs(act[idx-1]).max()) if len(idx) else 0)
 return dict(label=e['label'],boundary=e['boundary'],largest_event=event,joints=joint,phases=phases)
rows=[]
for job in [R/'A-prepare-01/job',R/'C-final-02/job',R/'D-final-03/job']:
 if not (job/'summaries.json').exists():continue
 for row in json.loads((job/'summaries.json').read_text()):
  label=row['label'];f=job/(label+'.json.gz')
  if 'take_' in label:continue
  target=out/(label+'-quality.json')
  if not target.exists():target.write_text(json.dumps(quality(read(f)),indent=2))
  q=json.loads(target.read_text());rows.append(dict(**row,quality_path=str(target),largest_event=q['largest_event'],release=q['phases'].get('recover'),path=str(f)))
(out/'comparison.json').write_text(json.dumps(rows,indent=2)); rec=[]
for f in sorted((R/'A-prepare-01/job').glob('*-recovery.json')):rec.append(dict(path=str(f),**json.loads(f.read_text())))
(out/'recovery-index.json').write_text(json.dumps(rec,indent=2));print('analysed',len(rows),'conditions;',sum(len(x['records']) for x in rec),'takeovers')
# First sustained state separation is descriptive, not causal attribution. Threshold is reported, not a pass gate.
for condition in ['normal','early_stop','late_stop','sealed_early','sealed_late']:
 job=R/'A-prepare-01/job';tf=job/('source_'+condition+'.json.gz')
 if not tf.exists():tf=job/('Teacher30_'+condition+'.json.gz')
 if not tf.exists():continue
 te=read(tf);tp=te['physics'];tt=np.array([x['time'] for x in tp]);tw=np.array([x['wrist'] for x in tp]);tr=np.array([x['root'] for x in tp]);div=[]
 for lab in ['S0','S1warm','S1matched','S2']:
  sf=job/(lab+'_'+condition+'.json.gz')
  if not sf.exists():continue
  se=read(sf);sp=se['physics'];st=np.array([x['time'] for x in sp]); n=min(len(st),len(tt));assert np.max(abs(st[:n]-tt[:n]))<1e-8
  wd=np.linalg.norm(np.array([x['wrist'] for x in sp[:n]])-tw[:n],axis=1);rd=np.linalg.norm(np.array([x['root'] for x in sp[:n]])-tr[:n],axis=1);first=None
  for i in range(n):
   j=np.searchsorted(st[:n],st[i]+.2)
   if j<n and np.all(wd[i:j+1]>.02):first=float(st[i]);break
  div.append(dict(student=lab,first_wrist_separation_over_2cm_for_200ms=first,max_wrist_separation=float(wd.max()),max_root_separation=float(rd.max()),interpretation='comparison with Teacher own rollout; not proof of failure or information insufficiency'))
 (out/(condition+'-separation.json')).write_text(json.dumps(div,indent=2))
