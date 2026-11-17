"""20ms per-joint and physical-stage continuity from fixed actual archives only."""
import sys,os,time,json,hashlib,traceback,collections,math,xml.etree.ElementTree as ET
from pathlib import Path
START=time.monotonic();S=Path(__file__).resolve().parents[1];R=S/'eval-runtime'
sys.path[:0]=[str(R),str(R/'research_notes')]
import torch,numpy as np
torch.set_num_threads(1);torch.set_grad_enabled(False)
from research.interaction_fb.journal import read
OUT=S/'runs/final-stage-quality-03';OUT.mkdir(exist_ok=False)
def jw(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def phase(sp,t,active):
 if not active:return 'precontact' if t<sp['onset'] else 'released'
 stop=sp.get('stop_time',sp['onset']+sp['ramp'])
 return 'moving' if t<stop-1e-8 else ('decelerating' if t<stop+sp.get('deceleration',0)-1e-8 else 'holding')
def kin(ph,cc):
 if not ph or len(cc)<2:return None
 qpd=np.array([c['q_PD']for c in cc]);q=np.array([p['qpos'][7:]for p in ph]);dq=np.array([p['qvel'][6:]for p in ph]);torque=np.array([p['torque']for p in ph]);jump=abs(np.diff(qpd,axis=0))
 root=np.array([p['root']for p in ph]);w=np.array([p['wrist']for p in ph]);rv=np.diff(root,axis=0)/.005;wv=np.diff(w,axis=0)/.005
 rr={'qPD_jump_p95_rad_per_joint':np.quantile(jump,.95,axis=0).tolist(),'qPD_jump_max_rad_per_joint':jump.max(0).tolist(),'qPD_jump_max_rad':float(jump.max()),'actual_dq_p95_rad_s_per_joint':np.quantile(abs(dq),.95,axis=0).tolist(),'actual_dq_max_rad_s_per_joint':abs(dq).max(0).tolist(),'actual_ddq_p95_rad_s2_per_joint':np.quantile(abs(np.diff(dq,axis=0)/.005),.95,axis=0).tolist(),'actual_q_min_rad_per_joint':q.min(0).tolist(),'actual_q_max_rad_per_joint':q.max(0).tolist(),'actual_q_5ms_jump_max_rad':float(abs(np.diff(q,axis=0)).max()),'torque_p95_Nm_per_joint':np.quantile(abs(torque),.95,axis=0).tolist(),'torque_peak_Nm_per_joint':abs(torque).max(0).tolist(),'torque_saturation_max':max(p['torque_saturation']for p in ph),'root_speed_p95_m_s':float(np.quantile(np.linalg.norm(rv,axis=1),.95)),'wrist_speed_p95_m_s':float(np.quantile(np.linalg.norm(wv,axis=1),.95)),'root_accel_p95_m_s2':float(np.quantile(np.linalg.norm(np.diff(rv,axis=0)/.005,axis=1),.95)),'wrist_accel_p95_m_s2':float(np.quantile(np.linalg.norm(np.diff(wv,axis=0)/.005,axis=1),.95))}
 # q_PD is a target. q/dq are independent measured physics values.
 times=np.array([c['time']for c in cc]);pdtimes=np.array([p['force_time']for p in ph]);ix=np.clip(np.searchsorted(times,pdtimes,side='right')-1,0,len(cc)-1)
 rr['PD_target_actual_error_p95_rad_per_joint']=np.quantile(abs(qpd[ix]-q),.95,axis=0).tolist()
 return rr
result={};records=[];probe_records=[]
try:
 D=S/'runs/formal-dev-02';rows=json.loads((D/'evaluation.json').read_text())
 xml=S/'runtime/humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'
 import mujoco
 model=mujoco.MjModel.from_xml_path(str(xml));ids=[j for j in range(model.njnt)if model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE];names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j)for j in ids]
 assert len(names)==29 and [int(model.jnt_qposadr[j])for j in ids]==list(range(7,36)),names
 for r in rows:
  ci=r['case_index'];a=read(D/f'case-{ci:03d}-actual.ptz',r['archive']);ph=a['online_physics'];cc=a['controls'];ins=a['instructions'];times=np.array([i['time']for i in ins]);sp=a['spec']
  pkeys=[]
  for p in ph:
   idx=int(np.clip(np.searchsorted(times,p['force_time']+1e-8,side='right')-1,0,len(ins)-1));pkeys.append(phase(sp,p['force_time'],p['active'])+(':'+('FOLLOW'if ins[idx]['goal_mode']==0 else'HOLD')))
  ft=np.array([p['force_time']for p in ph]);ckeys=[pkeys[int(np.clip(np.searchsorted(ft,c['time']+1e-8,side='right')-1,0,len(ph)-1))]for c in cc]
  for key in dict.fromkeys(pkeys):
   pp=[p for p,k in zip(ph,pkeys)if k==key];ct=[c for c,k in zip(cc,ckeys)if k==key];metrics=kin(pp,ct)
   records.append(dict(case=ci,method=r['method'],parent=r['parent_id'],role=r['role'],family=r['family'],window=key,physics_ticks=len(pp),control_rows=len(ct),metrics=metrics,loaded_slip_p50_p90_p95_max=np.quantile([p['stage_loaded_slip_mps']for p in pp],[.5,.9,.95,1]).tolist(),loaded_slip_overlimit_s=.005*sum(p['stage_loaded_slip_mps']>.45 for p in pp)))
  # Pre-registered label support first. All eligible 20ms starts, frozen I primary actual trajectories.
  if r['method']=='I-final'and r['role']=='primary':
   for h,n in ((.5,100),(1.,200)):
    for row in a['rows']:
     b=row['physics_begin']
     if b+n>len(ph)or float(torch.linalg.vector_norm(row['observation']['contact_privileged_v1'][0,6:9])*5)<1:continue
     ss=ph[b:b+n];yaw=np.unwrap([math.atan2(p['pelvis_matrix'][1][0],p['pelvis_matrix'][0][0])for p in ss]);fr=math.sqrt(np.mean([np.dot(p['force'],p['force'])for p in ss]));good=fr<=4 and max(p['stage_loaded_slip_mps']for p in ss)<=.45 and np.degrees(abs(yaw-yaw[0])).max()<=20 and max(p['joint_limit_excess_max_rad']for p in ss)<=.02 and all(any(p['support'])for p in ss)
     probe_records.append(dict(case=ci,parent=r['parent_id'],ancestor=sp['execution_source_parent_id'],horizon=h,row_index=row['physics_begin']//4,label=int(good),start_force_N=float(torch.linalg.vector_norm(row['observation']['contact_privileged_v1'][0,6:9])*5)))
  jw('progress.json',dict(completed_cases=ci+1));print('STAGE',ci,flush=True)
 by={(r['method'],r['parent'],r['role'],r['family'],r['window']):r for r in records};bad=[]
 for r in records:
  if r['method']not in('E-final','I-final')or r['metrics']is None:continue
  b=by.get(('Goal-only',r['parent'],r['role'],r['family'],r['window']))
  if b is None or b['metrics']is None:continue
  for field,limit in(('qPD_jump_p95_rad_per_joint',.03),('actual_dq_p95_rad_s_per_joint',.5)):
   x=np.array(r['metrics'][field]);y=np.array(b['metrics'][field]);ix=np.where((x>1.25*y)&(x-y>limit))[0]
   if len(ix):bad.append(dict(case=r['case'],window=r['window'],field=field,joints=[names[j]for j in ix],new=x[ix].tolist(),baseline=y[ix].tolist()))
 support={}
 for h in(.5,1.):
  rr=[x for x in probe_records if x['horizon']==h];pos={x['ancestor']for x in rr if x['label']};neg={x['ancestor']for x in rr if not x['label']};anc={x['ancestor']for x in rr}
  support[str(h)]=dict(rows=len(rr),positive_rows=sum(x['label']for x in rr),negative_rows=sum(not x['label']for x in rr),positive_ancestors=len(pos),negative_ancestors=len(neg),ancestors=len(anc),status='ELIGIBLE_FOR_GROUPED_PROBE'if len(pos)>=4 and len(neg)>=4 and len(anc)>=8 else'UNDERPOWERED',start_measurement='actual actor observation contact[6:9]*5N; no future physics eligibility')
 jw('probe-label-records.json',probe_records);jw('probe-label-support.json',support)
 jw('stage-continuity.json',records);jw('joint-order.json',dict(joints=names,source_xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest()));jw('continuity-degradations.json',bad)
 result=dict(status='STAGE_QUALITY_COMPLETE',cases=100,windows=len(records),degradations=bad,physical_steps=0,optimizer_updates=0,ledger_merge_pending=True)
except BaseException as ex:result=dict(status='STAGE_QUALITY_FAILED',error=repr(ex),traceback=traceback.format_exc(),completed_windows=len(records),physical_steps=0,optimizer_updates=0,ledger_merge_pending=True)
finally:
 result['wall_s']=time.monotonic()-START;result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();jw('result.json',result);print(json.dumps(result),flush=True)
if result['status']!='STAGE_QUALITY_COMPLETE':raise RuntimeError(result['error'])
