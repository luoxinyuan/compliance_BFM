"""Saved physics only; no model forward, simulation, fitting or selection."""
import gzip,json
from pathlib import Path
import numpy as np
from m0_analysis import summarize
from m2_analysis import window
from ccl02a_teacher import audit_episode
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs');RUN=R/'m4-continuous-feedback-20261031A'
def read(p):return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
def active_window(e,lo,hi):
 rr=[r for r in e['physics'] if lo-1e-8<=r['force_time']<hi-1e-8 and r['active']];err=np.array([np.array(r['anchor'])-r['wrist'] for r in rr]);f=np.array([r['force'] for r in rr]);return dict(error_cm=float(np.sqrt(np.mean(np.sum(err**2,axis=1)))*100),xyz_cm=(np.sqrt(np.mean(err**2,axis=0))*100).tolist(),force_rms_n=float(np.sqrt(np.mean(np.sum(f*f,axis=1)))),force_mean_xyz_n=f.mean(0).tolist(),cap_fraction=float((np.linalg.norm(f,axis=1)>=4.999999).mean()),wrist_end=rr[-1]['wrist'])
def main():
 dest=RUN/'analysis';dest.mkdir(exist_ok=True);res={};raw={};parity={};initial=read(R/'issue21-ccl02a-20261029T153900Z/attempt-01/outputs/dev-2-initial-state.json')
 old=read(R/'m3-continuous-body-20261031A/final-attempt-01/outputs/u500__m3_student.json.gz')
 native=read(R/'m3-continuous-body-20261031A/qualification-attempt-01/outputs/u500__native_hold.json.gz')
 for a in read(RUN/'compute-ledger.json')['attempts']:
  if a.get('result_status')!='completed' or a['stage']=='audit':continue
  src=Path(a['path'])/'outputs';contract=read(src/'action-contract.json')
  for p in src.glob('*.json.gz'):
   e=read(p);raw[e['label']]=e;x=summarize(e,{});x['source']=str(p);x['audit']=audit_episode(e,contract,initial);rel=e['condition']['release'];on=e['condition']['onset'];end=e['condition']['end'];cs=e['control'];rows=e['physics'];ct=np.array([c['time'] for c in cs]);ts=np.array([r['time'] for r in rows]);q=np.array([c['q_PD'] for c in cs]);dq=abs(np.diff(q,axis=0));dq[ct[1:]<10.4-1e-8]=-1;i,j=np.unravel_index(dq.argmax(),dq.shape)
   x['command_peak']=dict(time=cs[i+1]['time'],joint=contract['dof_names'][j],qPD_step_deg=float(np.degrees(dq[i,j])),actual_20ms_deg=float(np.degrees(cs[i+1]['pre_qpos'][7+j]-cs[i]['pre_qpos'][7+j])),z_step_deg=cs[i+1]['z_step_deg'])
   x['recovery']=window(e,rel,end);x['recovery']['force_peak_n']=max(np.linalg.norm(r['force']) for r in rows if r['force_time']>=rel-1e-8)
   x['late_recovery']=window(e,rel+6.4,end);x['precontact']=window(e,10.4,on)
   x['recovery_speed_2_4_6_8']=[float(np.linalg.norm(rows[int(np.argmin(abs(ts-rel-t)))]['qvel'][:3])) for t in [2,4,6,8]]
   z=np.array([c['executed_z'][0] for c in cs]);x['z_norm_error']=float(abs(np.linalg.norm(z,axis=1)-16).max());assert x['z_norm_error']<1e-4
   x['events']=[dict(time=c['time'],**c['rule_record']) for c in cs if c.get('rule_record') and c['rule_record'].get('event')]
   pp=[c['projection'] for c in cs if c.get('projection')];x['projection']=None if not pp else dict(frames=len(pp),arm_rms_deg=float(np.sqrt(np.mean([p['arm_rms_deg']**2 for p in pp]))),arm_max_deg=max(p['arm_rms_deg'] for p in pp),nonarm_rms_deg=float(np.sqrt(np.mean([p['nonarm_rms_deg']**2 for p in pp]))),nonarm_max_deg=max(p['nonarm_rms_deg'] for p in pp))
   if '__at' in e['label']:
    t=float(e['label'].split('__at')[1]);x['remaining_contact']={k:active_window(v,t,rel) for k,v in [('branch',e),('frozen_M3',old),('native_reference',native)]};x['after_switch_1s']=active_window(e,t,min(t+1,rel))
    ss=read(src/(e['label']+'-switch-state.json'));x['switch_parity']=ss['parity'];x['switch_gate']=ss['gate'];ii=int(np.argmin(abs(ct-t)));x['switch_action_jump_deg']=float(np.degrees(max(abs(np.array(cs[ii]['q_PD'])-cs[ii-1]['q_PD']))))
   res[e['label']]=x
 for ctl in ['native_hold','teacher','rule','m3_student']:
  key='u250__'+ctl;free=key+'__free'
  if key not in raw or free not in raw:continue
  e,f=raw[key],raw[free];on=e['condition']['onset'];pre=np.array([r['force_time'] for r in e['physics']])<on-1e-8
  parity[key]={k:float(abs(np.array([r[k] for r in e['physics']])[pre]-np.array([r[k] for r in f['physics']])[pre]).max()) for k in ['qpos','qvel','wrist','force']};assert max(parity[key].values())==0
  ix=min(range(len(e['physics'])),key=lambda i:abs(e['physics'][i]['time']-e['condition']['release']));res[key]['contact_minus_free_end_cm']={k:((np.array(e['physics'][ix][k])-f['physics'][ix][k])*100).tolist() for k in ['wrist','shoulder_xyz','root']}
 result=dict(episodes=res,precontact_parity=parity,ledger=read(RUN/'compute-ledger.json'),scope='Realcontact XYZ errors only. Recovery compares fullcurves. Singlefixedseed notstatisticalproof; no outcome-selected training.')
 (dest/'physical-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 for k,x in res.items():
  q=x['phases'].get('all_contact',{});print(k,'3D/hold',q.get('tracking_rms_cm'),x['phases'].get('hold',{}).get('tracking_rms_cm'),'F',q.get('force_rms_n'),'cap',q.get('force_cap_fraction'),'recovery',x['recovery_speed_2_4_6_8'],'qPD',x['command_peak']);
  if x.get('remaining_contact'):print('REMAINING',json.dumps(x['remaining_contact']),'PROJECTION',x['projection'])
if __name__=='__main__':main()
