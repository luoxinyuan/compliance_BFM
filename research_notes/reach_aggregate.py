"""Aggregate actual per-contact force/error, stage motion and intervention parity."""
import json,gzip,hashlib,time,sys
from pathlib import Path
import numpy as np
R=Path(sys.argv[1]);start=time.monotonic();results=[]
def read(p):return json.load(gzip.open(p,'rt')) if str(p).endswith('.gz') else json.loads(p.read_text())
def rms(a):return float(np.sqrt(np.mean(np.asarray(a)**2)))
for O in [R/'physical-01',R/'physical-02']:
 if not (O/'analysis.json').exists():continue
 analysis={r['label']:r for r in read(O/'analysis.json')}
 for p in sorted(O.glob('*.json.gz')):
  if p.name.endswith('-source-prefix.json.gz') or p.stem=='timelines.json':continue
  e=read(p)
  if not isinstance(e,dict) or 'physics' not in e:continue
  rs=e['physics'];s=e['condition'];label=e['label'];r=analysis[label];on=s['onset'];release=s['release'];a=[v for v in rs if v['active']];initial=min(rs,key=lambda v:abs(v['time']-on));finalactive=a[-1];f=np.array([v['contacts'][0]['force'] for v in a]);err=np.array([np.array(v['contacts'][0]['anchor'])-v['contacts'][0]['point'] for v in a]);qpd=np.array([v['q_PD'] for v in e['control']]);q=np.array([v['qpos'][7:] for v in rs]);root0=np.array(initial['root']);root=np.array([v['root'] for v in rs]);feet0=np.array(initial['feet_xyz']);firstfoot=next((v['time'] for v in a if np.max(np.linalg.norm((np.array(v['feet_xyz'])-feet0)[:,:2],axis=1))>.02),None)
  phases={}
  for name,lo,hi in [('first4',on,min(on+4,release)),('move',on,on+s['ramp']),('hold',on+s['ramp'],release),('recover',release,s['end'])]:
   rr=[v for v in rs if lo<=v['force_time']<hi]
   if not rr:continue
   ff=np.array([v['contacts'][0]['force'] for v in rr]);ee=np.array([np.array(v['contacts'][0]['anchor'])-v['contacts'][0]['point'] for v in rr]);phases[name]=dict(n=len(rr),force_rms=rms(np.linalg.norm(ff,axis=1)),force_xyz_rms=np.sqrt(np.mean(ff*ff,axis=0)).tolist(),error_cm=None if name=='recover' else 100*rms(np.linalg.norm(ee,axis=1)),signed_error_xyz_cm=(100*ee.mean(0)).tolist() if name!='recover' else None,root_delta_cm=(100*(np.array(rr[-1]['root'])-rr[0]['root'])).tolist(),root_peak_speed=max(np.linalg.norm(v['qvel'][:3]) for v in rr),max_tilt=max(v['tilt_deg'] for v in rr),slip_peak=max(v['contact_tangent_speed_max_mps'] for v in rr))
  item=dict(label=label,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),condition=s,error_cm=100*rms(np.linalg.norm(err,axis=1)),force_rms=rms(np.linalg.norm(f,axis=1)),force_xyz_rms=np.sqrt(np.mean(f*f,axis=0)).tolist(),cap_fraction=float(np.mean(np.linalg.norm(f,axis=1)>=4.999)),signed_error_xyz_cm=(100*err.mean(0)).tolist(),root_motion_end_cm=r['root_at_motion_end_cm'],root_contact_end_cm=(100*(np.array(finalactive['root'])-root0)).tolist(),root_active_max_xy_cm=100*max(np.linalg.norm(np.array(v['root'])[:2]-root0[:2]) for v in a),wrist_contact_delta_cm=(100*(np.array(finalactive['wrist'])-initial['wrist'])).tolist(),foot_xy_2cm_time=firstfoot,events=r['events'],phase=phases,boundary=e.get('boundary'),qpd_max_step_deg=float(np.degrees(abs(np.diff(qpd,axis=0))).max()),actual_q_max_step_deg=float(np.degrees(abs(np.diff(q,axis=0))).max()),qpd_p99_step_deg=float(np.percentile(np.degrees(abs(np.diff(qpd,axis=0))),99)),samples=len(rs),control_steps=len(e['control']),source='actual per-tick per-contact rows; release excluded from tracking')
  results.append(item)
parity=[]
for who in ['I','T30']:
 ref=read(R/'physical-01'/f'{who}_duration_80.json.gz')
 for folder,label in [('physical-01',who+'_noforce_geometry_80'),('physical-02',who+'_mask_error_velocity_80')]:
  p=R/folder/(label+'.json.gz')
  if not p.exists():continue
  e=read(p);pairs=list(zip(ref['physics'],e['physics']));pre=[(a,b) for a,b in pairs if a['force_time']<ref['condition']['onset']];parity.append(dict(label=label,pre_qpos_max=float(max(np.max(abs(np.array(a['qpos'])-b['qpos'])) for a,b in pre)),pre_qvel_max=float(max(np.max(abs(np.array(a['qvel'])-b['qvel'])) for a,b in pre)),anchor_max=float(max(np.max(abs(np.array(a['anchor'])-b['anchor'])) for a,b in pairs)),lengths=[len(ref['physics']),len(e['physics'])]))
(R/'final-metrics.json').write_text(json.dumps(dict(episodes=results,parity=parity,analysis_wall=time.monotonic()-start),indent=2))
for r in results:print(r['label'],round(r['error_cm'],3),round(r['force_rms'],3),'root_end',round(r['root_contact_end_cm'][0],2),'foot',r['foot_xy_2cm_time'],'boundary',r['boundary'])
print('PARITY',parity)
