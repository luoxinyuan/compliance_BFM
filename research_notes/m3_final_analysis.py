"""Read saved, frozen final M3 physics. No simulation, model, fitting or selection."""
import gzip,json
from pathlib import Path
import numpy as np
from m0_analysis import summarize
from m2_analysis import window
from ccl02a_teacher import audit_episode
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
R=P/'bridge-v2-owned/runs/m3-continuous-body-20261031A'
def read(p):
 p=Path(p);return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))
def vector(x):
 x=np.asarray(x);return dict(mean=x.mean(0).tolist(),end=x[-1].tolist(),rms=np.sqrt(np.mean(x*x,0)).tolist(),peak_norm=float(np.linalg.norm(x,axis=-1).max()))
def main():
 ledger=read(R/'compute-ledger.json');a=[a for a in ledger['attempts'] if a['stage']=='final'];assert len(a)==1 and a[0]['result_status']=='completed'
 src=Path(a[0]['path'])/'outputs';contract=read(src/'action-contract.json');initial=read(P/'bridge-v2-owned/runs/issue21-ccl02a-20261029T153900Z/attempt-01/outputs/dev-2-initial-state.json')
 raw={};out={};pairs={};comparisons={};ready={}
 extra=[a for a in ledger['attempts'] if a['stage']=='contact_ablation'];assert len(extra)==1 and extra[0]['result_status']=='completed'
 paths=sorted(src.glob('*.json.gz'))+sorted((Path(extra[0]['path'])/'outputs').glob('*.json.gz'))
 for p in paths:
  e=read(p);assert e['boundary'] is None;raw[e['label']]=e;q=summarize(e,{})
  rows=e['physics'];cs=e['control'];ts=np.array([r['time'] for r in rows]);ct=np.array([c['time'] for c in cs]);on=e['condition']['onset'];rel=e['condition']['release'];rd=e['control_ready_s'];end=e['condition']['end'];u=e['goal_candidate']['u']
  q.update(source=str(p),configuration_u=u,ready_sha=e['ready_physical_state_sha256'],deployment_audit=audit_episode(e,contract,initial),requested_end_s=end,actual_end_s=e['actual_end_s'])
  ready.setdefault(e['configuration'],set()).add(q['ready_sha'])
  q['windows']={k:window(e,lo,hi) for k,(lo,hi) in dict(ready=(rd-1,rd),precontact=(rd,on),handoff=(rd,rd+.75),post_handoff=(rd+.75,on),full_recovery=(rel,end),late_recovery=(rel+6.4,end),no_contact_deployed=(rd,end)).items()}
  target=np.asarray([c['q_PD'] for c in cs]);actual=np.asarray([c['pre_qpos'][7:] for c in cs]);dq=np.abs(np.diff(target,axis=0));dq[ct[1:]<rd-1e-8]=-1;i,j=np.unravel_index(dq.argmax(),dq.shape)
  q['command_peak']=dict(time=cs[i+1]['time'],joint=contract['dof_names'][j],target_step_deg=float(np.degrees(dq[i,j])),actual_20ms_motion_deg=float(np.degrees(actual[i+1,j]-actual[i,j])),latent_step_deg=cs[i+1]['z_step_deg'])
  z=np.asarray([c['executed_z'][0] for c in cs]);q['latent_norm_max_error']=float(abs(np.linalg.norm(z,axis=-1)-16).max());assert q['latent_norm_max_error']<1e-4
  q['body_windows']={}
  for name,lo,hi in [('contact',on,rel),('release',rel,end),('precontact',rd,on)]:
   mask=(ts>=lo-1e-8)&(ts<hi-1e-8);rr=[r for r,k in zip(rows,mask) if k];qq=np.array([r['qpos'][7:] for r in rr]);r0=rr[0]
   q['body_windows'][name]={k:vector((np.array([r[k] for r in rr])-r0[k])*100) for k in ['wrist','shoulder_xyz','root']}
   q['body_windows'][name]['wrist_relative_root_cm']=vector((np.array([np.array(r['wrist'])-r['root'] for r in rr])-(np.array(r0['wrist'])-r0['root']))*100)
   q['body_windows'][name]['actual_joint_motion_deg']={g:dict(rms=float(np.degrees(rms(qq[:,sl]-qq[0,sl]))),max_abs=float(np.degrees(abs(qq[:,sl]-qq[0,sl]).max()))) for g,sl in dict(legs=slice(0,12),waist=slice(12,15),left_arm=slice(15,22),right_arm=slice(22,29)).items()}
  coords=[c for c in cs if c['native_coords'] is not None]
  if coords:
   q['native_coordinate_windows']={}
   for name,lo,hi in [('precontact',rd,on),('contact',on,rel),('recovery',rel,end),('late_recovery',rel+6.4,end)]:
    cc=[c['native_coords'] for c in coords if lo-1e-8<=c['time']<hi-1e-8];v=np.asarray(cc);q['native_coordinate_windows'][name]=dict(start=v[0].tolist(),end=v[-1].tolist(),min=v.min(0).tolist(),max=v.max(0).tolist())
  q['recovery_curve']=[]
  for dt in np.arange(0,8.001,.1):
   r=rows[int(np.argmin(abs(ts-rel-dt)))];c=cs[int(np.argmin(abs(ct-rel-dt)))];q['recovery_curve'].append(dict(after_release_s=round(float(dt),2),root_speed_mps=float(np.linalg.norm(r['qvel'][:3])),wrist=r['wrist'],shoulder=r['shoulder_xyz'],root=r['root'],height=r['height'],tilt_deg=r['tilt_deg'],foot_load=r['foot_normal_force_n'],force_xyz=r['force'],native_coords=c['native_coords'],gate=c['local_gate']))
  q['recovery_speed_2_4_6_8']=[next(r['root_speed_mps'] for r in q['recovery_curve'] if r['after_release_s']==t) for t in [2,4,6,8]]
  out[e['label']]=q
 assert all(len(v)==1 for v in ready.values())
 for label,e in raw.items():
  if not e['condition']['enabled']:continue
  q=out[label];base=out.get(e['configuration']+'__native_hold');on=e['condition']['onset'];rel=e['condition']['release'];rd=e['control_ready_s']
  if base:
   comparisons[label]={phase:{metric:1-q['phases'][phase][metric]/base['phases'][phase][metric] for metric in ['tracking_rms_cm','force_rms_n']} for phase in ['move','hold','all_contact']}
  free_key=label.removesuffix('_z_replay')+'__free' if label.endswith('_z_replay') else label+'__free'
  if free_key not in raw:continue
  free=raw[free_key];rows=e['physics'];rr=free['physics'];ts=np.array([r['time'] for r in rows]);ft=np.array([r['force_time'] for r in rows]);pre=ft<on-1e-8;cs=e['control'];ds=free['control'];ct=np.array([c['time'] for c in cs]);cp=ct<on-1e-8
  parity={k:float(abs(np.array([r[k] for r in rows])[pre]-np.array([r[k] for r in rr])[pre]).max()) for k in ['qpos','qvel','wrist','force']};parity['action']=float(abs(np.asarray([c['action'] for c in cs])[cp]-np.asarray([c['action'] for c in ds])[cp]).max());assert max(parity.values())==0
  pair=dict(precontact_parity=parity,physical_contact_effect={},command_contact_effect={})
  for phase,lo,hi in [('move',on,on+e['condition']['ramp']),('hold',on+e['condition']['ramp'],rel),('contact',on,rel),('recovery',rel,e['condition']['end'])]:
   mask=(ft>=lo-1e-8)&(ft<hi-1e-8);cm=(ct>=lo-1e-8)&(ct<hi-1e-8)
   pair['physical_contact_effect'][phase]={k:vector((np.array([r[k] for r in rows])-np.array([r[k] for r in rr]))[mask]*100) for k in ['wrist','shoulder_xyz','root']}
   d=(np.array([np.array(r['wrist'])-r['root'] for r in rows])-np.array([np.array(r['wrist'])-r['root'] for r in rr]))[mask]*100;pair['physical_contact_effect'][phase]['wrist_relative_root']=vector(d)
   dq=(np.asarray([c['q_PD'] for c in cs])-np.asarray([c['q_PD'] for c in ds]))[cm]
   pair['command_contact_effect'][phase]={g:float(np.degrees(rms(dq[:,sl]))) for g,sl in dict(legs=slice(0,12),waist=slice(12,15),right_arm=slice(22,29)).items()}
  pairs[label]=pair
 # No-contact pose departure is measured against a matched, freely moving native controller.
 for label,e in raw.items():
  if e['condition']['enabled']:continue
  b=raw.get(e['configuration']+'__native_hold__free')
  if b is None:continue
  ts=np.asarray([r['time'] for r in e['physics']]);mask=ts>=e['control_ready_s']-1e-8
  out[label]['extra_vs_native_free']={k:vector((np.asarray([r[k] for r in e['physics']])-np.asarray([r[k] for r in b['physics']]))[mask]*100) for k in ['wrist','shoulder_xyz','root']}
 result=dict(episodes=out,contact_vs_free=pairs,relative_gains_vs_native=comparisons,ready_parity={k:list(v) for k,v in ready.items()},no_fitting=True,scope='Tracking only actual contact. Native free body is not mechanically fixed. Contact/free effect includes physics and BFM feedback; no claim to isolate explicit force channel. One unseen u and seed. No final-data training.')
 (R/'analysis/final-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 for label,q in out.items():
  p=q['phases'].get('all_contact',{});print(label,'error/hold',p.get('tracking_rms_cm'),q['phases'].get('hold',{}).get('tracking_rms_cm'),'force',p.get('force_rms_n'),'cap',p.get('force_cap_fraction'),'pre_wrist',q['windows']['precontact']['wrist_delta_cm'],'recover',q['recovery_speed_2_4_6_8'],'qPD',q['command_peak'])
 print('GAINS',json.dumps(comparisons))
if __name__=='__main__':main()
