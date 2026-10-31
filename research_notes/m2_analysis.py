"""Saved M2 traces: physical comparisons, precontact parity and full recovery; no models."""
import gzip,json
from pathlib import Path
import numpy as np
from m0_analysis import summarize
from ccl02a_teacher import audit_episode
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
RUN=P/'bridge-v2-owned/runs/m2-unseen-body-20261031A'
def read(p):return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
def window(e,lo,hi):
 rows=e['physics'];cs=e['control'];ts=np.array([r['time'] for r in rows]);mask=(ts>=lo-1e-8)&(ts<=hi+1e-8);rr=[r for r,k in zip(rows,mask) if k];cc=[c for c in cs if lo-1e-8<=c['time']<hi-1e-8]
 if not rr:return None
 w=np.array([r['wrist'] for r in rr]);q=np.array([r['qpos'] for r in rr]);vel=np.array([r['qvel'] for r in rr]);loads=np.array([r['foot_normal_force_n'] for r in rr])>1.;feet=np.array([r['feet_xyz'] for r in rr]);speed=np.linalg.norm(vel[:,:3],axis=-1)
 return dict(start=lo,end=hi,wrist_delta_cm=((w[-1]-w[0])*100).tolist(),wrist_range_cm=((w.max(0)-w.min(0))*100).tolist(),root_delta_cm=((q[-1,:3]-q[0,:3])*100).tolist(),root_speed_rms_mps=float(np.sqrt(np.mean(speed**2))),root_speed_max_mps=float(speed.max()),root_speed_end_mps=float(speed[-1]),q_actual_range_max_deg=float(np.rad2deg((q[:,7:].max(0)-q[:,7:].min(0)).max())),both_feet_load_fraction=float(loads.all(-1).mean()),any_foot_load_fraction=float(loads.any(-1).mean()),foot_xy_displacement_max_cm=float(np.linalg.norm(feet[:,:,:2]-feet[0,:,:2],axis=-1).max()*100),tilt_max_deg=max(r['tilt_deg'] for r in rr),qPD_step_max_deg=None if not cc else float(np.rad2deg(max(c['q_PD_step_max_rad'] for c in cc))),force_peak_n=max(float(np.linalg.norm(r['force'])) for r in rr))
def main():
 ledger=read(RUN/'compute-ledger.json');initial=read(P/'bridge-v2-owned/runs/issue21-ccl02a-20261029T153900Z/attempt-01/outputs/dev-2-initial-state.json');report={};raw={};parity=[]
 for a in ledger['attempts']:
  if a.get('result_status')!='completed':continue
  out=Path(a['path'])/'outputs';contract=read(out/'action-contract.json')
  for f in sorted(out.glob('*.json.gz')):
   e=read(f)
   if 'physics' not in e:continue
   key=a['stage']+'/'+e['label'];x=summarize(e,{});x['alpha_semantics']='unused legacy field; use native_coords or rule_record';x['source']=str(f);x['ready_sha']=e['ready_physical_state_sha256'];x['rule_context']=next((c['rule_context'] for c in e['control'] if c['rule_context'] is not None),None);x['geometry']=e['geometry'];on=e['condition']['onset'];rel=e['condition']['release'];ready=e['control_ready_s']
   x['native_pre_ready_last_second']=window(e,ready-1,ready);x['post_handoff_precontact']=window(e,ready+.75,on);x['recovery_full']=window(e,rel,rel+6)
   cs=e['control'];rows=e['physics'];ts=np.array([r['time'] for r in rows]);ct=np.array([c['time'] for c in cs]);q=np.array([c['q_PD'] for c in cs]);dq=np.abs(np.diff(q,axis=0));dq[ct[1:]<ready-1e-8]=-1;i,j=np.unravel_index(dq.argmax(),dq.shape)
   x['deployed_command_peak']=dict(time=cs[i+1]['time'],joint=contract['dof_names'][j],target_step_deg=float(np.rad2deg(dq[i,j])),actual_20ms_motion_deg=float(np.rad2deg(cs[i+1]['pre_qpos'][7+j]-cs[i]['pre_qpos'][7+j])),latent_step_deg=cs[i+1]['z_step_deg'])
   x['recovery_curve']=[]
   for dt in np.arange(0,6.001,.2):
    idx=int(np.argmin(abs(ts-(rel+dt))));ci=int(np.argmin(abs(ct-(rel+dt))));r=rows[idx];c=cs[ci]
    x['recovery_curve'].append(dict(after_release_s=round(float(dt),2),time=r['time'],root_speed_mps=float(np.linalg.norm(r['qvel'][:3])),root_height_m=r['height'],pelvis_tilt_deg=r['tilt_deg'],wrist_xyz_m=r['wrist'],root_xyz_m=r['root'],force_xyz_n=r['force'],foot_normal_force_n=r['foot_normal_force_n'],native_coords=c['native_coords'],rule=c['rule_record']))
   x['root_speed_release_2_4_6_mps']=[next(r['root_speed_mps'] for r in x['recovery_curve'] if r['after_release_s']==dt) for dt in [2.,4.,6.]]
   x['deployment_parity']=audit_episode(e,contract,initial);z=np.array([c['executed_z'][0] for c in cs]);x['latent_norm_max_error']=float(abs(np.linalg.norm(z,axis=-1)-16).max());assert x['latent_norm_max_error']<1e-4
   assert np.linalg.norm(np.array([r['force'] for r in rows]),axis=-1).max()<5.000001
   report[key]=x;raw[key]=e
 for ctl in ['native_hold','teacher','rule','distill','alignment']:
  key='comparison/midpoint__'+ctl;nk=key+'__free';basekey='comparison/midpoint__native_hold__free'
  if key not in raw or nk not in raw:continue
  a,b=raw[key],raw[nk];on=a['condition']['onset'];rr=[r for r in a['physics'] if r['force_time']<on-1e-8];ss=[r for r in b['physics'] if r['force_time']<on-1e-8];cc=[r for r in a['control'] if r['time']<on-1e-8];dd=[r for r in b['control'] if r['time']<on-1e-8]
  errors={k:float(abs(np.array([r[k] for r in rr])-np.array([r[k] for r in ss])).max()) for k in ['qpos','qvel','wrist','force']};errors['action']=float(abs(np.array([r['action'] for r in cc])-np.array([r['action'] for r in dd])).max());assert max(errors.values())==0;parity.append(dict(controller=ctl,errors=errors))
  if basekey in raw:
   base=raw[basekey];mask=np.array([r['time'] for r in b['physics']])>=b['control_ready_s'];dw=np.array([r['wrist'] for r in b['physics']])-np.array([r['wrist'] for r in base['physics']]);report[nk]['extra_wrist_vs_native_rms_cm']=float(np.sqrt(np.mean(np.sum(dw[mask]**2,axis=-1)))*100);report[nk]['extra_wrist_vs_native_peak_cm']=float(np.linalg.norm(dw[mask],axis=-1).max()*100)
 if raw:
  hashes={e['ready_physical_state_sha256'] for e in raw.values()};assert len(hashes)==1,hashes
 result=dict(episodes=report,precontact_parity=parity,no_training=True,scope='Actualbody previouslyphysicallyqualified but notStudenttraining. Recovery6s; no zero-forceanchorerrorcounted. Allphysical andqualitycosts retained.')
 dest=RUN/'analysis';dest.mkdir(exist_ok=True);(dest/'physical-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 for key,e in report.items():
  a=e['phases'].get('all_contact',{});h=e['phases'].get('hold',{});print(key, 'error_all/hold_cm',a.get('tracking_rms_cm'),h.get('tracking_rms_cm'),'force',a.get('force_rms_n'),'recover2/4/6',e['root_speed_release_2_4_6_mps'],'preReady',e['native_pre_ready_last_second'])
if __name__=='__main__':main()
