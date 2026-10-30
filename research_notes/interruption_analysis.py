"""Saved-record interruption metrics and force-memory audit; no learning/simulator."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
from contact_gate_summary import summarize
from contact_gate import ContactGate
from alpha_memory import ForceMemory


def analyze(run):
 read=lambda p:json.loads(Path(p).read_text())
 notes=Path(__file__).parent;protocol=read(notes/'interruption_protocol.json');gate_cfg=read(notes/'contact_gate_protocol.json')['gate']
 out=run/'physics-attempt-01/outputs';contract=read(out/'action-contract.json');summaries={};episodes={};manifest=[];audit={}
 def window(ep,lo,hi,score=False):
  ph=[r for r in ep['physics'] if lo-1e-8<=r['force_time']<hi-1e-8 and (r['active'] or not score)]
  cs=[c for c in ep['control'] if lo-1e-8<=c['time']<hi-1e-8]
  if not ph:return None
  force=np.asarray([r['force'] for r in ph]);error=np.asarray([r['anchor'] for r in ph])-np.asarray([r['wrist'] for r in ph]);fn=np.linalg.norm(force,axis=1)
  return dict(physics_samples=len(ph),tracking_rms_cm=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))*100) if score else None,
    tracking_xyz_rms_cm=(np.sqrt(np.mean(error**2,axis=0))*100).tolist() if score else None,
    force_rms_n=float(np.sqrt(np.mean(fn**2))),force_peak_n=float(fn.max()),force_mean_xyz_n=force.mean(0).tolist(),force_xyz_rms_n=np.sqrt(np.mean(force**2,axis=0)).tolist(),force_cap_fraction=float((fn>=5-1e-6).mean()),
    alpha_first=cs[0]['alpha'] if cs else None,alpha_last=cs[-1]['alpha'] if cs else None,alpha_min=min(c['alpha'] for c in cs) if cs else None,alpha_max=max(c['alpha'] for c in cs) if cs else None,
    qPD_step_max_deg=float(np.degrees(max(c['q_PD_step_max_rad'] for c in cs))) if cs else None,
    wrist_delta_xyz_cm=((np.asarray(ph[-1]['wrist'])-ph[0]['wrist'])*100).tolist(),root_delta_xyz_cm=((np.asarray(ph[-1]['root'])-ph[0]['root'])*100).tolist(),
    root_min_height_cm=min(r['height'] for r in ph)*100,tilt_max_deg=max(r['tilt_deg'] for r in ph),no_foot_load_samples=sum(not any(f>1 for f in r['foot_normal_force_n']) for r in ph),
    peak_wrist_applied_wrench=np.abs(np.asarray([r['applied_wrist_wrench'] for r in ph])).max(0).tolist())
 for cond in protocol['conditions']:
  for ctl in protocol['controllers']:
   label=cond['name']+'__'+ctl;p=out/(label+'.json')
   if not p.exists():continue
   ep=read(p);episodes[label]=ep;v=summarize(ep,gate_cfg);summaries[label]=v
   if ctl in ('A','B'):v['shadow_gate_events']=v.pop('gate_events')
   start,end=cond['disconnect_start'],cond['disconnect_end'];release=cond['onset']+cond['duration'];cs=ep['control'];ph=ep['physics']
   v['interruption_phases']={name:window(ep,lo,hi,score) for name,lo,hi,score in [('pre_gap',cond['onset'],start,True),('gap',start,end,False),('reconnect1s',end,end+1,True),('post_reconnect',end,release,True),('final_recovery',release,release+2,False)]}
   alpha=np.array([c['alpha'] for c in cs]);times=np.array([c['time'] for c in cs]);at=lambda t:float(alpha[abs(times-t).argmin()])
   plateau=(times>=start+.06-1e-8)&(times<end-1e-8)
   v['alpha_landmarks']={str(round(t,3)):at(t) for t in [start-.02,start,start+.06,end,end+.12,end+.5,end+1,release,release+2-.02]}
   v['short_plateau_max_change']=float(abs(alpha[plateau]-at(start+.06)).max())
   v['gap_end_alpha_fraction_drop']=1-at(end)/max(at(start+.06),1e-8)
   if ctl in ('contact','A','B'):
    shadow=np.array([c.get('shadow_alpha',c['alpha']) for c in cs]);roi=(times>=start-1e-8)&(times<end+1-1e-8)
    v['shadow_alpha_rms_gap_resume1s']=float(np.sqrt(np.mean((alpha[roi]-shadow[roi])**2)))
    v['shadow_alpha_max_gap_resume1s']=float(abs(alpha[roi]-shadow[roi]).max())
    bad=np.flatnonzero(roi&(abs(alpha-shadow)>.1));v['first_shadow_error_over_0_1']=None if not len(bad) else dict(time=float(times[bad[0]]),alpha=float(alpha[bad[0]]),shadow=float(shadow[bad[0]]))
    errors=[]
    for i,c in enumerate(cs):
     gate=ContactGate(gate_cfg)
     for prev in cs[max(0,i-319):i+1]:last=gate.step(prev['measured_force_xyz'])
     errors.append(abs(last['alpha']-shadow[i]))
    v['force_fifo320_reconstruct_shadow_max']=float(max(errors))
    if ctl in ('A','B'):
     q=np.asarray([c['q_PD'] for c in cs]);tq=np.asarray([c['shadow_q_PD'] for c in cs]);v['shadow_qPD_rms_gap_resume1s_deg']=float(np.degrees(np.sqrt(np.mean((q[roi]-tq[roi])**2))))
   # Independent real wrench + sensor chronology + exact pre-interruption parity.
   previous=[ep['initial']]+ph[3::4][:-1]
   force_lag=max(float(abs(np.asarray(c['measured_force_xyz'])-r['force']).max()) for c,r in zip(cs,previous));assert force_lag==0
   force=np.asarray([r['force'] for r in ph]);wrench=np.asarray([r['applied_wrist_wrench'] for r in ph]);assert np.array_equal(force,wrench[:,:3])
   gm=np.array([start-1e-8<=r['force_time']<end-1e-8 for r in ph]);assert gm.sum()==round((end-start)*200) and not force[gm].any() and not wrench[gm].any()
   assert all(not r['active'] and not r['connection_enabled'] for r in np.asarray(ph,dtype=object)[gm])
   parity=read(out/(label+'-uninterrupted-prefix-parity.json'));assert max(parity.values())==0
   audit[label]=dict(actual_gap_physics_samples=int(gm.sum()),zero_force_and_wrench_exact=True,sensor_lag_max=force_lag,uninterrupted_prefix_exact=True)
   manifest.append(dict(label=label,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 improvements={};checks={}
 for cond in protocol['conditions']:
  name=cond['name'];baseline=summaries[name+'__standing']
  for ctl in ['contact','A','B']:
   label=name+'__'+ctl
   if label not in summaries:continue
   v=summaries[label];improvements[label]={}
   for phase in ['all_contact','hold']:
    p=v['phases'][phase];b=baseline['phases'][phase];improvements[label][phase]=dict(error_reduction=1-p['tracking_rms_cm']/b['tracking_rms_cm'],force_reduction=1-p['force_rms_n']/b['force_rms_n'])
   a=v['interruption_phases']['post_reconnect'];b=baseline['interruption_phases']['post_reconnect'];improvements[label]['post_reconnect']=dict(error_reduction=1-a['tracking_rms_cm']/b['tracking_rms_cm'],force_reduction=1-a['force_rms_n']/b['force_rms_n'])
   whole=v['phases']['whole'];checks[label]=dict(complete=v['boundary'] is None,physical_safe=whole['failed_samples']==0 and whole['samples_without_either_foot_load']==0 and whole['actual_joint_limit_excess_max_deg']==0,
    recovery=v['root_speed_release_plus_2s_mps']<=.02,post_reconnect_better_than_standing=improvements[label]['post_reconnect']['error_reduction']>0,
    shadow_alpha_rms=v['shadow_alpha_rms_gap_resume1s']<=protocol['qualification']['alpha_shadow_rms_gap_and_first_1s_after_resume_max'])
   if name=='short02':checks[label]['short_retains']=v['short_plateau_max_change']<=protocol['qualification']['short_alpha_plateau_abs_change_max']
   else:checks[label]['long_returns']=v['gap_end_alpha_fraction_drop']>=protocol['qualification']['long_fraction_alpha_reduction_at_gap_end_min']
 result=dict(episodes=summaries,improvements=improvements,qualification=checks,all_qualified={ctl:all(all(v.values()) for k,v in checks.items() if k.endswith('__'+ctl)) for ctl in ['contact','A','B']},
   independent_physics_audit=audit,manifest=manifest,protocol=protocol)
 (run/'analysis/interruption-results.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(dict(qualification=checks,all_qualified=result['all_qualified']),indent=2))

if __name__=='__main__':analyze(Path(sys.argv[1]))
