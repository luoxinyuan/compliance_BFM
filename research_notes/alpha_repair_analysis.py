"""Paired saved-record metrics, old regression and sealed interruption qualification."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
from contact_gate_summary import summarize
from contact_gate import ContactGate


def analyze(run):
 read=lambda p:json.loads(Path(p).read_text())
 protocol=read(Path(__file__).with_name('alpha_repair_protocol.json'));gp=read(Path(__file__).with_name('contact_gate_protocol.json'))
 ledger=read(run/'compute-ledger.json')
 done=lambda stage:next(Path(a['path'])/'outputs' for a in ledger['attempts'] if a['stage']==stage and a.get('result_status')=='completed')
 phys=done('physics');train=done('train');runs=run.parent
 normal=runs/'issue21-alpha-student-20261030A/physics-attempt-01/outputs'
 teacher=runs/'issue21-contact-gate-20261030A/physics-attempt-01/outputs'
 oldgap=runs/'issue21-alpha-interruption-20261030A/physics-attempt-01/outputs'
 episodes={};old={};audit={};manifest=[];checks={};regression={};gains={}
 for cond in protocol['conditions']:
  name=cond['name'];gap='disconnect_start' in cond
  refdir=oldgap if gap else normal if name=='final_new' else teacher
  for ctl in ['standing','contact','A','B']:
   label=name+'__'+ctl;f=(phys if ctl in cond['controllers'] else refdir)/(label+'.json')
   raw=read(f);assert raw['boundary'] is None
   v=summarize(raw,gp['gate']);episodes[label]=v
   if ctl in ('A','B'):v['shadow_gate_events']=v.pop('gate_events')
   cs=raw['control'];ph=raw['physics'];t=np.array([c['time'] for c in cs]);alpha=np.array([c['alpha'] for c in cs])
   at=lambda tt:float(alpha[abs(t-tt).argmin()])
   w=v['phases']['whole'];v['physical_safe']=w['failed_samples']==0 and w['samples_without_either_foot_load']==0 and w['actual_joint_limit_excess_max_deg']==0 and w['undesired_contacts_max']==0
   v['alpha_step_max']=float(abs(np.diff(alpha)).max());v['alpha_final']=float(alpha[-1]);v['alpha_peak']=float(alpha.max())
   v['torque_5ms_step_max_Nm']=float(abs(np.diff(np.asarray([r['torque'] for r in ph]),axis=0)).max())
   force=np.array([r['force'] for r in ph]);wrench=np.asarray([r['applied_wrist_wrench'] for r in ph]) if 'applied_wrist_wrench' in ph[0] else None
   if f.parent==phys:
    assert len(ph)==round((cond['end']-3)*200)
    prior=[raw['initial']]+ph[3::4][:-1]
    lag=max(float(abs(np.asarray(c['measured_force_xyz'])-r['force']).max()) for c,r in zip(cs,prior));assert lag==0
    assert np.array_equal(force,wrench[:,:3])
    audit[label]=dict(complete=True,force_sensor_lag_max=lag,actual_wrench_equals_force=True,common_state_sha256=raw['common_state_sha256'])
    if not cond['enabled']:assert not force.any() and not any(r['connection_enabled'] for r in ph)
   if ctl in ('contact','A','B'):
    shadow=np.array([c.get('shadow_alpha',c['alpha']) for c in cs]);gate=ContactGate(gp['gate'])
    assert max(abs(gate.step(c['measured_force_xyz'])['alpha']-s) for c,s in zip(cs,shadow))==0
    v['shadow_alpha_rms_whole']=float(np.sqrt(np.mean((alpha-shadow)**2)))
   if gap:
    start,end=cond['disconnect_start'],cond['disconnect_end'];release=cond['onset']+cond['duration']
    pt=np.array([r['force_time'] for r in ph]);mask=(pt>=start-1e-8)&(pt<end-1e-8)
    assert int(mask.sum())==round((end-start)*200) and not force[mask].any() and not wrench[mask].any()
    assert all(not r['active'] for r in np.array(ph,dtype=object)[mask])
    v['zero_force_gap_samples']=int(mask.sum())
    v['alpha_landmarks']={str(round(tt,3)):at(tt) for tt in [start-.02,start+.06,end,end+.12,end+.5,end+1,release,cond['end']-.02]}
    plateau=(t>=start+.06-1e-8)&(t<end-1e-8);roi=(t>=start-1e-8)&(t<end+1-1e-8)
    v['short_plateau_max_change']=float(abs(alpha[plateau]-at(start+.06)).max())
    v['gap_end_alpha_fraction_drop']=1-at(end)/max(at(start+.06),1e-8)
    pm=(pt>=end-1e-8)&(pt<release-1e-8)&np.array([r['active'] for r in ph])
    err=np.array([r['anchor'] for r in ph])-np.array([r['wrist'] for r in ph])
    v['post_reconnect']=dict(tracking_rms_cm=float(np.sqrt(np.mean(np.sum(err[pm]**2,axis=1)))*100),force_rms_n=float(np.sqrt(np.mean(np.sum(force[pm]**2,axis=1)))))
    if ctl in ('contact','A','B'):
     v['shadow_alpha_rms_gap_resume1s']=float(np.sqrt(np.mean((alpha[roi]-shadow[roi])**2)))
     v['shadow_alpha_max_gap_resume1s']=float(abs(alpha[roi]-shadow[roi]).max())
     bad=np.flatnonzero(roi&(abs(alpha-shadow)>.1));v['first_shadow_error_over_0_1']=None if not len(bad) else dict(time=float(t[bad[0]]),alpha=float(alpha[bad[0]]),shadow=float(shadow[bad[0]]))
     if ctl in ('A','B'):
      q=np.array([c['q_PD'] for c in cs]);sq=np.array([c['shadow_q_PD'] for c in cs]);v['shadow_qPD_rms_gap_resume1s_deg']=float(np.degrees(np.sqrt(np.mean((q[roi]-sq[roi])**2))))
    if f.parent==phys:audit[label]['real_zero_gap_samples']=int(mask.sum())
   manifest.append(dict(label=label,path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),new_branch=f.parent==phys))
   if ctl in ('A','B') and cond['split']!='sealed_new_gap':
    oldfile=(oldgap if gap else normal)/(label+'.json');ov=summarize(read(oldfile),gp['gate']);old[label]=ov
    record=dict(old_checkpoint=('original_'+ctl),old_path=str(oldfile),old_sha256=hashlib.sha256(oldfile.read_bytes()).hexdigest(),qPD_step_change_deg=w['qPD_step_max_deg']-ov['phases']['whole']['qPD_step_max_deg'],alpha_peak_change=v['alpha_peak']-ov['phases']['whole']['alpha_max'])
    if cond['enabled']:
     a=v['phases']['all_contact'];b=ov['phases']['all_contact'];record.update(error_relative_change=a['tracking_rms_cm']/b['tracking_rms_cm']-1,force_relative_change=a['force_rms_n']/b['force_rms_n']-1)
    regression[label]=record
  base=episodes[name+'__standing']
  for ctl in ['contact','A','B']:
   label=name+'__'+ctl;v=episodes[label]
   if cond['enabled']:
    a=v['phases']['all_contact'];b=base['phases']['all_contact'];gains[label]=dict(error_reduction=1-a['tracking_rms_cm']/b['tracking_rms_cm'],force_reduction=1-a['force_rms_n']/b['force_rms_n'])
   check=dict(safe=v['physical_safe'])
   if cond['enabled']:check['recovery']=v['root_speed_release_plus_2s_mps']<=.02
   if gap:
    check['alpha_shadow']=v['shadow_alpha_rms_gap_resume1s']<=.05
    check['short_retains' if end-start<.4 else 'long_returns']=(v['short_plateau_max_change']<=.05 if end-start<.4 else v['gap_end_alpha_fraction_drop']>=.2)
    check['post_reconnect_better_than_standing']=v['post_reconnect']['tracking_rms_cm']<base['post_reconnect']['tracking_rms_cm']
   elif not cond['enabled']:check['no_contact_false_trigger']=v['alpha_peak']<=.01
   elif cond['name']!='pulse':
    check.update(normal_tracking=gains[label]['error_reduction']>=.30,normal_force=gains[label]['force_reduction']>=.05)
    if ctl in ('A','B'):
     change=regression[label];check['no_material_task_regression']=max(change['error_relative_change'],change['force_relative_change'])<=.05
   checks[label]=check
 assert len(audit)==24 and len({v['common_state_sha256'] for v in audit.values()})==1
 result=dict(episodes=episodes,old_student_regression=old,regression=regression,gains=gains,checks=checks,
   qualified={ctl:all(all(v.values()) for k,v in checks.items() if k.endswith('__'+ctl)) for ctl in ['contact','A','B']},
   sealed_gap_qualified={ctl:all(all(checks[c['name']+'__'+ctl].values()) for c in protocol['conditions'] if c['split']=='sealed_new_gap') for ctl in ['contact','A','B']},
   independent_audit=audit,manifest=manifest,fit=read(train/'paired-fit-result.json'),compute=ledger)
 (run/'analysis/repair-results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 print(json.dumps(dict(qualified=result['qualified'],sealed=result['sealed_gap_qualified'],checks=checks),indent=2))

if __name__=='__main__':analyze(Path(sys.argv[1]))
