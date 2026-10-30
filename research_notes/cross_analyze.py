"""Analyze saved physics; fixed semantics and real interaction metrics kept separate."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
from contact_gate_summary import summarize
from contact_gate import ContactGate

def metric(path):
 ep=json.loads(Path(path).read_text());gp=json.loads(Path(__file__).with_name('contact_gate_protocol.json').read_text())['gate']
 assert ep['boundary'] is None
 c=ep['condition'];cs=ep['control'];ph=ep['physics'];v=summarize(ep,gp)
 t=np.array([r['time'] for r in cs]);alpha=np.array([r['alpha'] for r in cs]);force=np.array([r['force'] for r in ph]);pt=np.array([r['force_time'] for r in ph]);wrench=np.array([r['applied_wrist_wrench'] for r in ph]) if 'applied_wrist_wrench' in ph[0] else None
 assert len(ph)==round((c['end']-3)*200)
 if 'issue21-alpha-cross-20261031A' in str(path):assert wrench is not None
 if wrench is not None:assert np.array_equal(force,wrench[:,:3])
 prior=[ep['initial']]+ph[3::4][:-1]
 assert all(np.array_equal(r['measured_force_xyz'],p['force']) for r,p in zip(cs,prior))
 whole=v['phases']['whole']
 v.update(safe=whole['failed_samples']==0 and whole['samples_without_either_foot_load']==0 and whole['actual_joint_limit_excess_max_deg']==0 and whole['undesired_contacts_max']==0,alpha_peak=float(alpha.max()),alpha_step_max=float(abs(np.diff(alpha)).max()),source=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),common_state_sha256=ep['common_state_sha256'])
 v['torque_step_max_Nm']=float(abs(np.diff(np.array([p['torque'] for p in ph]),axis=0)).max())
 v['actual_wrench_log_verified']=wrench is not None
 if not c['enabled']:
  assert not force.any()
  if 'connection_enabled' in ph[0]:assert not any(p['connection_enabled'] for p in ph)
 if ep['controller'] in ('contact','A','B'):
  shadow=np.array([r.get('shadow_alpha',r['alpha']) for r in cs]);g=ContactGate(gp)
  assert max(abs(g.step(r['measured_force_xyz'])['alpha']-s) for r,s in zip(cs,shadow))==0
 if 'disconnect_start' in c:
  lo,hi=c['disconnect_start'],c['disconnect_end'];mask=(pt>=lo-1e-8)&(pt<hi-1e-8)
  assert mask.sum()==round((hi-lo)*200) and not force[mask].any() and not wrench[mask].any()
  at=lambda tt:float(alpha[abs(t-tt).argmin()]);roi=(t>=lo-1e-8)&(t<hi+1-1e-8);plateau=(t>=lo+.06-1e-8)&(t<hi-1e-8)
  v.update(plateau_change=float(abs(alpha[plateau]-at(lo+.06)).max()),fraction_drop=1-at(hi)/max(at(lo+.06),1e-8),alpha_start=at(lo+.06),alpha_gap_end=at(hi),gap_zero_physics_steps=int(mask.sum()))
  pm=(pt>=hi-1e-8)&(pt<c['onset']+c['duration']-1e-8)&np.array([p['active'] for p in ph]);err=np.array([p['anchor'] for p in ph])-np.array([p['wrist'] for p in ph])
  v['post_reconnect_error_cm']=float(np.sqrt(np.mean(np.sum(err[pm]**2,axis=1)))*100)
  if ep['controller'] in ('contact','A','B'):
   v['shadow_rms']=float(np.sqrt(np.mean((alpha[roi]-shadow[roi])**2)))
   v['behavior_checks']=dict(shadow=v['shadow_rms']<=.05,pause_or_retract=v['plateau_change']<=.05 if hi-lo<.4 else v['fraction_drop']>=.2)
   v['behavior_pass']=all(v['behavior_checks'].values())
   v['alpha_landmarks']={str(round(tt,3)):dict(student=at(tt),shadow=float(shadow[abs(t-tt).argmin()])) for tt in [lo-.02,lo+.06,hi,hi+.12,hi+.5,hi+1]}
 return v

def analyze(run,stage):
 study=json.loads(Path(__file__).with_name('alpha_cross_protocol.json').read_text());runs=run.parent
 out=run/(stage+'-attempt-01/outputs');conditions=study['matrix'] if stage=='diagnostic' else study['conditions'];results={};regression={}
 prior=runs/'issue21-alpha-repair-20261031A/physics-attempt-01/outputs'
 for c in conditions:
  for ctl in ['standing','contact','A','B']:
   folder=Path(c['reference_dir']) if stage=='diagnostic' or ctl not in c['controllers'] else out
   path=folder/(c['name']+'__'+ctl+'.json');key=c['name']+'__'+ctl;v=metric(path);results[key]=v
   if ctl in ('A','B') and stage=='final' and c['split']!='sealed_final':
    oldfolder=prior if c['name'] not in ('early_long07','late_short03') else run/'diagnostic-attempt-01/outputs'
    ov=metric(oldfolder/(key+'.json'))
    reg=dict(old_source=ov['source'],old_sha256=ov['sha256'],old_alpha_peak=ov['alpha_peak'])
    if c['enabled']:
     reg.update(error_change=v['phases']['all_contact']['tracking_rms_cm']/ov['phases']['all_contact']['tracking_rms_cm']-1,force_change=v['phases']['all_contact']['force_rms_n']/ov['phases']['all_contact']['force_rms_n']-1)
    regression[key]=reg
  base=results[c['name']+'__standing']
  for ctl in ['contact','A','B']:
   v=results[c['name']+'__'+ctl]
   if c['enabled']:
    a=v['phases']['all_contact'];b=base['phases']['all_contact'];v.update(error_gain=1-a['tracking_rms_cm']/b['tracking_rms_cm'],force_gain=1-a['force_rms_n']/b['force_rms_n'],recovery_pass=v['root_speed_release_plus_2s_mps']<=.02)
   else:v['negative_pass']=v['alpha_peak']<=.01
   if 'disconnect_start' in c:v['post_reconnect_gain']=1-v['post_reconnect_error_cm']/base['post_reconnect_error_cm']
 assert len({v['common_state_sha256'] for v in results.values()})==1
 summary=dict(stage=stage,results=results,regression=regression,all_safe=all(v['safe'] for v in results.values()),source_protocol_sha256=hashlib.sha256(Path(__file__).with_name('alpha_cross_protocol.json').read_bytes()).hexdigest())
 if stage=='diagnostic':
  summary['factor_effects']={}
  for ctl in ['A','B']:
   y=[results[c['name']+'__'+ctl]['shadow_rms'] for c in study['matrix']]
   summary['factor_effects'][ctl]=dict(cells_earlyshort_earlylong_lateshort_latelong=y,late_minus_early_short=y[2]-y[0],late_minus_early_long=y[3]-y[1],long_minus_short_early=y[1]-y[0],long_minus_short_late=y[3]-y[2],difference_in_differences=(y[3]-y[2])-(y[1]-y[0]))
 (run/'analysis'/(stage+'-analysis.json')).write_text(json.dumps(summary,indent=2,allow_nan=False))
 print(json.dumps(dict(stage=stage,all_safe=summary['all_safe'],effects=summary.get('factor_effects'),cells={k:{x:v[x] for x in ['shadow_rms','plateau_change','fraction_drop','behavior_pass','error_gain','force_gain','negative_pass'] if x in v} for k,v in results.items() if not k.endswith('standing')}),indent=2))
if __name__=='__main__':analyze(Path(sys.argv[1]),sys.argv[2])
