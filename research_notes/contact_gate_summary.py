"""Saved-record analysis only; no model, simulator, training, or fitting."""
import argparse,json
from pathlib import Path
import numpy as np
from contact_gate import ContactGate


def summarize(ep,cfg):
    rows=ep['physics'];cs=ep['control'];initial=ep['initial'];cond=ep['condition']
    arr=lambda k:np.asarray([r[k] for r in rows])
    ft=arr('force_time');post=arr('time');active=arr('active').astype(bool)
    force=arr('force');error=arr('anchor')-arr('wrist')
    onset=cond['onset'];release=onset+cond['duration']
    T0=np.asarray(initial['torso_matrix']);Ts=arr('torso_matrix')
    torso=np.degrees(np.arccos(np.clip((np.einsum('nij,ij->n',Ts,T0)-1)/2,-1,1)))
    q=arr('qpos');q0=np.asarray(initial['qpos']);loads=arr('foot_normal_force_n')>1
    spans={'pre':(3,onset),'move':(onset,min(onset+4,release)),
           'hold':(onset+4,release),'recovery':(release,release+2)}
    masks={k:(ft>=lo-1e-8)&(ft<hi-1e-8)&(active if k in ('move','hold') else ~active) for k,(lo,hi) in spans.items()}
    masks.update(all_contact=active,whole=np.ones(len(rows),dtype=bool))
    result=dict(label=ep['label'],condition=cond,seconds=len(rows)/200,boundary=ep['boundary'],phases={})
    for name,mask in masks.items():
        if not mask.any():continue
        ix=np.flatnonzero(mask);track=name in ('move','hold','all_contact')
        if name=='all_contact':sel=[c for c in cs if cond['enabled'] and onset-1e-8<=c['time']<release-1e-8]
        elif name=='whole':sel=cs
        else:
            lo,hi=spans[name];sel=[c for c in cs if lo-1e-8<=c['time']<hi-1e-8]
        fn=np.linalg.norm(force[mask],axis=1)
        v=dict(samples=len(ix),tracking_rms_cm=float(np.sqrt(np.mean(np.sum(error[mask]**2,axis=1)))*100) if track else None,
          tracking_xyz_rms_cm=(np.sqrt(np.mean(error[mask]**2,axis=0))*100).tolist() if track else None,
          tracking_mean_xyz_cm=(error[mask].mean(0)*100).tolist() if track else None,
          force_rms_n=float(np.sqrt(np.mean(fn**2))),force_peak_n=float(fn.max()),force_mean_xyz_n=force[mask].mean(0).tolist(),
          force_xyz_rms_n=np.sqrt(np.mean(force[mask]**2,axis=0)).tolist(),force_cap_fraction=float((fn>=5-1e-6).mean()),
          negative_force_x_fraction=float((force[mask,0]<-.1).mean()),
          root_min_height_cm=float(arr('height')[mask].min()*100),pelvis_tilt_max_deg=float(arr('tilt_deg')[mask].max()),
          torso_rotation_max_deg=float(torso[mask].max()),waist_change_max_abs_deg=np.degrees(abs(q[mask,19:22]-q0[19:22]).max(0)).tolist(),
          both_feet_load_fraction=float(loads[mask].all(1).mean()),any_foot_load_fraction=float(loads[mask].any(1).mean()),
          samples_without_either_foot_load=int((~loads[mask].any(1)).sum()),contact_tangent_speed_max_cm_s=float(arr('contact_tangent_speed_max_mps')[mask].max()*100),
          undesired_contacts_max=int(arr('undesired_ground_contacts')[mask].max()),actual_joint_limit_excess_max_deg=float(np.degrees(arr('joint_limit_excess_max_rad')[mask].max())),
          torque_saturation_mean_fraction=float(arr('torque_saturation')[mask].mean()),failed_samples=int(arr('failed')[mask].sum()))
        for field,key in [('wrist','wrist'),('shoulder_xyz','shoulder'),('root','root')]:
            xyz=(arr(field)[mask]-np.asarray(initial[field]))*100
            v[key+'_end_xyz_cm']=xyz[-1].tolist();v[key+'_mean_xyz_cm']=xyz.mean(0).tolist()
        v['wrist_relative_root_mean_xyz_cm']=(((arr('wrist')-arr('root'))[mask]-(np.asarray(initial['wrist'])-np.asarray(initial['root']))).mean(0)*100).tolist()
        if sel:
            v.update(qPD_step_max_deg=float(np.degrees(max(c['q_PD_step_max_rad'] for c in sel))),
                action_clip_fraction=float(np.mean([c['action_saturation_fraction'] for c in sel])),alpha_min=min(c['alpha'] for c in sel),alpha_max=max(c['alpha'] for c in sel))
            delta=np.asarray([c['delta_q_PD'] for c in sel])
            v['same_obs_additional_qPD_rms_deg']={g:float(np.degrees(np.sqrt(np.mean(delta[:,sl]**2)))) for g,sl in {'legs':slice(0,12),'waist':slice(12,15),'left_arm':slice(15,22),'right_arm':slice(22,29)}.items()}
        result['phases'][name]=v
    ri=int(np.argmin(abs(post-(release+2))))
    if cond['enabled']:
        assert abs(post[ri]-(release+2))<1e-8
        result['root_speed_release_plus_2s_mps']=float(np.linalg.norm(rows[ri]['qvel'][:3]))
        result['wrist_displacement_release_plus_2s_cm']=((np.asarray(rows[ri]['wrist'])-initial['wrist'])*100).tolist()
    result['final_root_speed_mps']=float(np.linalg.norm(rows[-1]['qvel'][:3]))
    result['qpos_20ms_step_max_deg']=float(np.degrees(abs(np.diff(np.asarray([c['pre_qpos'][7:] for c in cs]),axis=0))).max())
    result['first_alpha_over_0_01_time']=next((c['time'] for c in cs if c['alpha']>.01),None)
    result['last_alpha_over_0_01_time']=next((c['time'] for c in reversed(cs) if c['alpha']>.01),None)
    result['gate_events']=[dict(time=c['time'],**c['gate']) for c in cs if c['gate'] and c['gate']['event']]
    if ep['controller']=='contact':
        gate=ContactGate(cfg);zero=ContactGate(cfg)
        result['gate_causal_replay_max_error']=max(abs(gate.step(c['measured_force_xyz'])['alpha']-c['alpha']) for c in cs)
        result['gate_zero_force_replay_alpha_max']=max(zero.step([0,0,0])['alpha'] for c in cs)
        result['entries']=cs[-1]['gate']['entries'];result['exits']=cs[-1]['gate']['exits']
        assert result['gate_causal_replay_max_error']==0 and result['gate_zero_force_replay_alpha_max']==0
    if active.any():
        p=result['phases']['all_contact']
        assert abs(p['tracking_rms_cm']**2-sum(x*x for x in p['tracking_xyz_rms_cm']))<1e-8
    if 'recovery' in result['phases']:
        assert result['phases']['recovery']['tracking_rms_cm'] is None and result['phases']['recovery']['force_peak_n']==0
    return result


def analyze(out,protocol):
    d={}
    for cond in protocol['conditions']:
        for ctl in protocol['controllers']:
            key=cond['name']+'__'+ctl;p=out/(key+'.json')
            if p.exists():d[key]=summarize(json.loads(p.read_text()),protocol['gate'])
    improvements={}
    for cond in protocol['conditions']:
        if not cond['enabled']:continue
        name=cond['name'];base=d.get(name+'__standing',{}).get('phases',{}).get('all_contact')
        if not base:continue
        for ctl in ['clock','contact']:
            p=d.get(name+'__'+ctl,{}).get('phases',{}).get('all_contact')
            if p:improvements[name+'__'+ctl]=dict(error_reduction=1-p['tracking_rms_cm']/base['tracking_rms_cm'],force_reduction=1-p['force_rms_n']/base['force_rms_n'])
    checks={}
    if len(d)==15:
        held=improvements['heldout__contact'];causal=[v for v in d.values() if v['label'].endswith('__contact')]
        checks=dict(complete=all(v['boundary'] is None and abs(v['seconds']-(v['condition']['end']-3))<1e-8 for v in d.values()),
          heldout_tracking=held['error_reduction']>=.30,heldout_force=held['force_reduction']>=.05,
          no_contact_false_trigger=d['no_contact__contact']['entries']==0,
          no_fall=all(v['phases']['whole']['failed_samples']==0 for v in causal),
          no_support_loss=all(v['phases']['whole']['samples_without_either_foot_load']==0 for v in causal),
          recovery_2s=all(v['root_speed_release_plus_2s_mps']<=.02 for v in causal if v['condition']['enabled']))
    return dict(episodes=d,improvements=improvements,qualification_checks=checks,qualification_passed=bool(checks) and all(checks.values()),
      scope='Single seed, fixed standing/skill path/ideal force, unseen onset only; no trained Student or general contact understanding')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('outputs',type=Path);p.add_argument('destination',type=Path);a=p.parse_args()
    protocol=json.loads(Path(__file__).with_name('contact_gate_protocol.json').read_text());result=analyze(a.outputs,protocol)
    a.destination.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(dict(qualification=result['qualification_checks'],improvements=result['improvements']),indent=2))
