"""Read actual saved physics. No BFM, solver stepping, policy or label generation."""
import json,gzip,argparse,math
from pathlib import Path
import numpy as np

def load(p):
    with gzip.open(p,'rt') as f:return json.load(f)

def analyze(ep,switch=None):
    rows=ep['physics'];c=ep['condition'];dt=.005
    t=np.array([r['force_time'] for r in rows]);active=np.array([r['active'] for r in rows],bool)
    root=np.array([r['root'] for r in rows]);w=np.array([r['wrist'] for r in rows]);s=np.array([r['shoulder_xyz'] for r in rows])
    # Force/anchor/contact.point are evaluated together BEFORE this 5ms step.
    # r.wrist/root/shoulder/qpos are the physical response AFTER that step.
    # Match the original geometry_env error metric, rather than mixing times.
    err=np.array([np.array(r['contacts'][0]['anchor'])-r['contacts'][0]['point'] for r in rows]);f=np.array([r['force'] for r in rows])
    feet=np.array([r['feet_xyz'] for r in rows]);load_n=np.array([r['foot_normal_force_n'] for r in rows])
    q=np.array([r['qpos'][7:] for r in rows]);dq=np.array([r['qvel'][6:] for r in rows]);tau=np.array([r['torque'] for r in rows])
    def window(mask):
        ix=np.flatnonzero(mask)
        if len(ix)==0:return None
        ff=f[mask];ee=err[mask];a,b=ix[0],ix[-1]
        return dict(samples=len(ix),duration_s=len(ix)*dt,force_rms=float(np.mean(np.sum(ff*ff,1))**.5),
          force_xyz_rms=np.sqrt(np.mean(ff*ff,0)).tolist(),force_xyz_mean=ff.mean(0).tolist(),force_peak=float(np.linalg.norm(ff,axis=1).max()),
          cap_fraction=float(np.mean(np.linalg.norm(ff,axis=1)>=c.get('total_cap',10)-.001)),
          error_cm=100*float(np.mean(np.sum(ee*ee,1))**.5),error_xyz_mean_cm=(100*ee.mean(0)).tolist(),
          error_xyz_rms_cm=(100*np.sqrt(np.mean(ee*ee,0))).tolist(),root_delta_cm=(100*(root[b]-root[a])).tolist(),
          wrist_world_delta_cm=(100*(w[b]-w[a])).tolist(),wrist_relative_shoulder_delta_cm=(100*((w-s)[b]-(w-s)[a])).tolist(),
          feet_xy_path_cm=(100*np.linalg.norm(np.diff(feet[ix,:,:2],axis=0),axis=2).sum(0)).tolist(),
          root_speed_peak_mps=float(np.linalg.norm(np.array([r['qvel'][:3] for r in rows])[mask],axis=1).max()),
          min_root_height=float(root[mask,2].min()),max_tilt_deg=max(rows[i]['tilt_deg'] for i in ix),
          mechanical_abs_work_J=float(np.sum(abs(tau[mask]*dq[mask]))*dt),external_work_J=float(sum(rows[i]['external_power'] for i in ix)*dt),
          joint_limit_max_rad=max(rows[i]['joint_limit_excess_max_rad'] for i in ix),torque_sat_peak=max(rows[i]['torque_saturation'] for i in ix))
    moving_end=c['onset']+c['ramp']
    masks=dict(active=active,moving=active&(t<moving_end),hold=active&(t>=moving_end),
               recovery=(~active)&(t>=c['release']-1e-8))
    if 'stop_time' in c:
        masks['moving']=active&(t<c['stop_time'])
        masks['decelerating']=active&(t>=c['stop_time'])&(t<c['stop_time']+c['deceleration'])
        masks['hold']=active&(t>=c['stop_time']+c['deceleration'])
    if switch is not None:masks['matched_tail']=active&(t>=switch-1e-8)
    out={k:window(v) for k,v in masks.items()}
    # After true release, anchor geometry is descriptive and excluded from task error.
    if out['recovery']:
        for k in ['error_cm','error_xyz_mean_cm','error_xyz_rms_cm']:out['recovery'].pop(k)
        rr=np.flatnonzero(masks['recovery']);out['recovery']['final_root_speed_mps']=float(np.linalg.norm(rows[rr[-1]]['qvel'][:3]))
    def first(mask,seconds):
        start=None
        for i,ok in enumerate(mask):
            if ok and start is None:start=i
            if not ok:start=None
            if start is not None and (i-start+1)*dt>=seconds:return float(rows[start]['time'])
        return None
    onset_index=np.flatnonzero(t>=c['onset']-1e-8)[0]
    events={}
    for j,side in enumerate(['left','right']):
        events[side+'_unload_below5N_60ms']=first((load_n[:,j]<5)&(t>=c['onset']),.06)
        events[side+'_lift_above_onset_1cm_50ms']=first((feet[:,j,2]-feet[onset_index,j,2]>.01)&(t>=c['onset']),.05)
    vx=np.array([r['qvel'][0] for r in rows]);events['root_vx_above005_200ms']=first((vx>.05)&(t>=c['onset']),.2)
    events['first_cap_50ms']=first(active&(np.linalg.norm(f,axis=1)>=c.get('total_cap',10)-.001),.05)
    out['events']=events
    cp=np.array([np.asarray(r['q_PD']).reshape(-1) for r in ep['control']]);delta=np.degrees(abs(np.diff(cp,axis=0)))
    j=np.unravel_index(delta.argmax(),delta.shape) if delta.size else (0,0)
    out['quality']=dict(qpd_jump_peak_deg=float(delta.max()) if delta.size else 0.,qpd_jump_peak_joint_index=int(j[1]),
        qpd_jump_p99_deg=float(np.quantile(delta,.99)) if delta.size else 0.,qpd_jump_time=ep['control'][j[0]+1]['time'] if delta.size else None,
        actual_q_5ms_jump_peak_deg=float(np.degrees(abs(np.diff(q,axis=0))).max()),actual_dq_peak_radps=float(abs(dq).max()),
        no_foot_contact_ticks=int(sum(not any(r['support']) for r in rows)),
        no_load_above5N_ticks=int(np.sum(np.max(load_n,axis=1)<5)))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();d=a.directory
    summaries=json.loads((d/'summaries.json').read_text());result={};episodes={}
    for r in summaries:
        ep=load(d/(r['label']+'.json.gz'))
        if r.get('suffix') is None and r.get('fixed_from') is None and not r.get('force_off'):episodes[r['label']]=ep
        result[r['label']]=dict(condition=ep['condition'],metrics=analyze(ep,r.get('switch')),intervention=r.get('intervention'),boundary=r['boundary'])
    paired=[]
    for r in summaries:
        if r.get('suffix') is None or r.get('fixed_from') is not None:continue
        condition=r['label'].split('__')[0];name=condition+'__'+r['prefix'];base=analyze(episodes[name],r['switch'])['matched_tail'];after=result[r['label']]['metrics']['matched_tail']
        paired.append(dict(label=r['label'],switch=r['switch'],base=base,after=after,
            tail_error_change_percent=100*(after['error_cm']/base['error_cm']-1),tail_force_change_percent=100*(after['force_rms']/base['force_rms']-1),prefix_exact=all(r['intervention']['sections_equal'].values())))
    early=[r for r in paired if r['label'].startswith('long72__C0_to_I') and r['switch']<=7.5]
    valid=any(r['tail_error_change_percent']<0 and r['tail_force_change_percent']<0 for r in early)
    report=dict(results=result,paired=paired,conditional_PPO_gate=dict(early_valid_recovery=valid,scope='Only measured early matched prefixes; not all visited states; no supervised recovery labels generated'))
    report['time_alignment']='Tracking uses same-force-time contact.point/anchor; kinematics are saved post-step states. Earlier v1 used post-step wrist for tracking and is preserved as the original PPO gate file.'
    (d/'physical-analysis-v2.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'episodes':len(summaries),'early_gate':valid,'paired':[{k:r[k] for k in ['label','tail_error_change_percent','tail_force_change_percent','prefix_exact']} for r in paired]},indent=2))

if __name__=='__main__':main()
