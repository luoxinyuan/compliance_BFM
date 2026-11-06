"""Aggregate real saved physics only. Multi-point burden is not net force."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import numpy as np

def load(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def common_prefix(pair):
    cut=pair[0]['condition']['stop_time'];checks={}
    rows=[[r for r in e['physics'] if r['force_time']<cut-1e-8] for e in pair]
    checks['physics']=dict(equal=digest(rows[0])==digest(rows[1]),records=len(rows[0]),sha256=digest(rows[0]))
    # Commands are timestamped before the 20ms substep. Several diagnostic fields
    # are measured AFTER it, and a substep may straddle a non-grid change time.
    keys=['time','bfm_obs','executed_z','action','q_PD','pre_qpos','pre_qvel','measured_force_xyz']
    cmd=[[{k:r[k] for k in keys} for r in e['control'] if r['time']<cut-1e-8] for e in pair]
    checks['control_commands']=dict(equal=digest(cmd[0])==digest(cmd[1]),records=len(cmd[0]),sha256=digest(cmd[0]))
    full=[[r for r in e['control'] if r['time']+.02<=cut+1e-8] for e in pair]
    checks['complete_control_records']=dict(equal=digest(full[0])==digest(full[1]),records=len(full[0]),sha256=digest(full[0]))
    dec=[[v for i,v in enumerate(e['decisions']) if e['control'][5*i]['time']<cut-1e-8] for e in pair]
    checks['decisions']=dict(equal=digest(dec[0])==digest(dec[1]),records=len(dec[0]),sha256=digest(dec[0]))
    first={}
    for key in ['obs','raw']:
        first[key]=next((pair[0]['control'][5*i]['time'] for i,(a,b) in enumerate(zip(pair[0]['decisions'],pair[1]['decisions'])) if a[key]!=b[key]),None)
    return dict(cut_time=cut,checks=checks,first_highlevel_input_output_difference_time=first,scope='Full physical rows before demand change; command/prestate before change; post-state metadata only if the full 20ms substep ends before change. No physics modified.')
def analyze(ep):
    rr=ep['physics'];c=ep['condition'];dt=.005
    t=np.array([r['force_time'] for r in rr]);active=np.array([r['active'] for r in rr],bool)
    root=np.array([r['root'] for r in rr]);feet=np.array([r['feet_xyz'] for r in rr]);loadn=np.array([r['foot_normal_force_n'] for r in rr])
    forces=np.array([[q['force'] for q in r['contacts']] for r in rr]);errs=np.array([[np.array(q['anchor'])-q['point'] for q in r['contacts']] for r in rr])
    force_norm=np.linalg.norm(forces,axis=2);burden=force_norm.sum(1)
    wrist=np.array([r['wrist'] if c['geometry']=='wrist' else r['body_positions']['right_wrist_roll_link'] for r in rr]);shoulder=np.array([r['shoulder_xyz'] for r in rr])
    torso=np.array([r['body_positions']['torso_link'] for r in rr]);T=np.array([r['torso_matrix'] for r in rr]);relative_torso=np.einsum('tji,tj->ti',T,wrist-torso)
    dq=np.array([r['qvel'][6:] for r in rr]);torque=np.array([r['torque'] for r in rr]);q=np.array([r['qpos'][7:] for r in rr])
    def window(mask,tracking):
        ix=np.flatnonzero(mask)
        if not len(ix):return None
        a,b=ix[0],ix[-1];ff=forces[mask];norm=force_norm[mask];ee=errs[mask]
        caps=np.array([point['cap'] for point in rr[a]['contacts']])
        out=dict(duration_s=len(ix)*dt,sum_force_norm_rms=float(np.sqrt(np.mean(burden[mask]**2))),net_force_xyz_rms=np.sqrt(np.mean(ff.sum(1)**2,0)).tolist(),sum_total_cap_fraction=float(np.mean(burden[mask]>=c['total_cap']-.001)),any_point_cap_fraction=float(np.mean(np.any(norm>=caps[None]-.001,axis=1))),
          root_delta_cm=(100*(root[b]-root[a])).tolist(),root_path_cm=100*float(np.linalg.norm(np.diff(root[ix,:2],axis=0),axis=1).sum()),wrist_world_delta_cm=(100*(wrist[b]-wrist[a])).tolist(),wrist_relative_shoulder_cm=(100*((wrist-shoulder)[b]-(wrist-shoulder)[a])).tolist(),wrist_relative_torso_cm=(100*(relative_torso[b]-relative_torso[a])).tolist(),
          feet_xy_path_cm=(100*np.linalg.norm(np.diff(feet[ix,:,:2],axis=0),axis=2).sum(0)).tolist(),min_root_height_m=float(root[mask,2].min()),max_tilt_deg=max(rr[j]['tilt_deg'] for j in ix),root_peak_speed_mps=max(float(np.linalg.norm(rr[j]['qvel'][:3])) for j in ix),final_root_speed_mps=float(np.linalg.norm(rr[b]['qvel'][:3])),
          no_support_flag_ticks=sum(not any(rr[j]['support']) for j in ix),no_foot_load_over5N_ticks=int(np.sum(np.max(loadn[mask],axis=1)<5)),max_contact_tangent_speed_mps=max(rr[j]['contact_tangent_speed_max_mps'] for j in ix),contact_tangent_speed_integral_m=float(sum(rr[j]['contact_tangent_speed_max_mps'] for j in ix)*dt),
          max_joint_limit_excess_rad=max(rr[j]['joint_limit_excess_max_rad'] for j in ix),peak_torque_saturation_fraction=max(rr[j]['torque_saturation'] for j in ix),mechanical_abs_work_proxy_J=float(np.sum(abs(torque[mask]*dq[mask]))*dt),external_work_J=float(sum(rr[j]['external_power'] for j in ix)*dt),yaw_moment_rms_Nm=float(np.sqrt(np.mean([rr[j]['net_yaw_moment_com']**2 for j in ix]))),points=[])
        if tracking:out['all_point_error_rms_cm']=100*float(np.sqrt(np.mean(np.sum(ee*ee,axis=2))))
        for k in range(ff.shape[1]):
            z=dict(link=rr[a]['contacts'][k]['link'],local=rr[a]['contacts'][k]['local'],cap_N=float(caps[k]),cap_fraction=float(np.mean(norm[:,k]>=caps[k]-.001)),force_xyz_rms=np.sqrt(np.mean(ff[:,k]**2,0)).tolist(),force_xyz_mean=ff[:,k].mean(0).tolist(),force_rms=float(np.sqrt(np.mean(norm[:,k]**2))),force_peak=float(norm[:,k].max()))
            if tracking:z.update(error_xyz_mean_cm=(100*ee[:,k].mean(0)).tolist(),error_xyz_rms_cm=(100*np.sqrt(np.mean(ee[:,k]**2,0))).tolist(),error_rms_cm=100*float(np.sqrt(np.mean(np.sum(ee[:,k]**2,1)))))
            out['points'].append(z)
        return out
    m=dict(pre_contact=(t<c['onset']),active=active,moving=active&(t<c['onset']+c['ramp']),hold=active&(t>=c['onset']+c['ramp']),recovery=(t>=c['release']-1e-8)&~active)
    if 'stop_time' in c:m.update(moving=active&(t<c['stop_time']),decelerating=active&(t>=c['stop_time'])&(t<c['stop_time']+c['deceleration']),hold=active&(t>=c['stop_time']+c['deceleration']))
    if not c['enabled']:m['no_contact_window']=(t>=c['onset'])&(t<c['release'])
    out={k:window(v,k in ['active','moving','decelerating','hold']) for k,v in m.items()}
    def first(mask,duration):
        start=None
        for i,ok in enumerate(mask):
            if ok and start is None:start=i
            if not ok:start=None
            if start is not None and (i-start+1)*dt>=duration:return float(rr[start]['time'])
        return None
    begin=int(np.flatnonzero(t>=c['onset']-1e-8)[0]);contact_window=(t>=c['onset'])&(t<c['release'])
    events={'root_vx_over005_200ms':first(contact_window&(np.array([r['qvel'][0] for r in rr])>.05),.2),'first_cap_50ms':first(active&(burden>=c['total_cap']-.001),.05)}
    for i,side in enumerate(['left','right']):events[side+'_unload_below5N_60ms']=first(contact_window&(loadn[:,i]<5),.06);events[side+'_lift_over1cm_50ms']=first(contact_window&(feet[:,i,2]-feet[begin,i,2]>.01),.05)
    controls=ep['control'];cp=np.array([np.asarray(r['q_PD']).reshape(-1) for r in controls]);delta=np.degrees(abs(np.diff(cp,axis=0)));j=np.unravel_index(delta.argmax(),delta.shape)
    out.update(events=events,quality=dict(qPD_20ms_jump_max_deg=float(delta.max()),qPD_jump_p99_deg=float(np.quantile(delta,.99)),qPD_peak_joint_index=int(j[1]),qPD_peak_time=controls[j[0]+1]['time'],actual_q_5ms_jump_max_deg=float(np.degrees(abs(np.diff(q,axis=0))).max()),actual_dq_max_radps=float(abs(dq).max()),qPD_actual_q_error_rms_deg=float(np.sqrt(np.mean([(np.degrees(np.asarray(r['q_PD']).reshape(-1)-np.asarray(r['pre_qpos'])[7:])**2).mean() for r in controls])))))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('output',type=Path);a=p.parse_args();beg=time.monotonic();d=a.directory
    index=json.loads((d/'index.json').read_text());result={'manifest':json.loads((d/'manifest.json').read_text()),'models':{},'common_prefix':{},'definition':'3D error/XYZ force use same-force-time contact.point/anchor/force. Kinematics use saved post-step actual state. Multi-point burden uses sum of point norms; released anchor is not a task metric.'}
    for name,r in index.items():
        if r.get('reused_actor'):continue
        rows=json.loads((d/name/'summary.json').read_text());out={}
        for row in rows:
            ep=load(d/name/(row['label']+'.json.gz'));out[row['label']]={'summary':row,'physical':analyze(ep)}
        result['models'][name]=out
        pair=[load(d/name/(s+'.json.gz')) for s in ['new_prefix_stop','new_prefix_continue']]
        result['common_prefix'][name]=common_prefix(pair)
        a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print('FINAL_ANALYSIS_MODEL',name,len(out),flush=True)
    result['analysis_seconds']=time.monotonic()-beg;a.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print('FINAL_ANALYSIS_COMPLETE',result['analysis_seconds'],flush=True)
if __name__=='__main__':main()
