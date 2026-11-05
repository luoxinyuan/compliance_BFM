"""Issue47: read saved physics/commands only. No torch, policy or simulator calls.

Decision records in clean_env are timestamped AFTER step(), while obs/raw were
captured BEFORE it. Align them by their sequential 5-control blocks, not by
subtracting blindly at a truncated terminal decision.
"""
import argparse
import csv
import gzip
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

P = Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
OLD = P / 'bridge-v2-owned/runs/feedback-coexistence-20261105'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def angle(a, b):
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def first_sustained(mask, times, seconds):
    start = None
    for ok, t in zip(mask, times):
        start = t if ok and start is None else start
        if not ok:
            start = None
        if start is not None and t - start >= seconds - .005 - 1e-8:
            return float(start)
    return None


def write_csv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def summarize(path, out, names, limits):
    ep = json.load(gzip.open(path, 'rt'))
    p, controls, dec = ep['physics'], ep['control'], ep['decisions']
    spec = ep['condition']; onset, release = spec['onset'], spec['release']
    ts = np.array([r['time'] for r in p]); ft = np.array([r['force_time'] for r in p])
    active = np.array([r['active'] for r in p]); ids = np.flatnonzero(active); begin = ids[0]
    root = np.array([r['root'] for r in p]); q = np.array([r['qpos'][7:] for r in p])
    dq = np.array([r['qvel'][6:] for r in p]); assert q.shape[1] == len(names) == dq.shape[1]
    feet = np.array([r['feet_xyz'] for r in p]); load = np.array([r['foot_normal_force_n'] for r in p])
    force = np.array([r['contacts'][0]['force'] for r in p])
    rawforce = np.array([r['contacts'][0]['raw_force'] for r in p])
    anchor = np.array([r['contacts'][0]['anchor'] for r in p])
    wrist = np.array([r['contacts'][0]['point'] for r in p])
    shoulder = np.array([r['shoulder_xyz'] for r in p]); err = anchor - wrist
    foot_speed = np.vstack([np.zeros((1, 2)), np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=2) / .005])
    foot_path = np.vstack([np.zeros((1, 2)), np.cumsum(np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=2), axis=0)])
    foot_path -= foot_path[begin]
    work = np.cumsum(np.array([r['external_power'] for r in p]) * .005)
    mech = np.cumsum(np.sum(np.abs(np.array([r['torque'] for r in p]) * dq), axis=1) * .005)
    cap = np.linalg.norm(force, axis=1) >= spec['total_cap'] - .001
    cap_time = np.cumsum(cap & active) * .005
    standing = controls[0]['prior_z']
    aligned, sub = [], []
    assert len(controls) <= 5 * len(dec) and len(controls) > 5 * (len(dec) - 1)
    for i, d in enumerate(dec):
        cc = controls[5*i:5*i+5]
        t = cc[0]['time']; end = cc[-1]['time'] + .02
        assert abs(d['time'] - end) < .02 + 1e-7
        raw = np.array(d['raw']).reshape(-1); z = 16 * raw / np.linalg.norm(raw)
        if len(cc)==5:
            assert np.max(np.abs(z - np.array(cc[-1]['executed_z']).reshape(-1))) < 1e-4
        # Last-z memory comes from the actual PRE-decision input.
        prevz = np.array(d['obs'])[565:821] * 16
        j = int(np.argmin(np.abs(ts - t)))
        r = p[j]; ri = max(0, j-1)
        margin = {n: dict(lower_deg=float(np.degrees(q[j,k]-limits[n][0])),
                          upper_deg=float(np.degrees(limits[n][1]-q[j,k])))
                  for k,n in enumerate(names) if k in [22,23,24,25]}
        row = dict(time=t,logged_post_time=d['time'],physics_time=ts[j],force_time=ft[j],
            partner_displacement_cm=100*float(anchor[j,0]-anchor[begin,0]),
            partner_velocity_xyz=r['contacts'][0]['anchor_velocity'],error_xyz_cm=(100*err[j]).tolist(),
            force_xyz=force[j].tolist(),uncapped_force_xyz=rawforce[j].tolist(),
            cumulative_cap_seconds=float(cap_time[j]),external_work_J=float(work[j]-work[begin]),
            mechanical_work_proxy_J=float(mech[j]-mech[begin]),
            root_delta_xyz_cm=(100*(root[j]-root[begin])).tolist(),root_velocity_xyz=r['qvel'][:3],
            wrist_delta_xyz_cm=(100*(wrist[j]-wrist[begin])).tolist(),
            wrist_relative_shoulder_delta_xyz_cm=(100*((wrist[j]-shoulder[j])-(wrist[begin]-shoulder[begin]))).tolist(),
            wrist_relative_torso_xyz=(np.array(r['torso_matrix']).T @ (wrist[j]-np.array(r['body_positions']['torso_link']))).tolist(),
            foot_load_n=load[j].tolist(),feet_xyz=feet[j].tolist(),foot_speed_xy=foot_speed[j].tolist(),
            foot_path_since_onset_cm=(100*foot_path[j]).tolist(),
            shoulder_elbow_margin_deg=margin,torso_matrix=r['torso_matrix'],pelvis_matrix=r['pelvis_matrix'],
            raw_norm=float(np.linalg.norm(raw)),z_vs_standing_deg=angle(z,standing),z_vs_previous_deg=angle(z,prevz),
            right_arm_q_deg=np.degrees(q[j,22:29]).tolist(),right_arm_dq=dq[j,22:29].tolist())
        aligned.append(row)
        for k,c in enumerate(cc):
            u = min(int(np.searchsorted(ts,c['time'])),len(ts)-1)
            sub.append(dict(time=c['time'],decision_start=t,substep=k+1,raw_norm=float(np.linalg.norm(raw)),
                z_vs_standing_deg=angle(c['executed_z'],standing),
                z_vs_decision_start_deg=angle(c['executed_z'],prevz),
                q_PD_deg=np.degrees(c['q_PD']).tolist(),actual_q_deg=np.degrees(q[u]).tolist(),
                actual_dq=dq[u].tolist(),q_PD_actual_error_deg=np.degrees(np.array(c['q_PD'])-q[u]).tolist(),
                q_PD_step_max_deg=float(np.degrees(c['q_PD_step_max_rad']))))
    events = dict(first_cap_50ms=first_sustained(active & cap,ft,.05),
        first_raw_demand_above_cap_50ms=first_sustained(active & (np.linalg.norm(rawforce,axis=1)>spec['total_cap']),ft,.05),
        root_speed_X_gt_005_200ms=first_sustained(active & (np.array([r['qvel'][0] for r in p])>.05),ts,.2),
        first_any_foot_path_2cm=next((float(t) for t,v in zip(ts,foot_path) if t>=onset and np.max(v)>.02),None))
    for side,k in [('left',0),('right',1)]:
        events[side+'_unload_below_5N_50ms']=first_sustained(active & (load[:,k]<5),ts,.05)
        events[side+'_unload_below_30N_50ms']=first_sustained(active & (load[:,k]<30),ts,.05)
        events[side+'_lift_1cm_speed_005_50ms']=first_sustained(active & (feet[:,k,2]-feet[begin,k,2]>.01) & (foot_speed[:,k]>.05),ts,.05)
    bouts=[]
    for side,k in [('left',0),('right',1)]:
        unloaded=active & (load[:,k]<5)
        edges=np.diff(np.r_[False,unloaded,False].astype(int))
        for lo,hi in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            if hi-lo<10:continue
            landing_time=first_sustained((np.arange(len(ts))>=hi)&(load[:,k]>=30),ts,.05)
            li=int(np.argmin(abs(ts-landing_time))) if landing_time is not None else min(hi,len(ts)-1)
            # Before unload, infer the preceding loaded foot position from 50ms.
            bi=max(begin,lo-10)
            bouts.append(dict(side=side,unload_time=float(ts[lo]),unloaded_duration_s=(hi-lo)*.005,
                loaded_landing_time=landing_time,
                foot_delta_xyz_cm=(100*(feet[li,k]-feet[bi,k])).tolist(),
                root_delta_xyz_cm=(100*(root[li]-root[bi])).tolist()))
    def point(t):
        j=int(np.argmin(abs(ts-t)));r=p[j]
        return dict(time=float(ts[j]),force_time=float(ft[j]),active=bool(active[j]),
            partner_displacement_cm=float((anchor[j,0]-anchor[begin,0])*100),
            error_xyz_cm=(100*err[j]).tolist() if active[j] else None,force_xyz=force[j].tolist(),raw_force_xyz=rawforce[j].tolist(),
            root_delta_cm=(100*(root[j]-root[begin])).tolist(),root_vx=float(r['qvel'][0]),
            wrist_relative_shoulder_delta_cm=(100*((wrist[j]-shoulder[j])-(wrist[begin]-shoulder[begin]))).tolist(),
            foot_load_n=load[j].tolist(),foot_xy_path_cm=(100*foot_path[j]).tolist(),
            foot_lift_cm=(100*(feet[j,:,2]-feet[begin,:,2])).tolist(),
            shoulder_elbow_q_deg=np.degrees(q[j,22:26]).tolist(),
            min_right_shoulder_elbow_joint_margin_deg=float(min(min(q[j,k]-limits[n][0],limits[n][1]-q[j,k]) for k,n in enumerate(names) if k in [22,23,24,25])*180/np.pi))
    keytimes=[onset,6,7.5,9,10.3,11,onset+spec['ramp'],release]
    stages={}
    middle=min(7.5,onset+spec['ramp'])
    for name,l,h in [('early',onset,middle),('late_move',middle,onset+spec['ramp']),('hold',onset+spec['ramp'],release),('recovery',release,spec['end'])]:
        mask=(ft>=l-1e-7)&(ft<h-1e-7)&active
        if name=='hold':mask=(ft>=l-1e-7)&active
        if name=='recovery':mask=(ft>=release-1e-7)&~active
        ix=np.flatnonzero(mask)
        if not len(ix):
            stages[name]=dict(seconds=0,scope='no such stage in this condition');continue
        a,b=ix[0],ix[-1]
        stages[name]=dict(seconds=len(ix)*.005,root_delta_cm=(100*(root[b]-root[a])).tolist(),
            error_rms_cm=float(100*np.sqrt(np.mean(np.sum(err[ix]**2,axis=1)))) if name!='recovery' else None,
            force_rms=float(np.sqrt(np.mean(np.sum(force[ix]**2,axis=1)))),cap_fraction=float(np.mean(cap[ix])),
            external_work_J=float(sum(p[i]['external_power']*.005 for i in ix)),
            mechanical_work_proxy_J=float(sum(np.sum(np.abs(np.array(p[i]['torque'])*dq[i]))*.005 for i in ix)))
    result=dict(label=ep['label'],path=str(path),sha256=sha(path),condition=spec,events=events,
        key_times=[point(t) for t in keytimes if t<=ts[-1]],stages=stages,foot_unload_bouts=sorted(bouts,key=lambda x:x['unload_time']),
        time_alignment='physics time is post-tick; force_time is load timestamp; decision logged time is POST decision; table time is true first 50Hz control start',
        limitations=['No BFM or policy inference performed; no same-state I counterfactual action.',
          'Joint scalar margins do not establish Cartesian reachability; no step onset single threshold.',
          'Different closed-loop trajectories are descriptive, not same-state causal interventions.'])
    stem=ep['label'];write_csv(out/(stem+'-100ms.csv'),aligned);write_csv(out/(stem+'-50Hz.csv'),sub)
    return result,dict(time=ts,onset=onset,anchor=anchor,wrist=wrist,root=root,force=force,load=load,feet=feet)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    start=time.monotonic();args.output.mkdir(parents=True,exist_ok=False)
    contract=json.loads((OLD/'final-reference/action-contract.json').read_text());names=contract['dof_names']
    xml=P/'repo/humanoidverse/data/robots/g1/g1_29dof_mujoco.xml'
    root=ET.parse(xml).getroot();joint={n.attrib['name']:n for n in root.iter('joint') if 'name' in n.attrib}
    limits={n:np.array([float(v) for v in joint[n].attrib['range'].split()]) for n in names}
    assert root.find('compiler').attrib.get('angle','degree')=='radian'
    models=json.loads((OLD/'final-frozen-models.json').read_text())['all']
    ids={k:models[k] for k in ['I','C0','seed4601_last']}
    for k,v in ids.items():assert sha(v['path'])==v['sha256'],k
    files=[OLD/'final-reference/C0__holdout_long72.json.gz',OLD/'final-reference/I__holdout_long72.json.gz',
           OLD/'final-learned/seed4601_last__holdout_long72.json.gz',OLD/'final-reference/C0__holdout_local6.json.gz']
    results=[];plot=[]
    for p in files:
        r,a=summarize(p,args.output,names,limits);results.append(r);plot.append((r['label'],a))
    result=dict(scope='saved-data numerical analysis only; no GPU, BFM or physics rollout',
        identities=ids,joint_limits_source=str(xml),joint_limits_sha256=sha(xml),rows=results)
    (args.output/'saved-timeline.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(4,1,figsize=(11,10),sharex=True)
    for label,a in plot[:3]:
        t=a['time'];axs[0].plot(t,100*(a['wrist'][:,0]-a['wrist'][0,0]),label=label)
        axs[1].plot(t,100*(a['root'][:,0]-a['root'][0,0]))
        axs[2].plot(t,np.linalg.norm(a['force'],axis=1))
    a=plot[0][1];axs[0].plot(a['time'],100*(a['anchor'][:,0]-a['anchor'][0,0]),'k--',label='C0 actual partner')
    for k in range(2):axs[3].plot(a['time'],a['load'][:,k],label=['C0 left foot','C0 right foot'][k])
    for ax,title in zip(axs,['Contact wrist X displacement (cm)','Root X displacement (cm)','Actual force norm (N)','C0 foot normal loads (N)']):
        ax.set_ylabel(title);ax.grid(alpha=.2)
        for t in [4.93,12.93,14.93]:ax.axvline(t,color='gray',ls=':',lw=.8)
    axs[0].legend(fontsize=8);axs[3].legend();axs[-1].set_xlabel('Actual physical episode time (s)')
    fig.tight_layout();fig.savefig(args.output/'saved-timeline.png',dpi=140);plt.close(fig)
    (args.output/'analysis-ledger.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,
        physics_rollouts=0,RL_decisions=0,policy_inferences=0,BFM_inferences=0,GPU_jobs=0),indent=2))
    print(json.dumps([{k:r[k] for k in ['label','events','key_times','stages']} for r in results],indent=2))


if __name__=='__main__':main()
