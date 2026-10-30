"""Saved-record qualification; no model execution or fitting."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
from contact_gate_summary import summarize

def analyze(out,protocol):
    contract=json.loads((out/'action-contract.json').read_text());episodes={};audits={}
    scale=np.array(contract['q_PD_scale']);offset=np.array(contract['q_PD_offset'])
    for path in sorted(out.glob('*__*.json')):
        if path.stem.endswith(('-parity','-state','-difference')):continue
        ep=json.loads(path.read_text())
        if 'physics' not in ep or 'control' not in ep:continue
        rows=ep['physics'];cs=ep['control'];label=ep['label']
        episodes[label]=summarize(ep,{})
        q=np.array([r['q_PD'] for r in cs]);a=np.array([r['action'] for r in cs])
        actual=np.array([r['force'] for r in rows]);wrench=np.array([r['applied_wrist_wrench'][:3] for r in rows])
        sensed=np.array([r['measured_force_xyz'] for r in cs]);expected=np.vstack([np.zeros(3),actual[3:-1:4]])
        zs=np.array([r['executed_z'][0] for r in cs]);same=np.array([r['local_q_PD'] for r in cs])-np.array([r['native_q_PD'] for r in cs])
        audits[label]=dict(qPD_recompute_max_rad=float(abs(q-(offset+np.clip(a*5,-contract['clip'],contract['clip'])*scale/5)).max()),
            applied_force_max_abs=float(abs(wrench-actual).max()),sensor_delay_max_abs=float(abs(expected-sensed).max()),
            common_state_sha256=ep['common_state_sha256'],z_norm_max_error=float(abs(np.linalg.norm(zs,axis=1)-16).max()),
            z_step_max_deg=max(c['z_step_deg'] for c in cs),z_angle_max_deg=max(c['z_angle_deg'] for c in cs),
            same_state_direct_skill_swap_qPD_max_deg=float(np.degrees(abs(same).max())),
            same_state_direct_skill_swap_qPD_rms_deg=float(np.degrees(np.sqrt((same**2).mean()))),
            local_decision_intervals=sorted(set(round(y['time']-x['time'],6) for x,y in zip(ep['decisions'],ep['decisions'][1:]))),
            control_count=len(cs),physics_count=len(rows),raw_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        assert audits[label]['qPD_recompute_max_rad']<1e-6 and audits[label]['applied_force_max_abs']<1e-10 and audits[label]['sensor_delay_max_abs']<1e-10
        assert audits[label]['local_decision_intervals']==[.1]
        # Real 3-frame local history must be the 3 last completed 50Hz forces.
        for i,d in enumerate(ep['decisions']):
            ix=i*5;last=[np.zeros(3) if j<0 else actual[4*j+3] for j in range(ix-3,ix)]
            assert np.max(np.abs(np.asarray(d['force_history'])-last))<1e-6
    if len(episodes)==6:
        def phase(task,ctl):return episodes[task+'__'+ctl]['phases']['all_contact']
        es={t:{c:phase(t,c)['tracking_rms_cm'] for c in ['standing','local','native']} for t in ['small','large']}
        gains={'small_local_over_native':1-es['small']['local']/es['small']['native'],'large_native_over_local':1-es['large']['native']/es['large']['local'],
               'small_local_over_standing':1-es['small']['local']/es['small']['standing'],'large_native_over_standing':1-es['large']['native']/es['large']['standing']}
        checks={k:v>=.05 for k,v in gains.items()}
        checks['winner_force_not_worse']=all(phase(t,c)['force_rms_n']<=1.05*phase(t,'standing')['force_rms_n'] for t,c in [('small','local'),('large','native')])
        checks['local_small_less_torso']=phase('small','local')['torso_rotation_max_deg']<phase('small','native')['torso_rotation_max_deg']
        checks['all_safe']=all(e['boundary'] is None and e['phases']['whole']['failed_samples']==0 and e['phases']['whole']['samples_without_either_foot_load']==0 and e['phases']['whole']['undesired_contacts_max']==0 and e['phases']['whole']['actual_joint_limit_excess_max_deg']==0 for e in episodes.values())
        checks['recovery']=all(e['root_speed_release_plus_2s_mps']<=.02 for e in episodes.values())
        checks['continuity']=all(e['phases']['whole']['qPD_step_max_deg']<=3 for e in episodes.values())
        checks['initial_equal']=len({a['common_state_sha256'] for a in audits.values()})==1
        return dict(episodes=episodes,audits=audits,gains=gains,checks=checks,complementarity_passed=all(checks.values()))
    return dict(episodes=episodes,audits=audits)

if __name__=='__main__':
    out=Path(sys.argv[1]);dest=Path(sys.argv[2]);protocol=json.loads((out/'protocol.json').read_text())
    result=analyze(out,protocol);dest.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ('gains','checks','complementarity_passed') if k in result},indent=2))
