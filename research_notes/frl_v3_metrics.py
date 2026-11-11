"""Three independent reports: hard physics, unchanged relative guard, absolute quality."""
import math
from collections import defaultdict
import numpy as np
from frl_v3_risk import h

def absolute(task):
    ph=task.physics;cs=task.controls;severity=0.;events=0
    for i,c in enumerate(cs):
        end=cs[i+1]['time'] if i+1<len(cs) else c['time']+c['actual_duration_s']+1e-8
        rr=[r for r in ph if c['time']-1e-8<=r['force_time']<end-1e-8];v=max(r['stage_loaded_slip_mps'] for r in rr);severity+=c['actual_duration_s']*h(max(v-.45,0)/.45);events+=v>.45
    moving=[r for r in ph if r['active'] and r['force_time']<task.spec.get('stop_time',task.spec['onset']+task.spec['ramp'])]
    progress=moving[-1]['wrist'][0]-moving[0]['wrist'][0] if len(moving)>1 else 0.
    yaw=abs(math.degrees(math.atan2(ph[-1]['pelvis_matrix'][1][0],ph[-1]['pelvis_matrix'][0][0])))
    return dict(any_slip_over_045=any(r['stage_loaded_slip_mps']>.45 for r in ph),risk_severity_seconds=severity,events=events,loaded_slip_peak=max(r['stage_loaded_slip_mps'] for r in ph),loaded_slip_p90=float(np.quantile([r['stage_loaded_slip_mps'] for r in ph],.9)),abs_final_yaw_deg=yaw,moving_wrist_dx_m=float(progress),physics_boundary=task.boundary,normal_threshold_N=1e-6)

def summarize(rows):
    by=defaultdict(list)
    for r in rows:
        if r.get('source_failure'):continue
        by[r['parent_id']].append(r)
    p=[]
    for parent,rr in by.items():
        p.append(dict(parent_id=parent,family=rr[0]['family'],any_slip=any(r['absolute']['any_slip_over_045'] for r in rr),severity=float(np.mean([r['absolute']['risk_severity_seconds'] for r in rr])),abs_yaw=max(r['absolute']['abs_final_yaw_deg'] for r in rr),slip_peak=max(r['absolute']['loaded_slip_peak'] for r in rr)))
    main=[r for r in p if r['family'] in ('near','mid','stop')]
    fam={f:dict(parents=sum(r['family']==f for r in p),slip_fraction=float(np.mean([r['any_slip'] for r in p if r['family']==f])),severity=float(np.mean([r['severity'] for r in p if r['family']==f]))) for f in ('near','mid','stop','zero','none') if any(r['family']==f for r in p)}
    return dict(parents=len(p),primary_parents=len(main),exceeding_parent_fraction=float(np.mean([r['any_slip'] for r in main])),family_balanced_severity=float(np.mean([fam[f]['severity'] for f in ('near','mid','stop') if f in fam])),absolute_yaw_p90=float(np.quantile([r['abs_yaw'] for r in main],.9)),absolute_yaw_max=max(r['abs_yaw'] for r in main),slip_peak_p90=float(np.quantile([r['slip_peak'] for r in main],.9)),hard_boundary_cases=sum(bool(r.get('boundary')) for r in rows),family=fam,parent_rows=p)

def scientific_choice(rows,initial):
    old={(r['parent_id'],r['child']):r for r in initial};bad=[]
    for family in ('near','mid','stop'):
        rr=[r for r in rows if r['family']==family and not r.get('source_failure')];ee=[];ff=[];pp=[]
        for r in rr:
            b=old[(r['parent_id'],r['child'])];ee.append([r['all']['error_cm'],b['all']['error_cm']]);ff.append([r['all']['force_rms'],b['all']['force_rms']]);pp.append([r['absolute']['moving_wrist_dx_m'],b['absolute']['moving_wrist_dx_m']])
        if not rr:bad.append(family+'_source_unqualified');continue
        e,be=np.median(ee,axis=0);f,bf=np.median(ff,axis=0);p,bp=np.median(pp,axis=0)
        if e>be+max(2.,.2*be):bad.append(family+'_error')
        if f>bf+.75:bad.append(family+'_force')
        if bp>0 and p<.8*bp:bad.append(family+'_lost_progress')
    if any(r.get('boundary') for r in rows):bad.append('hard_boundary')
    q=summarize(rows);bq=summarize(initial);frac_gain=1-q['exceeding_parent_fraction']/bq['exceeding_parent_fraction'] if bq['exceeding_parent_fraction'] else 0.;sev_gain=1-q['family_balanced_severity']/bq['family_balanced_severity'] if bq['family_balanced_severity'] else 0.
    return dict(error_force_progress_admissible=not bad,rejections=bad,slip_fraction_reduction=frac_gain,severity_reduction=sev_gain,absolute_improvement_atleast25=bool(frac_gain>=.25 or sev_gain>=.25),absolute_yaw_over20=bool(q['absolute_yaw_max']>20),scope='absolute scientific coordinates; does not relax original relative guard or imply promotion')
