"""One specified risk event per executed20ms; no load-threshold or physics change."""
import math
import numpy as np
THRESHOLD=.45
LAMBDA=.33

def h(u):
    assert u>=0
    return u*u if u<=1 else 2*u-1

def event(rows,coefficient=LAMBDA):
    assert rows and coefficient>=0
    v=max(float(r['stage_loaded_slip_mps']) for r in rows)
    return coefficient*h(max(v-THRESHOLD,0.)/THRESHOLD)

def score(physics,controls,start_time):
    old=0.;risk=0.;terms={};events=[]
    for i,c in enumerate(controls):
        end=controls[i+1]['time'] if i+1<len(controls) else c['time']+c['actual_duration_s']+1e-8
        rows=[r for r in physics if c['time']-1e-8<=r['force_time']<end-1e-8];assert rows
        p=event(rows);discount=.99**((c['time']-start_time)/.1)
        old+=discount*(c['reward_integral']-c.get('absorbing_failure_penalty',0.));risk+=discount*p
        terms.update({k:terms.get(k,0)-discount*c['actual_duration_s']*v for k,v in c['reward_terms'].items()})
        v=max(r['stage_loaded_slip_mps'] for r in rows)
        if v>THRESHOLD:
            rr=max(rows,key=lambda r:r['stage_loaded_slip_mps'])
            contacts=rr['stage_foot_contacts'];con=max((r for r in contacts if r['normal_N']>1e-6),key=lambda r:r['tangent_speed_mps'])
            events.append(dict(time=rr['force_time'],v=v,normal_N=con['normal_N'],body=con['body'],event=p,all_foot_contacts=contacts,contact_details=rr.get('stage_foot_contact_details',[])))
    return dict(old_discounted_return=old,v3_discounted_return=old-risk,risk_discounted=risk,legacy_term_returns=terms,risk_events=events)

def metrics(physics,controls,spec):
    from native_transfer import metrics as original
    result=original(physics,controls);loaded=[r['stage_loaded_slip_mps'] for r in physics]
    active=[r for r in physics if r['active']];yaw=abs(math.degrees(math.atan2(physics[-1]['pelvis_matrix'][1][0],physics[-1]['pelvis_matrix'][0][0])))
    desired=float(active[-1]['anchor'][0]-active[0]['anchor'][0]) if active else 0.
    actual=float(active[-1]['wrist'][0]-active[0]['wrist'][0]) if active else 0.
    err=4 if spec['demand_family']=='near' else 8;force=4 if spec['demand_family']=='near' else 7
    stop=spec.get('stop_time',spec['onset']+spec['ramp']);hold=stop+spec.get('deceleration',0)
    holding=[r for r in active if r['force_time']>=hold-1e-8]
    hold_error=100*np.sqrt(np.mean([r['error']**2 for r in holding])) if holding else None
    hold_force=np.sqrt(np.mean([np.dot(r['force'],r['force']) for r in holding])) if holding else None
    good=result['error_cm'] is not None and result['error_cm']<=err and result['force_rms']<=force and max(loaded)<=.45 and yaw<=20
    if holding and spec['demand_family']=='stop':good=good and hold_error<=8 and hold_force<=6.5
    progress=actual>=desired-err/100
    result.update(loaded_slip_peak=max(loaded),loaded_slip_p90=float(np.quantile(loaded,.90)),yaw_abs_deg=yaw,partner_dx_m=desired,wrist_dx_m=actual,progress_pass=bool(progress),holding_error_cm=hold_error,holding_force_N=hold_force,low_slip_task_success=bool(good and progress))
    return result
