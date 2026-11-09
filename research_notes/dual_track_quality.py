"""Physical phase metrics, real source handoff, actual loaded contacts when saved."""
import math
import numpy as np
from native_transfer import metrics


def quality(task):
    s=task.spec;stop=s.get('stop_time',s['onset']+s['ramp']);hold=stop+s.get('deceleration',0)
    spans=[('precontact',s['handoff_time'],s['onset']),('startup',s['onset'],min(s['onset']+1,stop)),
        ('sustained',min(s['onset']+1,stop),stop),('deceleration',stop,hold),
        ('holding',hold,s['release']),('released',s['release'],s['end'])]
    out={}
    for name,a,b in spans:
        rows=[r for r in task.physics if a-1e-8<=r['force_time']<b-1e-8]
        controls=[r for r in task.controls if a-1e-8<=r['time']<b-1e-8]
        if not rows:continue
        m=metrics(rows,controls);slip=[r['stage_loaded_slip_mps'] for r in rows]
        yaw=[math.degrees(math.atan2(r['pelvis_matrix'][1][0],r['pelvis_matrix'][0][0])) for r in rows]
        active=[r for r in rows if r['active']]
        err=np.array([np.asarray(r['contacts'][0]['anchor'])-r['contacts'][0]['point'] for r in active])
        m.update(start=a,end=b,contact_physics=len(active),slip_peak_mps=float(max(slip)),slip_p95_mps=float(np.quantile(slip,.95)),
            yaw_deg_initial=yaw[0],yaw_deg_final=yaw[-1],
            error_xyz_rms_cm=(100*np.sqrt(np.mean(err**2,axis=0))).tolist() if len(active) else None,
            error_xyz_signed_cm=(100*np.mean(err,axis=0)).tolist() if len(active) else None,
            loaded_slip_definition='actual ground contact with normal force>1e-6N; per-contact speed before wrench removal',
            legacy_contact_slip_proxy_peak=max(r['contact_tangent_speed_max_mps'] for r in rows))
        out[name]=m
    return out
