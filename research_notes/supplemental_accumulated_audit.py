"""Post-run read-only trace diagnostics: error distribution and transition command attribution."""
import json,sys
from pathlib import Path
import numpy as np

def analyze(out):
    runs=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs')
    baseline=runs/'issue21-goalcalibration-20261030A/attempt-01/outputs'
    summary=json.loads((out/'accumulated-analysis.json').read_text());results={}
    for label,p in [('anchored',baseline),('accumulated',out)]:
        ep=json.loads((p/(label+'.json')).read_text())
        names=json.loads((p/'action-contract.json').read_text())['dof_names']
        hold=[r for r in ep['physics'] if r['active'] and 7<=r['force_time']<8]
        errors=np.array([r['error']*100 for r in hold])
        d=dict(hold_error_cm=dict(min=float(errors.min()),median=float(np.median(errors)),p95=float(np.percentile(errors,95)),max=float(errors.max()),endpoint=float(errors[-1]),fraction_lt1=float((errors<1).mean())))
        for key in ['waist_joint_delta_endpoint_deg','outside_initial_arm_sphere_fraction','actual_wrist_radius_from_initial_shoulder_max_cm']:
            d[key]=summary['results'][label]['hold'][key]
        controls=ep['control'];q=np.array([c['q_PD'] for c in controls]);step=np.diff(q,axis=0)
        times=np.array([c['tether_time'] for c in controls]);d['peak_steps']={}
        for phase in ['active','release_scheduled','after_observed_release']:
            if phase=='after_observed_release':indices=[i for i in range(1,len(q)) if not ep['physics'][4*i-1]['active']]
            else:
                lo,hi=(3,8) if phase=='active' else (8,9)
                indices=[i for i in range(1,len(q)) if lo-1e-8<=times[i]<hi-1e-8]
            ix,j=np.unravel_index(abs(step[np.array(indices)-1]).argmax(),(len(indices),29));i=indices[ix]
            v=dict(time=times[i],joint=names[j],signed_step_deg=float(np.degrees(step[i-1,j])))
            if label=='accumulated':
                v['goal_BFM_step_deg']=float(np.degrees(controls[i]['goal_q_PD'][j]-controls[i-1]['goal_q_PD'][j]))
                v['direct_step_deg']=float(np.degrees(controls[i]['correction_q_PD'][j-22]-controls[i-1]['correction_q_PD'][j-22])) if j>=22 else 0.
            d['peak_steps'][phase]=v
        if label=='accumulated':
            d['first_totalcap_time']=next((c['tether_time'] for c in controls if c['accumulation']['at_box']),None)
            d['release_zero_time']=next((c['tether_time'] for c in controls if c['accumulation']['release_start'] is not None and max(abs(np.array(c['correction_q_PD'])))<1e-10),None)
            d['hold_cap_fraction_by_joint']={names[22+j]:float(np.mean([abs(c['correction_q_PD'][j])>=.2-1e-8 for c in controls if 7-1e-8<=c['tether_time']<8-1e-8])) for j in range(7)}
            d['online_action_or_joint_projection_count']=sum(c['accumulation']['projected'] for c in controls)
        results[label]=d
    return results

if __name__=='__main__':
    out=Path(sys.argv[1]);r=analyze(out)
    (out/'supplemental-diagnostics.json').write_text(json.dumps(r,indent=2))
    print(json.dumps(r,indent=2))
