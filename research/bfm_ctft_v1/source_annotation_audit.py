"""Archive-only source/transition and material-contact evidence audit.

No source split is created and no untested state is admitted for training.
Instantaneous material-point slip is reported separately from fixed-point
cumulative slip, which this audit does not claim to qualify.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from source_evidence_readonly import read
from quality import stance_segments


def runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    return [(int(a),int(b)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();results=[]
    summary=json.loads((a.source/'progress.json').read_text())
    for info in summary:
        path=a.source/info['evidence_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==info['evidence_sha256']
        x=read(path);ph=x['physics'];speed=np.linalg.norm(np.asarray([r['qvel'][:2] for r in ph]),axis=1)
        load=np.asarray([r['foot_normal_force_n'] for r in ph]);weight=35.112342*9.81
        # Identity independently matches the source/model qualification report.
        assert all(np.linalg.norm(r['force'])==0 for r in ph)
        support=load>=.05*weight
        labels=np.where(support[:,0]&~support[:,1],0,np.where(support[:,1]&~support[:,0],1,-1))
        prior=-1;switches=[]
        for i,label in enumerate(labels):
            if label<0:continue
            if prior>=0 and label!=prior:switches.append(i)
            prior=int(label)
        events=[]
        for begin,end in runs(speed>.15):
            if end-begin<40 or not (speed[:begin]<.05).any():continue
            valid=[s for s in switches if begin<=s<end]
            if valid:
                events.append(dict(kind='native_speed_build_up',tick=begin,time_s=ph[begin]['time'],
                    sustained_s=(end-begin)*.005,actual_support_switch_ticks=valid,
                    windows_s=[[-1.,-.1],[-.1,.3],[.3,1.]],contact_trigger_evidence=False))
        for begin,end in runs(speed<.05):
            if end-begin>=40 and any(b<=begin and e-b>=40 for b,e in runs(speed[:begin]>.15)):
                events.append(dict(kind='native_speed_reduction',tick=begin,time_s=ph[begin]['time'],sustained_s=(end-begin)*.005))
        stances=[];identity_error=0.;contact_rows=0
        for seg in stance_segments(load,weight):
            if not seg['qualified']:continue
            values=[]
            for row in ph[seg['start']:seg['end_exclusive']]:
                for c in row['ctft_material_contacts']:
                    if c['foot']!=seg['foot']:continue
                    point=np.asarray(c['body_rotation'])@np.asarray(c['material_local'])+np.asarray(c['body_world'])
                    identity_error=max(identity_error,float(np.max(np.abs(point-np.asarray(c['contact_world'])))))
                    contact_rows+=1;values.append(float(np.linalg.norm(c['tangent_velocity'])))
            stances.append(dict(**seg,contact_material_samples=len(values),instantaneous_material_slip_p95_mps=
                           float(np.percentile(values,95)) if values else None,
                           cumulative_fixed_material_point_slip='UNMEASURED',
                           scope='p95 of actual contact point instantaneous velocities, not fixed-point tracking qualification'))
        results.append(dict(case=info['case'],ancestor=info['ancestor'],source_sha256=info['evidence_sha256'],
            actual_controls=len(x['controls']),physical_rows=len(ph),complete=info['complete'],
            failed_tail_retained=not info['complete'],provisional_transition_events=events,
            actual_support_switch_count=len(switches),qualified_stance_segments=stances,
            material_local_to_world_max_abs_m=identity_error,material_contact_rows=contact_rows,
            admitted_new_training_states=0,split_status='NOT_CREATED',full_fixed_point_metric='PENDING'))
        (a.output/'progress.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    result=dict(status='ARCHIVE_ANNOTATION_EVIDENCE_ONLY',ancestors=len(results),sources=results,
          new_physics_ticks=0,updates=0,source_split_or_full_material_tracking_qualified=False,wall_s=time.monotonic()-started)
    (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sources'}),flush=True)


if __name__=='__main__':main()
