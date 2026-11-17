"""Source ancestry and causal-state admission, no learning or new dynamics.

Training reserves the original12 walk ancestors. Candidate holdout ancestors
must be disjoint. Motion names never substitute for actual speed/support.
No final split is published from a partial qualification run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from source_evidence_readonly import read


def runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    return zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))


def audit(folder,weight):
    summary=json.loads((folder/'progress.json').read_text())
    results=[]
    for info in summary:
        path=folder/info['evidence_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==info['evidence_sha256']
        x=read(path);ph=x['physics'];controls=x['controls'];states=x['states']
        full=len(controls)-(1 if x['boundary'] is not None else 0)
        speed=np.linalg.norm(np.asarray([r['qvel'][:2] for r in ph]),axis=1)
        load=np.asarray([r['foot_normal_force_n'] for r in ph])>=.05*weight
        label=np.where(load[:,0]&~load[:,1],0,np.where(load[:,1]&~load[:,0],1,-1))
        previous=-1;switches=[]
        for j,side in enumerate(label):
            if side<0:continue
            if previous>=0 and side!=previous:switches.append(j)
            previous=int(side)
        valid=[]
        for i in range(10,min(full,len(states))):
            r=ph[i*4-1]
            legal=(not r['failed'] and r['height']>=.45 and r['tilt_deg']<=30.
                and r['joint_limit_excess_max_rad']<=.02 and r['undesired_ground_contacts']==0
                and not r['ctft_online_support']['stop'])
            if legal:valid.append(i)
        # Prior source restore protocol already fixed control50. Require the
        # current and previous200ms to be root-static, not merely low net drift.
        static=None;i=50
        if i in valid and i+12<full:
            end=i*4
            if (speed[end-40:end]<.05).all() and load[end-1].any():
                static=dict(control=i,time_s=controls[i]['time'],root_speed_mps=float(speed[end-1]),
                    identity='earlier_actual_root_static_source_not_claim_of_entire_motion_standing')
        events=[]
        for b,e in runs(speed>.15):
            if e-b<40 or not (speed[:b]<.05).any():continue
            support=[j for j in switches if b<=j<e]
            if not support:continue
            tstar=ph[b]['time']
            buckets={'pre':[],'at':[],'post':[]}
            for i in valid:
                delta=controls[i]['time']-tstar
                if -1.<=delta<-.1:buckets['pre'].append(i)
                elif -.1<=delta<.3:buckets['at'].append(i)
                elif .3<=delta<=1.:buckets['post'].append(i)
            events.append(dict(time_s=tstar,high_speed_duration_s=(e-b)*.005,
                actual_support_switch_ticks=support,window_control_indices=buckets,
                all_windows_nonempty=all(buckets.values())))
        moving=None
        for event in events:
            # Exactly the near-event boundary already in the source contract;
            # this is neither a new threshold nor a latent-distance success test.
            choices=[i for i in valid if i+12<full]
            if not choices:continue
            i=min(choices,key=lambda j:abs(controls[j]['time']-(event['time_s']-.1)))
            if abs(controls[i]['time']-(event['time_s']-.1))<=.0200001:
                moving=dict(control=i,time_s=controls[i]['time'],event_time_s=event['time_s'],
                    root_speed_mps=float(speed[i*4-1]),identity='actual_transition_near_source_not_always_already_walking')
                break
        results.append(dict(case=info['case'],ancestor=info['ancestor'],evidence_file=info['evidence_file'],
            source_sha256=info['evidence_sha256'],valid_control_indices=valid,valid_states=len(valid),
            early_static=static,transition_near=moving,actual_events=events,
            source_complete=info['complete'],original_failed_tail_retained=not info['complete'],
            admission_uses_only_actual_past_physical_state=True,
            source_is_not_an_expert_success_label=True))
        del x
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True)
    p.add_argument('--holdout',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runtime-root',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    import mujoco
    root=a.runtime_root or Path(__file__).resolve().parents[2]
    m=mujoco.MjModel.from_xml_path(str(root/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'))
    weight=float(m.body_mass.sum()*np.linalg.norm(m.opt.gravity))
    train=audit(a.train,weight);holdout=audit(a.holdout,weight)
    names={r['ancestor'] for r in train}
    assert len(names)==12 and not names.intersection(r['ancestor'] for r in holdout)
    all_static=[r for r in holdout if r['early_static'] is not None]
    all_moving=[r for r in holdout if r['transition_near'] is not None]
    # A deterministic bipartite quota: use non-static moving candidates first,
    # preserving as many independent early-static ancestors as possible.
    moving=sorted(all_moving,key=lambda r:(r['early_static'] is not None,r['ancestor']))[:12]
    used={r['ancestor'] for r in moving}
    static=sorted((r for r in all_static if r['ancestor'] not in used),key=lambda r:r['ancestor'])[:12]
    summary=a.holdout/'source-summary.json'
    complete=summary.exists() and json.loads(summary.read_text())['status']=='HOLDOUT_CANDIDATES_RECORDED_ADMISSION_PENDING' and len(holdout)==28
    supported=complete and len(moving)==len(static)==12 and all(r['valid_states'] for r in train)
    result=dict(status='ANCESTRY_AND_RESET_SUPPORT_AVAILABLE_RESTORE_PENDING' if supported else 'DATA_SUPPORT_INSUFFICIENT_OR_INCOMPLETE',
        full_holdout_collection_complete=complete,train_lineages=len(names),holdout_recorded_lineages=len(holdout),
        eligible_holdout_static_lineages=len(all_static),eligible_holdout_transition_lineages=len(all_moving),
        selected_static=static,selected_transition=moving,train=train,holdout=holdout,
        scientific_success_not_evaluated=True,formal_split_locked=False,selected_state_restore='PENDING',
        no_training_or_new_physics=True,wall_s=time.monotonic()-start)
    (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['train','holdout','selected_static','selected_transition']}),flush=True)


if __name__=='__main__':main()
