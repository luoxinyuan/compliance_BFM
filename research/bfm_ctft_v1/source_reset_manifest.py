"""Pre-training U/T reset domain and independent evaluation parents.

No model score, contact outcome, old sealed asset or search label is read.
Source validity is actual saved state, not a motion-name assertion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def create(stage,out):
    stage=Path(stage);out=Path(out);assert not out.exists()
    audit_path=stage/'runs/s0-source-admission-01/result.json'
    audit=json.loads(audit_path.read_text());train=audit['train']
    assert len(train)==12
    uniform=[];events=[]
    for r in train:
        uniform.extend(dict(case=r['case'],control=i) for i in r['valid_control_indices'])
        for e in r['actual_events']:
            if e['all_windows_nonempty']:
                events.append(dict(case=r['case'],event_time_s=e['time_s'],buckets=e['window_control_indices']))
    assert uniform and events
    selected=[]
    for r in audit['selected_static']:
        selected.append(dict(ancestor=r['ancestor'],case=r['case'],control=r['early_static']['control'],stratum='earlier_root_static',
            source_file=r['evidence_file'],source_sha256=r['source_sha256']))
    for r in audit['selected_transition']:
        selected.append(dict(ancestor=r['ancestor'],case=r['case'],control=r['transition_near']['control'],stratum='transition_near',
            source_file=r['evidence_file'],source_sha256=r['source_sha256']))
    # This is a source qualification manifest only. Final evaluation must also
    # include actual already-moving source points and qualify their restore.
    assert len(selected)==len({r['ancestor'] for r in selected})==24
    result=dict(protocol='CTFT_RESET_DOMAIN_V1',status='SOURCE_DOMAIN_LOCKED_EVAL_MOVING_SUBSTRATUM_PENDING',
        original_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        train_sources=[{k:r[k] for k in ['case','ancestor','evidence_file','source_sha256']} for r in train],
        uniform_states=uniform,transition_events=events,
        U_rule='Uniform over all legal20ms source states pooled across12 training ancestors.',
        T_rule='0.5 same uniform pool;0.5 uniform qualified actual event, then pre/at/post probabilities.5/.25/.25, then uniform legal state within the window.',
        source_phase_is_not_actor_input=True,transition_lineages=len({r['case'] for r in events}),
        transition_buckets=['pre','at','post'],transition_probabilities=[.5,.25,.25],
        initial_evaluation_source_qualification=selected,final_contact_eval_not_run=True,
        no_formal_sample_count_locked=True,no_learning_or_physical_steps=True)
    out.write_text(json.dumps(result,indent=2)+'\n');return result


def sample(lock,arm,random_values):
    u=np.asarray(random_values);assert u.shape==(4,) and ((u>=0)&(u<1)).all()
    if arm=='U' or u[0]<.5:
        return dict(lock['uniform_states'][int(u[1]*len(lock['uniform_states']))],reset_component='uniform')
    assert arm=='T'
    event=lock['transition_events'][int(u[1]*len(lock['transition_events']))]
    name=lock['transition_buckets'][int(np.searchsorted(np.cumsum(lock['transition_probabilities']),u[2],side='right'))]
    states=event['buckets'][name]
    return dict(case=event['case'],control=states[int(u[3]*len(states))],reset_component=name,event_time_s=event['event_time_s'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();x=create(a.stage,a.output)
    print(json.dumps(dict(uniform_states=len(x['uniform_states']),events=len(x['transition_events']),
        transition_lineages=x['transition_lineages'],train_lineages=len(x['train_sources']))))


if __name__=='__main__':main()
