"""Metadata-only sampling/label contract; never part of Student observation."""
def target_fields(record, label):
    prefix = 'shadow_' if label == 'shadow' else ''
    if label not in ('teacher', 'shadow'): raise ValueError('Unknown teacher label source')
    return record[prefix+'alpha'], record[prefix+'action'], record[prefix+'q_PD']


def phase_indices(controls, condition):
    a,b=condition['disconnect_start'],condition['disconnect_end'];onset=condition['onset']
    result={k:[] for k in ('active_before','gap','reconnect1s','rest')}
    for i,row in enumerate(controls):
        t=row['time']
        k=('gap' if a-1e-8<=t<b-1e-8 else 'reconnect1s' if b-1e-8<=t<b+1-1e-8 else 'active_before' if onset-1e-8<=t<a-1e-8 else 'rest')
        result[k].append(i)
    assert all(result.values())
    return result
