"""Counter-only IO batching: reserve4 ticks durably, retain every actual physics row.
No physics, policy, random draw, model, reward or checkpoint math is changed.
"""
import atexit
from pathlib import Path

def install(benchmark,quantum=4):
    assert quantum==4
    original=benchmark.write;state={'counts':None,'path':None,'reservation':0}
    def flush(final=False):
        if state['counts'] is None:return
        d=dict(state['counts']);d.update(counter_final=bool(final),counter_IO_quantum=quantum)
        if not final:d['reserved']=state['reservation']
        original(state['path'],d)
    def write(path,data):
        if Path(path).name!='physics-counter.json':return original(path,data)
        state['path']=path;state['counts']=data
        if data['actual']==data['reserved']:
            flush(True);return
        if data['reserved']>state['reservation']:
            state['reservation']=((data['reserved']+quantum-1)//quantum)*quantum;flush(False)
    benchmark.write=write
    atexit.register(lambda:flush(True))
    return dict(counter_IO_quantum=quantum,actual_step_tracking='same RAM counters, exact final flush; during abrupt death saved actual is lower bound and reserved is conservative upper',physics_trace_downsampled=False)

def self_test():
    import types
    records=[];b=types.SimpleNamespace(write=lambda p,v:records.append(dict(v)))
    install(b);c=dict(actual=0,reserved=0)
    for k in range(11):
        c['reserved']+=1;b.write(Path('physics-counter.json'),c)
        assert records[-1]['reserved']>=c['reserved'] and records[-1]['actual']<=c['actual']
        c['actual']+=1
    b.write(Path('physics-counter.json'),c)
    assert records[-1]['actual']==records[-1]['reserved']==11 and records[-1]['counter_final']
    return dict(status='COUNTER_ONLY_RESERVATION_FIXTURE_PASS',calls=11,durable_preallocated_writes=len(records)-1,final_exact=True,physics_or_policy_changed=False)
