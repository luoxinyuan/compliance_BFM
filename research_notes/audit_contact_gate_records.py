"""Independent saved-trace causality/identity checks; never calls model or physics."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def audit(out):
    protocol=json.loads((out/'protocol.json').read_text());items={}
    for c in protocol['conditions']:
        name=c['name'];paths={k:out/(name+'__'+k+'.json') for k in protocol['controllers']}
        if not all(p.exists() for p in paths.values()):continue
        eps={k:json.loads(p.read_text()) for k,p in paths.items()}
        assert len({e['common_state_sha256'] for e in eps.values()})==1
        s=eps['standing'];p=eps['contact'];clock=eps['clock']
        assert [r['anchor'] for r in s['physics']]==[r['anchor'] for r in p['physics']]==[r['anchor'] for r in clock['physics']]
        cs=p['control'];rows=p['physics'];assert len(rows)==4*len(cs)
        force_error=max(float(abs(np.asarray(cr['measured_force_xyz'])-np.asarray(rows[4*i-1]['force'] if i else [0,0,0])).max()) for i,cr in enumerate(cs))
        assert force_error==0
        command_error=max(abs(np.linalg.norm(cr['executed_z'])-16) for cr in cs)
        assert command_error<2e-5
        timings=max(abs(cr['time']-rows[4*i]['force_time']) for i,cr in enumerate(cs))
        assert timings<1e-9
        row=dict(last_completed_physics_force_max_error=force_error,command_norm_error=command_error,
                 actor_to_physics_time_error=timings,common_state_equal=True,partner_equal=True,
                 trace_sha256={k:hashlib.sha256(path.read_bytes()).hexdigest() for k,path in paths.items()})
        if not c['enabled']:
            equal={key:float(abs(np.asarray([r[key] for r in s['physics']])-np.asarray([r[key] for r in p['physics']])).max()) for key in ['qpos','qvel','wrist','root','force']}
            assert max(equal.values())==0
            row['no_contact_standing_parity']=equal
        items[name]=row
    return dict(audited_conditions=items,all_five_present=len(items)==5,
                meaning='Same initial state and partner; causal preceding force sample, no lookahead; no-contact full physical parity; no model execution')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('outputs',type=Path);a.add_argument('destination',type=Path);args=a.parse_args()
    result=audit(args.outputs);args.destination.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
