"""Recompute saved results and assert diagnostic/protocol integrity; no model."""
import argparse,hashlib,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import right_wrist_replay_minusx as p


def verify(out):
    result=json.loads((out/'result.json').read_text())
    episodes={n:json.loads((out/(n+'.json')).read_text()) for n in p.CONDITIONS}
    plus=json.loads((out/'plus_x_reference.json').read_text())
    assert hashlib.sha256((out/'plus_x_reference.json').read_bytes()).hexdigest()==p.PLUS_SHA
    assert result['scientific_outcome'] in ['no_force_replay_and_minus_x_completed','minus_x_negative']
    assert p.require_replay_gate(episodes['no_force'],episodes['no_force_replay'])==result['replay_gate']
    assert result['archived_no_force_gate']['passed']
    errors=[]
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):compare(x,y)
        elif isinstance(a,float):
            errors.append(abs(a-b));assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
        else:assert a==b
    compare(p.analyze_all(episodes,plus),result['analysis'])
    no_force=episodes['no_force'];origin=p.select_anchor(no_force)
    assert origin==result['anchor_source']['origin']
    counts={}
    for name,e in list(episodes.items())+[('plus_x_archived',plus)]:
        assert e['initial']==no_force['initial']
        assert e['trajectory']==[origin,[-.03 if name=='minus_x' else .03,0.,0.]]
        assert len(e['actor'])*4==len(e['physics'])
        assert all(a['time']==i*.02 and a['source_index']==i for i,a in enumerate(e['actor']))
        assert all(abs(row['time']-(i+1)*.005)<1e-10 for i,row in enumerate(e['physics']))
        assert all(a['z']==result['latent'] for a in e['actor'])
        assert all(p.norm(row['force'])<=5.+1e-10 for row in e['physics'])
        assert all(not any(row['force']) for row in e['physics'] if not row['active'])
        assert all(p.norm([a-b for a,b in zip(row['force'],row['audit_force'])])<1e-9 for row in e['physics'])
        counts[name]=sum(a['bfm_evaluated'] for a in e['actor'])
        if e['metrics']['fall_rate']:
            assert name=='minus_x' and result['scientific_outcome']=='minus_x_negative'
            assert any(row['failed'] for row in e['physics'])
        else:assert len(e['physics'])==1200 and len(e['actor'])==300
        if name in ['minus_x','plus_x_archived']:
            assert p.comparison(no_force,e)['matching_initial_and_prefix']
            first=next(r for r in e['physics'] if r['active'])
            assert p.norm(first['position_error_before'])<1e-10 and p.norm(first['spring_force'])<1e-8
            assert all(1.-1e-10<=r['force_time']<4.5+1e-10 for r in e['physics'] if r['active'])
    assert counts['no_force']==300 and counts['no_force_replay']==0 and counts['plus_x_archived']==300
    assert counts['minus_x']==len(episodes['minus_x']['actor'])
    assert not result['cuda_initialized'] and result['all_parameters_cpu'] and result['all_parameters_frozen']
    assert result['no_parameter_gradients'] and result['model_eval']
    assert 'torch' not in sys.modules and 'mujoco' not in sys.modules
    return dict(verification='pass',replay_gate=result['replay_gate'],live_actor_counts=counts,
        scientific_outcome=result['scientific_outcome'],recomputed_max_abs_error=max(errors),tolerance=1e-12,
        plus_x_reference_sha256=p.PLUS_SHA,first_scientific_boundary=result['first_boundary'])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    print(json.dumps(verify(parser.parse_args().output),indent=2))
