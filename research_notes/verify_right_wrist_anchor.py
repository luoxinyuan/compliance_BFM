"""Verify saved anchor/replay evidence, no model or simulation imports."""
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import right_wrist_anchor as p


def verify(root):
    episodes={n:json.loads((root/(n+'.json')).read_text()) for n in ['no_force','closed_loop','replay']}
    result=json.loads((root/'result.json').read_text())
    assert result['scientific_outcome'] in ('anchor_and_replay_completed','replay_negative')
    recomputed=p.analyze_all(episodes);errors=[]
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
    compare(recomputed,result['analysis'])
    no_force=episodes['no_force'];origin=p.select_anchor(no_force)
    assert origin==result['anchor_source']['origin']
    actor_counts={}
    for name,e in episodes.items():
        if name!='replay':assert len(e['physics'])==1200 and len(e['actor'])==300
        else:assert len(e['physics'])<=1200 and len(e['actor'])<=300
        if e['metrics']['fall_rate']:
            assert name=='replay' and result['scientific_outcome']=='replay_negative'
            assert any(row['failed'] for row in e['control_rows'])
        else:assert len(e['physics'])==1200
        assert e['initial']==no_force['initial']
        assert e['trajectory']==[origin,[.03,0.,0.]]
        assert all(a['z']==result['latent'] for a in e['actor'])
        if name!='replay':assert all(not row['failed'] for row in e['physics'])
        assert all(p.norm(row['force'])<=5.+1e-10 for row in e['physics'])
        assert all(row['force']==[0,0,0] for row in e['physics'] if not row['active'])
        assert all(p.norm([a-b for a,b in zip(row['force'],row['audit_force'])])<1e-9 for row in e['physics'])
        actor_counts[name]=sum(a['bfm_evaluated'] for a in e['actor'])
        if name!='no_force':
            first=next(r for r in e['physics'] if r['active'])
            assert p.norm(first['position_error_before'])<1e-10
            assert p.norm(first['spring_force'])<1e-8
            assert p.comparison(no_force,e)['matching_initial_and_prefix']
    assert actor_counts=={'no_force':300,'closed_loop':300,'replay':0}
    assert all(a['time']==b['time'] and a['action']==b['action'] for a,b in zip(no_force['actor'],episodes['replay']['actor']))
    assert any(a['action']!=b['action'] for a,b in zip(no_force['actor'],episodes['closed_loop']['actor']))
    assert any(a['torque']!=b['torque'] for a,b in zip(no_force['physics'],episodes['replay']['physics']))
    assert not result['cuda_initialized'] and result['all_parameters_frozen'] and result['all_parameters_cpu'] and result['no_parameter_gradients']
    return {'checks':'recorded data and negative outcomes verified; this is not a survival pass','recomputed_abs_max_difference':max(errors),
        'tolerance':1e-12,'live_actor_calls':actor_counts,'replayed_actions_exact':True,
        'same_initial_state_latent_and_anchor':True,'first_spring_position_force_zero':True,
        'actual_force_matches_original_law':True,'pd_torque_changes_despite_identical_replayed_actions':True}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    print(json.dumps(verify(parser.parse_args().output),indent=2))
