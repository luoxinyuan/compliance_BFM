"""Recompute and check recorded evidence without model/physics imports."""
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import right_wrist_pair as pilot

def verify(output):
    b,t,r=[json.loads((output/n).read_text()) for n in ['no_force.json','tether.json','result.json']]
    recomputed=pilot.analyze(b,t);errors=[]
    def compare(a,z):
        if isinstance(a,dict):
            assert a.keys()==z.keys()
            for k in a:compare(a[k],z[k])
        elif isinstance(a,list):
            assert len(a)==len(z)
            for c,d in zip(a,z):compare(c,d)
        elif isinstance(a,float):
            errors.append(abs(a-z));assert math.isclose(a,z,rel_tol=1e-12,abs_tol=1e-12),(a,z)
        else:assert a==z,(a,z)
    compare(recomputed,r['analysis'])
    bp,tp,ba,ta=b['physics'],t['physics'],b['actor'],t['actor']
    assert len(bp)==len(tp)==1200 and len(ba)==len(ta)==300
    assert all(v==0 for row in bp for v in row['force'])
    assert all(sum(v*v for v in row['force'])<=25+1e-10 for row in tp)
    assert all(not row['failed'] for row in bp+tp)
    assert b['initial']==t['initial'] and b['trajectory']==t['trajectory']
    assert all(row['z']==r['latent'] for row in ba+ta)
    assert not r['cuda_initialized'] and r['model_eval'] and r['all_parameters_frozen'] and r['all_parameters_cpu']
    active=[row for row in tp if row['active']]
    assert all(row['force']==[0,0,0] for row in tp if not row['active'])
    assert r['analysis']['pair_valid']
    first_action=next((a['time'] for a,z in zip(ta,ba) if a['action']!=z['action']),None)
    first_state=next((a['time'] for a,z in zip(tp,bp) if a['qpos']!=z['qpos']),None)
    return {'checks_passed':10,'analysis_recompute_abs_max_difference':max(errors),
        'floating_point_comparison_tolerance':1e-12,
        'first_nonzero_force_time_s':active[0]['force_time'],'last_active_force_time_s':active[-1]['force_time'],
        'first_state_difference_s':first_state,'first_actor_difference_s':first_action,
        'active_force_rms_n':math.sqrt(sum(sum(v*v for v in row['force']) for row in active)/len(active)),
        'active_force_mean_xyz_n':[sum(row['force'][i] for row in active)/len(active) for i in range(3)],
        'active_samples':len(active),'capped_samples':sum(pilot.norm(row['force'])>=5-1e-8 for row in active),
        'force_at_first_activation_xyz_n':active[0]['force'],
        'anchor_initial_xyz_m':t['trajectory'][0],'anchor_delta_m':t['trajectory'][1],
        'pre_contact_wrist_displacement_m':[active[0]['point_before'][i]-t['trajectory'][0][i] for i in range(3)],
        'all_post_activation_samples_double_contact':all(all(row['support']) for row in tp if row['time']>=1),
        'first_runtime_failure_boundary':None,
        'scientific_boundary':'3D preload from reset-fixed anchor; no causal active/passive ablation; one seed and 6 s'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    print(json.dumps(verify(parser.parse_args().output),indent=2))
