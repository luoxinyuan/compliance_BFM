"""Approved S0 support/slip semantic qualification on real archived physics.

Reads only exposed original-native and historical failed physical trajectories.
No simulation, no policy loading, no training, no sealed access.
"""
import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import mujoco
from quality import OnlineSupport, stance_segments, verify_causal_guard_boundaries, DT

P = Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    begun=time.monotonic()
    xml=P/'repo/humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'
    m=mujoco.MjModel.from_xml_path(str(xml))
    weight=float(m.body_mass.sum()*np.linalg.norm(m.opt.gravity))
    native=P/'experiments/fb-native-audit-20261110/runs/native-inventory-01'
    failed=P/'experiments/interaction-pilot-b-diagnosis-20261112/runs'
    off=P/'experiments/interaction-pilot-adapter-off-20261112/runs'
    cases=[(n,native/(n+'-actual.json.gz'),'original_native',True) for n in
           ['standing','forward_motion','left_motion','right_motion']]
    cases += [(n,failed/n/'probe.json','historical_failure',False) for n in ['mean-B19','sample-B19']]
    cases += [('offmean-B19',off/'offmean-B19/probe.json','historical_comparator',False)]
    records=[]
    for name,path,role,require_no_stop in cases:
        raw=path.read_bytes();x=json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)
        rows=x['physics'];assert rows and all('foot_normal_force_n' in r for r in rows)
        times=np.array([r['time'] for r in rows],dtype=np.float64)
        assert np.allclose(np.diff(times),DT,atol=1e-10,rtol=0)
        normal=np.array([r['foot_normal_force_n'] for r in rows])
        monitor=OnlineSupport(weight);states=[monitor.update(f) for f in normal]
        stops=[i for i,r in enumerate(states) if r['stop']]
        seg=stance_segments(normal,weight)
        starts=[s for s in seg if s['start']>0]
        immediate=sum(states[s['start']]['loaded'][s['foot']] for s in starts)
        qualified_age=np.zeros_like(normal,dtype=bool)
        for s in seg:
            # Deliberately WRONG delayed online predicate, for contrast only.
            if s['qualified']:qualified_age[s['start']+19:s['end_exclusive'],s['foot']]=True
        wrong=OnlineSupport(100.)
        wrongstates=[wrong.update([5. if v else 0. for v in r]) for r in qualified_age]
        first=stops[0] if stops else None
        out=dict(name=name,role=role,path=str(path),file_sha256=hashlib.sha256(raw).hexdigest(),
            physics_rows=len(rows),archived_dt_s=DT,archived_boundary=x.get('boundary',x.get('error')),
            longest_no_support_s=monitor.maximum_no_support_ticks*DT,
            online_would_stop=bool(stops),first_stop_time_s=None if first is None else float(times[first]),
            first_stop_current_n=None if first is None else normal[first].tolist(),
            new_landing_events=len(starts),new_landing_immediate_support=int(immediate),
            offline_qualified_stance_segments=sum(s['qualified'] for s in seg),
            offline_short_load_segments=sum(not s['qualified'] for s in seg),
            wrongly_waiting_100ms_would_stop_ticks=sum(s['stop'] for s in wrongstates),
            online_and_offline_separate=True,
            trace_start_counter='0 conservatively; no claim to reconstruct pre-window past from missing rows',
            no_new_physics=True)
        records.append(out)
        if require_no_stop:assert not stops,('Native walking support qualification failed',out)
    negatives=[r for r in records if r['role']=='historical_failure' and r['online_would_stop']]
    assert negatives,'No actual archived loss-of-support negative found; fixture cannot replace it'
    result=dict(status='SUPPORT_SEMANTICS_QUALIFIED_ON_REAL_ARCHIVED_TRAJECTORIES',
        fixture_checks=verify_causal_guard_boundaries(),records=records,
        model_xml=str(xml),model_xml_sha256=digest(xml),total_mass_kg=float(m.body_mass.sum()),
        gravity_mps2=float(np.linalg.norm(m.opt.gravity)),weight_n=weight,
        effective_load_threshold_n=.05*weight,online_no_support_limit_s=.05,
        offline_stance_minimum_s=.10,
        no_material_point_slip_quality_pass_claim=True,
        evidence_boundary='Qualifies support timing only. Archived foot-force rows do not provide fixed material-point tracks; complete new stance-slip metric still pending.',
        real_loss_of_support_negative_cases=len(negatives),new_physics_ticks=0,
        model_weight_loads=0,updates=0,wall_s=time.monotonic()-begun)
    (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
