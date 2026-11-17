"""Read-only native-source rigid material transport, zero dynamics/learning.

Keeps solver-time contact poses distinct from post-integration poses. Each
actual material point is transported over the real next 5ms interval; the
per-foot cumulative statistic sums maxima, never contact-point set jumps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import mujoco
from source_evidence_readonly import read
from quality import stance_segments


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runtime-root',type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    root=a.runtime_root or Path(__file__).resolve().parents[2]
    m=mujoco.MjModel.from_xml_path(str(root/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'))
    d=mujoco.MjData(m)
    weight=float(m.body_mass.sum()*np.linalg.norm(m.opt.gravity))
    summary=json.loads((a.source/'progress.json').read_text())
    results=[]
    for info in summary:
        path=a.source/info['evidence_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==info['evidence_sha256']
        x=read(path)
        ph=x['physics']
        lengths=np.zeros((len(ph),2))
        speeds=np.zeros((len(ph),2))
        instantaneous=np.zeros((len(ph),2))
        contact_counts=np.zeros((len(ph),2),dtype=int)
        geometry_error=0.
        # Verify the actual source archive's contact-generation phase. These
        # poses are pre-integration; qpos is post-integration, not interchangeable.
        previous_qpos=x['states'][0]['mjdata_arrays']['qpos']
        for i,row in enumerate(ph):
            d.qpos[:]=previous_qpos
            mujoco.mj_kinematics(m,d)
            for c in row['ctft_material_contacts']:
                body=c['body']
                assert np.array_equal(d.xpos[body],np.asarray(c['body_world']))
                assert np.array_equal(d.xmat[body].reshape(3,3),np.asarray(c['body_rotation']))
                local=np.asarray(c['material_local'])
                actual=d.xpos[body]+d.xmat[body].reshape(3,3)@local
                geometry_error=max(geometry_error,float(np.max(abs(actual-np.asarray(c['contact_world'])))))
            d.qpos[:]=row['qpos']
            d.qvel[:]=row['qvel']
            mujoco.mj_kinematics(m,d)
            mujoco.mj_comPos(m,d)
            for c in row['ctft_material_contacts']:
                foot=c['foot'];body=c['body']
                local=np.asarray(c['material_local'])
                before=np.asarray(c['body_world'])+np.asarray(c['body_rotation'])@local
                after=d.xpos[body]+d.xmat[body].reshape(3,3)@local
                distance=float(np.linalg.norm((after-before)[:2]))
                lengths[i,foot]=max(lengths[i,foot],distance)
                speeds[i,foot]=max(speeds[i,foot],distance/.005)
                jac=np.zeros((3,m.nv))
                mujoco.mj_jac(m,d,jac,None,after,body)
                instantaneous[i,foot]=max(instantaneous[i,foot],float(np.linalg.norm((jac@d.qvel)[:2])))
                contact_counts[i,foot]+=1
            previous_qpos=row['qpos']
        segments=[]
        for seg in stance_segments([r['foot_normal_force_n'] for r in ph],weight):
            if not seg['qualified']:continue
            b,e,f=seg['start'],seg['end_exclusive'],seg['foot']
            assert (contact_counts[b:e,f]>0).all(),('Loaded stance has no actual contact material',info['case'],b,e,f)
            total=float(lengths[b:e,f].sum())
            p95=float(np.percentile(instantaneous[b:e,f],95))
            segments.append(dict(**seg,material_path_upper_envelope_m=total,
                instantaneous_post_velocity_p95_mps=p95,
                finite_transport_speed_p95_mps=float(np.percentile(speeds[b:e,f],95)),
                p95_within_005=p95<=.05,cumulative_envelope_within_002=total<=.02,
                full_segment_without_frame_clipping=True))
        results.append(dict(case=info['case'],ancestor=info['ancestor'],source_sha256=info['evidence_sha256'],
            complete=info['complete'],failure_tail_preserved=not info['complete'],
            contact_geometry_max_error_m=geometry_error,segments=segments,
            pose_phase_verified_exact=True,
            all_stances_within_bounds=all(s['p95_within_005'] and s['cumulative_envelope_within_002'] for s in segments)))
        (a.output/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
        del x
    result=dict(status='ARCHIVED_MATERIAL_TRANSPORT_MEASURED',sources=results,
        definition='For each actual solver-time contact body-local point, carry it through actual next5ms transform; max distance per foot/tick then sum the complete qualified stance, including its first/unloading interval. A conservative time-varying upper envelope, not a claim that one arbitrarily chosen point stays in contact throughout stance.',
        instantaneous_velocity='real post-integration qdot at the transported material point; world XY ground plane',
        comparison='same transport construction as existing F1/evaluation, now independently restricted to approved >=5%mg and >=.10s full stance segments',
        boundary='no point-count multiplication, no contact-set nearest-neighbour jumps, no force threshold change; old peak remains separate',
        new_physics_ticks=0,updates=0,wall_s=time.monotonic()-start)
    (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],ancestors=len(results),wall_s=result['wall_s'])),flush=True)


if __name__=='__main__':main()
