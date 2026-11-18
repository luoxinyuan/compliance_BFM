"""Mechanical common-N resource decision from actual S0 blocks.

Output is a conservative engineering projection, never evidence of learning.
No timing or quantity is taken from unrelated PPO or empty forward passes.
"""
import json
from pathlib import Path
import time
import numpy as np
from stage_storage import footprint


def retained_counts(n):
    permanent={0, 9216, n//4, n//2, 3*n//4, n}
    committed=[]
    seen=set()
    peak=0
    for rows in range(0, n+1, 1024):
        # Previous latesttwo survive while the new pending pack is written.
        pending=set(committed[-2:])|seen|{rows}
        peak=max(peak,len(pending))
        committed.append(rows)
        if rows in permanent:
            seen.add(rows)
    return dict(permanent=sorted(permanent), final=len(permanent|set(committed[-2:])), peak=peak)


def project(stage, output):
    stage, output=Path(stage),Path(output)
    assert not output.exists()
    run=stage/'runs/s0-round-probe-01'
    blocks=json.loads((run/'blocks.json').read_text())
    assert len(blocks)==10 and blocks[-1]['collected']==10240
    assert [x['actual_update_count'] for x in blocks]==[0]*7+[64,8,8]
    joint=[x for x in blocks if x['collected']>8192]
    collect=float(np.percentile([b['collect_s'] for b in blocks],90))
    update=max(max(b['update_s']) for b in blocks if b['update_s'])
    save=max(b['save']['save_verify_s'] for b in joint)
    overhead=max(b['total_s']-b['collect_s']-sum(b['update_s'])-b['save']['save_verify_s'] for b in joint)
    cold=max(120., json.loads((run/'cost.json').read_text())['wall_s']-sum(b['total_s'] for b in blocks))
    warm_full=next(b['save']['newly_written_blob_bytes'] for b in blocks if b['collected']==8192)
    delta=max(b['save']['newly_written_blob_bytes'] for b in joint)
    retained=stage/'runs/s0-production-reference-01/actual-retention.json'
    if retained.exists():
        delta=max(delta,json.loads(retained.read_text())['save']['newly_written_blob_bytes'])
    rowbytes=sum(p.stat().st_size for p in run.glob('actual-rows-*.pt'))/10240
    disk=footprint(stage)
    lock=json.loads((stage/'locks/s0-closeout-plus90m-signed-v1.json').read_text())
    elapsed=time.time()-lock['start_unix']+300.
    stage_now=lock['prior_charged_s']+elapsed
    candidates=[]
    for n in [131072,262144,524288,1048576]:
        rounds=n//1024
        arm=(cold+rounds*(collect+8*update+save+overhead))*1.25
        counts=retained_counts(n)
        # Sequential arms: completedU remains fully retained while T writes.
        packs=counts['final']+counts['peak']
        # Deliberately no unverified cross-arm/future-byte dedup savings.
        checkpoint_expected=2*warm_full+(packs-2)*delta
        checkpoint_conservative=packs*max(warm_full,delta)
        data=2*n*rowbytes
        evaluation=8*2**30
        expected=disk['logical_bytes']+checkpoint_expected+data+evaluation
        bound=disk['logical_bytes']+1.25*(checkpoint_conservative+data+evaluation)
        total_job=stage_now+2*arm+6*3600+2*3600+2*3600
        candidates.append(dict(N_per_arm=n, rounds_per_arm=rounds,
            native_update_groups_per_arm=8*rounds, actor_updates_per_arm=8*(n-8192)//1024,
            conservative_arm_job_s=arm, passes12h_per_arm=arm<=43200,
            projected_stage_job_s=total_job, passes40h_stage=total_job<=144000,
            retained_counts_per_arm=counts, simultaneous_full_logical_packs=packs,
            expected_output_bytes=expected, conservative_output_bytes=bound,
            passes120GiB_output=bound<=120*2**30,
            replay_finite_capacity_compatible=n<1000000,
            physical_training_ticks_two_arms=8*n))
    feasible=[x for x in candidates if x['passes12h_per_arm'] and x['passes40h_stage'] and x['passes120GiB_output'] and x['replay_finite_capacity_compatible']]
    time_only=[x for x in candidates if x['passes12h_per_arm'] and x['passes40h_stage']]
    result=dict(status='COMMON_N_RESOURCE_LOCK' if feasible else 'T0_REPORT_RESOURCE_BLOCKED',
        N=max(x['N_per_arm'] for x in feasible) if feasible else None,
        time_only_N=max(x['N_per_arm'] for x in time_only) if time_only else None,
        measured=dict(collect_p90_s_per1024=collect, update_max_s_per_group=update,
            complete_save_verify_max_s=save, other_round_overhead_max_s=overhead,
            cold_and_postloop_reserve_s=cold, measured_first64_update_s=sum(blocks[7]['update_s']),
            bytes_per_real_row=rowbytes, first_full_pack_bytes=warm_full, observed_delta_pack_bytes=delta,
            reset_s='UNKNOWN separately; already INCLUDED in actual collect timing, never added twice',
            full_learner_readback='Included in save_verify_s', safety_factor=1.25),
        stage_storage=disk, stage_job_s_at_projection=stage_now, candidates=candidates,
        actual_formal_training_rows=0, no_future_learning_result_read=True,
        checkpoint_policy='step0 + firstActorupdate + Nquarters/end + latesttwo; pending write included in peak',
        projection_boundary='Only2 real joint blocks; first formal16k would require remeasurement. Projection is not measured future cost.',
        evaluation_output_reserve_bytes=8*2**30,
        evaluation_output_reserve_is_allocation_not_measurement=True,
        actual_physical_evaluation_complete=False,
        possible_resource_decision='Output cap change requires explicit Yi approval; no deletion of old failures or omission of Adam/replay allowed')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
