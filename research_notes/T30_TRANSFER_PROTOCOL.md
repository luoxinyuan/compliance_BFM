# Teacher30 transfer/recovery — frozen protocol 2026-11-03

User-authorized stage; Issue35. No RL. Total job-wall18000s/output10GiB/Student updates16000, <=3 idle GPUs. Plan uses14000 updates, not a requirement to spend all.

Models: exact teacher/r0 paths and SHA in t30_protocol.json. Teacher30 and oldPPO15 frozen separately; BFM frozen and original200Hz physics/50Hz BFM-PD/10Hz high-level slerp remain. All train/main eval5N, no new7.5/10N studies in this stage plan.

Sources: Teacher30 own physical rollouts normal/early/late/middle stop/local15cm/no-contact, combined with824 original PPO15 labels. Old labels have explicit different provenance; not relabeled Teacher30. No online M10/Cyclic controller.

S1warm from originalr0,4000 updates. Shared warm model then forks S1matched3000 vsS2visited3000 with same freshAdam/seed805/batch128/lr3e-4. Matched continuation batch32old+96new Teacher30, aggregation32old+64new+32visited; only replace32Teacher30-own samples with visited samples. N30 physically removes force300 inputs/normalizers/first-layer weights from originalr0 and trains4000 with same base data/seed as S1warm. No-force is725D; ideal1025D. No privileged anchor/velocity/time/active/taskID input. Ideal force is not proven hardware sensing.

Recovery: S0 five phases × early/late=10 full-prefix takeovers. After S1warm, each of five phases tests both endpoints of a .4s window,20 takeovers. Prefix reexecuted using exact Student, full BFM/actual action/motor state and own z history; qpos/qvel/force must match continuation exactly before takeover. Not partial-qpos reset. Teacher direct policy has no hidden recurrent phase to fabricate; query uses actual Student executed history.

Representative phases: entry6s, stop-.5, stop+.2, stop+decel+1, release+.2. A window is admitted only if both endpoint branches remain physically safe, tail forceRMS+3*3Derror_m beats continuation, and postrelease root path <=continuation+.03m. For released-only tail compare rootpath+wrist speed. These are local label-admission comparisons, not universal scientific success thresholds. Report component tradeoffs and rejected regions; accepted interval only, no whole-trajectory relabel. If none qualify, skip S2, retain failure and complete S1/N.

Dev: old normal, early7.2/.5/hold3.2, late9.2/.85/hold3.8, local15cm, no-contact. New sealed early(onset4.31, stop7.65,decel.9,hold3.1), late(4.16,9.65,.7,3.3), +6s postrelease. These do not enter fit or selection; fixed final update checkpoint, no model reselection. All conditions retain per-phaseXYZ, support, q_PD vs actual q, force and recovery.

Parallelization: A GPU1 recovery/Teacher collection/ideal transfer; C separateGPU5 only after declared immutable Teacher dataset. Quality audit existing saved trajectories can proceed without new physical sampling. If a physically meaningful command jump is identified, at most one predeclared 0.1s high-level causal command filter comparison, frozenTeacher/unchangedPD, same dev conditions. It is a diagnostic, not automatic replacement. No new controller/reward changes. Existing skill evidence prioritized; no mandatory motion probes if prior evidence already answers scope.

Stop: resource/physics boundaries, unavailable validTeacher regions limit aggregation, not allStudent learning. No new research topic or morebudget. Preserve all attempts. Stage ends with report, code/identity,4-or-fewer real-qpos videos and CURRENT_STATE/ProjectMaster; no automatic next stage.
