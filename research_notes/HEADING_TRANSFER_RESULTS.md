# Contact-Guided Heading / Turning — bounded stage closeout

2026-11-03 · [Issue36](https://github.com/GaoE05/robot-research-bridge/issues/36). Completed; no running stage jobs, no next-stage authorization inferred.

## Research decision

Finite-domain Teacher→ideal-force Student straight transfer remains established. S0 remains the deployment baseline; neither new visit-sampling Student universally replaces it. Real contact-guided curved feedback exists, and narrow PPO improves some force/following conditions, but left heading and local requests regress. The selected Heading PPO is a research checkpoint, not a general upgrade. This stage does not establish symmetric reliable turning or a new shared turning Student.

## Straight coverage and controlled sampling

Six independent real Student episodes (alternating S0/S1warm), 36 complete-state takeover continuations, three phase windows per episode. Prefix qpos/qvel/force exactly match. Thirteen bracketed windows across five episodes pass the recorded tail-quality filter, yielding 65 labels, versus historical 20. Local endpoint evidence is not a recovery guarantee for every state or the entire trajectory. One episode yielded no accepted labels.

Two arms start from identical S0, seed805, same network/optimizer and 3000 updates each, batch128: 64 old samples; visited8/new-T30-own56 versus visited32/new-own32. Both use exactly the same 65 visited labels, 649 T30-own and 824 old labels. Teacher-own early-stop data known to be worse than S0 were not forced into this dataset. Mean visited repetitions fall to 369/1477, versus old S2 approximately4800. This isolates the new sampling-ratio contrast; it does not uniquely identify the historical S2 failure cause.

Active-contact 3D tracking RMS cm / actual force RMS N:

|Condition|S0|Low visit ratio|High visit ratio|
|---|---:|---:|---:|
|normal|6.505/3.698|7.006/3.738|8.441/4.457|
|early_stop|6.863/3.611|7.304/3.761|6.852/3.579|
|late_stop|6.457/3.716|6.862/3.488|6.337/3.288|
|local|3.675/2.611|3.801/2.578|5.461/3.950|
|new_early|6.456/3.471|8.331/4.448|6.921/3.994|
|new_late|6.785/4.017|7.006/3.677|5.941/3.329|
|no_contact|0.000/0.000|0.000/0.000|0.000/0.000|

Low ratio better preserves normal/local behavior; high ratio improves new late stop but both regress on new early stop. Neither replaces S0. One optimization seed is not a statistical generalization claim. No extra arm was launched after sealed results.

## Heading source and learning

Two bounded source attempts were retained: M10 rule/projected wrist adapter, then frozen M10 shared Student, each seven source conditions and 4000 Expert initialization updates. Their Direct clones were inferior to Teacher30 in this task adapter, so neither became the PPO prior. Historical M10 left/right heading evidence came from the shared Student, not the raw rule teacher; these new conditions are not exact historical reproductions and do not invalidate the earlier result.

The actual PPO run starts from Teacher30 weights with fresh Adam. Direct privileged825→normalized256D z (norm16)→frozen BFM→original PD; no online M10 Student, CyclicFeedback, force-to-turn state machine, future path or episode clock. Current privileged partner error/velocity is allowed for Expert only. Original reward, 5N cap and std0.008 remain fixed. Actor LR1e-5, critic3e-4, analytic KL bound0.02. Twenty batches×768=15360 RL decisions; 306572 physics steps, 76643 BFM control steps, plus50400 initialization physics steps. 84 logged completed episodes, no logged training physics-boundary termination. These counts are not interchangeable.

Training varies curvature[-0.23,0.30], distance[0.78,0.84]m, ramp[7.7,8.1]s, onset[4.15,4.4]s;35% changes curvature midcourse. Five-condition deterministic development score selected batch10, before final tests. Scores at0/5/10/15/20:2.000/1.989/1.943/2.240/2.078. Initial, every evaluated candidate and final weights remain. Higher critic fit is not counted as actor success; learning curves and PPO diagnostics are in results.json.

### Frozen final comparison

Contact error cm / force N / moving heading RMS deg (heading only while partner planar speed>0.02m/s):

|New condition|Teacher30|Heading PPO10|
|---|---:|---:|
|Left curve|5.220/3.353/10.28|5.914/3.416/13.34|
|Right curve|6.711/4.173/4.13|6.340/4.153/3.74|
|Same prefix→left|5.719/3.525/6.04|5.491/3.314/7.31|
|Same prefix→right|5.925/3.713/3.20|5.543/3.350/3.64|
|Local15cm regression|3.513/2.575|5.902/3.853|
|No contact|0/0|0/0|

Left/right prefix branches have identical qpos/qvel/force/anchor until8.6s. Actual partner curvature changes thereafter, producing real force and learned latent/action differences. This supports causal response to changed physical guidance; the later states also differ, so paired action differences are not isolated same-state actor effects. ForceY can have the same sign in both branches: path labels are not pure force directions.

Whole-contact improvements hide phase tradeoffs: prefix-right hold improves2.23cm/1.70N→1.45cm/1.16N; new-right hold worsens4.50/3.60→5.04/4.09, new-left hold1.11/0.90→2.76/2.20. Right cap fraction0.417→0.387, prefix-right0.287→0.264. New-left moving yaw does not establish a reliable left turn. Thus neither distance traveled nor lower force proves heading quality.

## Recovery, motion quality and analysis correction

Phase metrics use actual applied-force timestamp and Tether active status. Moving/decelerating/holding contact retain XYZ error/force; inactive free recovery uses root/wrist velocities, path, support/slip and continuity. Anchor geometry after release is descriptive, never active-follow success or part of contact-force averaging.

Free root path Teacher30→PPO: new left1.02→4.07cm, new right6.44→8.29cm, prefix left1.03→1.17cm, prefix right1.47→3.12cm. These are settling descriptions, not an arbitrary zero-motion requirement. Target q_PD peaks approximately41–66deg persist; this is not actual joints moving that amount in20ms. Previous right-ankle observation-feedback attribution and failed0.1s smoothing remain; no new smoothing/PD modification.

A postprocessing boundary bug initially included the last pre-contact row in free recovery, spuriously connecting it to post-release and reporting the whole walking displacement as recovery. Fixed with force_time+active, tested onset/release boundaries, reanalyzed saved trajectories only. Original derived v1 files remain as failed evidence; **only physical-metrics-v2 / published physics JSON is valid**. Training, selection and physics were unaffected. A separate analysis-only missing control-row active key was corrected; physical active fields define physical phase metrics.

## Budget, identities, reproducibility

Cumulative supervised job-wall4844.92s (~80.75min), stage output approximately2.80GiB. Student6000 updates; source Expert initialization8000 updates; RL15360 decisions. GPU1 then GPU2 used only after live CPU/RAM/I/O/process checks; maximum two concurrent jobs, all finished/owned process sets empty. Failed sparse-checkout launch6.57s is recorded; it did not load physics/model. No asset download, shared code/PD/weights/main edits, other-user process operation, or force-cap change.

Server root: `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/heading-transfer-20261103`.
- A: `A-prepare-05/job`; runtime `094d4492f7097d9f7ccabb64fefd6b359dcb8a6c`.
- B PPO: `B-train-04/job`; final: `B-final-06/job`; runtime `ee45292eb204e370b002470400b6f443c9f0c65c`.
- Heading selected `B-train-04/job/best.pt`: SHA256 `31963f604b56a500b004a1ed6019f3784a44871c6d840eba254ef06b4762ebc9`.
- Visit-low `A-prepare-05/job/visit_low.pt`: `e91697971921a5e6101144e45d53c4766a30474d3ce1a5faf4c0635f7f274503`.
- Visit-high `A-prepare-05/job/visit_high.pt`: `7f8abe5bc3658dc0908c146841b80cea85f6f0847fe5fddab08105b13ad82743`.
- Frozen identities, full ledger, source attempts, selection, curves: [results.json](heading_artifacts/results.json).
- Phase XYZ/body/foot/continuity: [Student](heading_artifacts/student-physics.json), [Heading](heading_artifacts/heading-physics.json); [causal response](heading_artifacts/causal-response.json).

Core changes: heading_env continuous physical partner curvature; heading_train/worker bounded Direct PPO and development-only selection; heading_transfer complete-state queries and matched sampling; heading_final frozen sealed tests; heading_analyze actual contact-phase accounting; saved-qpos renderer. Path derivative/same-prefix and phase-boundary tests passed. No shared humanoidverse change. Reproduction uses frozen runtime, stage protocols/launch commands in ledger and immutable source/checkpoint identities; do not launch from a later documentation HEAD and assume equivalence.

## Videos and next decision

Two 1×25fps saved-actual-qpos comparisons, not reference animation or resampled best seeds. Mac directory `~/Projects/compliance-bfm-diagnostics/videos/heading-turning-20261103/`; INDEX explains panels and conditions. Both SHA matched server, ffprobe duration and full ffmpeg decode passed. Ready-state3s onward includes all interaction and6s recovery; common unloaded initialization alone omitted. Fixed world and overhead views show feet, wrist, anchor, real force, heading and trajectory.

1. `01_midcourse_right_guidance.mp4`17.08s: force/follow gain, heading tradeoff.
2. `02_left_heading_limit.mp4`17.44s: left-turn limitation, retained negative evidence.

**Unique next recommendation:** a bounded curved-guidance ideal-force Student transfer using frozen Teacher30 as the stable primary teacher, with PPO10 only as a clearly separated feedback-quality comparison in its demonstrated useful domain. Keep S0 straight/local regressions; test same-prefix guidance changes. Do not promote PPO10 universally or require perfect left turns before learning from valid behavior. No next training was started. The immediate capability gap is robust task-appropriate left heading and local-request preservation, not absence of locomotion or failure of the established straight transfer chain.
