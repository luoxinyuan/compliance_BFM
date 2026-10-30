# Native reward skill qualification — completed 2026-10-30

The same-checkpoint replay-to-reward-latent chain now works. One fixed arm-dominant candidate met its native wrist-height semantics and returned to standing. The locomotion candidate produced actual root translation and changing foot support, then stopped, but has substantial lateral drift and abrupt PD targets. This is evidence of executable native whole-body behavior, not a qualified clean 0.3 m/s locomotion teacher or learned contact-conditioned transition.

## Provenance and fixed protocol

- Private branch: `gaoe05/bfm-reward-skill-qualification`.
- Inference runtime: `8b89e6b584770d1e0cf8d1cd7e77d2434abe44f3`; physical runtime: `5cc21b7c7545fc769c59f71386d00c5452fd173a`.
- BFM weights SHA256: `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`.
- Official HF revision: `62b4206d68e026de5e5dc7efb1529bccfb95164c`; buffer 26,675,212,288 bytes, full SHA256 `46eac53f603e45921d25504fcf7d3eea861f2a9f8ea503046855e701493b546b`.
- `reward_locomotion.pkl` SHA256: `4c4b811584807f2e6c0ddd9f6cb32f79ea2274781449b005e4c7e8e670f67374`.
- Seed 20261030, 150,000 original trajectory-sampler transitions, identical sampled indices for both rewards, one z per reward, no search or reward tuning. Original global softmax(10r), reward weighting, backward map and norm-16 projection; FP32, 4096-row inference chunks, AMP/TF32 off.
- The SHA-pinned legacy physics-field alignment repair described in `reward_stage_preparation.md` is applied. It pairs original successor observations with raw row-t physics (already a successor due to the old reused-info-dictionary behavior). Raw data are unchanged. The 31-row semantic audit is not a proof of every reset-edge record, and the exact buffer-generation commit remains unknown.
- All three branches exactly reproduce the full saved t=3 s standing state and original zero-force prefix. Standing -> candidate uses fixed 0.75 s smoothstep/slerp, candidate execution 2.5 s, return 0.75 s, standing recovery 2 s. Actor/PD 50 Hz; original MuJoCo 200 Hz. No Tether, direct joint residual, teacher, Student, or training.
- Native BFM inference modules ran on CUDA; original MuJoCo remains its CPU physics engine. This is the established GPU-BFM/MuJoCo chain, not a CPU fallback for BFM. The reward z needs no full future motion reference at deployment. Timing here is an externally scheduled qualification, not contact-conditioned selection.

## Physical evidence

All three completed 6 s and triggered none of the registered safety stops. Three 3 s reconstruction prefixes plus three 6 s branches give 27 s recorded prefix/branch physics; constructor/reset steps are not included. Values below describe the 2.5 s constant-command execution phase unless noted. Wrist positions use the native wrist-roll bodies; velocity is a diagnostic from 200 Hz COM position differences projected by pelvis yaw, not a recomputed full native reward score.

| Metric | Standing | raisearms-l-m | move-ego-0-0.3 |
|---|---:|---:|---:|
| Mean left/right wrist height, cm | 86.37 / 86.14 | 77.39 / 125.35 | 97.19 / 79.77 |
| Native arm height bounds attained | 0% | 100% | Not the task |
| Root endpoint XYZ displacement, cm | 0.002 / -0.021 / -0.059 | -1.30 / -3.50 / -0.11 | 54.89 / -27.66 / -7.31 |
| Mean ego forward/lateral COM speed, m/s | 0.00005 / -0.00039 | -0.0138 / -0.0087 | 0.1873 / -0.0913 |
| Minimum root height, cm | 77.98 | 77.92 | 69.56 |
| Maximum pelvis tilt, degrees | 2.42 | 2.69 | 9.18 |
| Maximum torso orientation change, degrees | 1.72 | 11.60 | 27.87 |
| Maximum waist yaw/roll/pitch change, degrees | 0.10 / 0.07 / 0.09 | 5.07 / 8.72 / 1.37 | 9.25 / 4.35 / 10.82 |
| Both-foot / any-foot geometric support | 100% / 100% | 100% / 100% | 3.20% / 92.60% |
| Maximum left/right foot lift, cm | <0.001 / <0.001 | 0 / 0.62 | 18.73 / 8.72 |
| Maximum 20 ms q_PD target change, degrees | 0.0055 | 2.55 | 58.99 |
| Torque saturation, fraction of joint-time samples | 0 | 0 | 0.001103 |

Native `raisearms-l-m` means left wrist 60–80 cm and right wrist >=100 cm. Its right wrist rises 39.26 cm on average, relative-root rise 39.33 cm; root translation remains small and both feet stay supported. Right shoulder mean XYZ displacement is [-1.73, 2.50, 1.38] cm. This is arm-dominant, not arm-only, and is not proof of forward wrist compliance. Its latent is 85.55 degrees from standing; move's is 82.90 degrees. Neither lies in the old Student's approximately 16.7-degree neighborhood, which was not changed.

For move, right wrist mean world XYZ displacement is [40.98, -3.09, -6.33] cm, versus [0.96, 2.93, -1.66] cm relative to root. Right shoulder displacement is [41.09, -7.45, -5.53] cm. Foot lift, new geometric contacts and subsequent normal forces (54.99–761.90 N within 50 ms of the detected recontacts) support actual support transitions; there is no external force dragging the robot. These observations do not establish a clean repetitive gait. Contact geometry and >1 N force-support differ in 9/1200 frames around transitions, so geometric support fractions must not be treated as exact load-bearing percentages. Execution contains 37/500 frames without either geometric foot contact. Ego forward speed std is 0.1804 m/s, with substantial side motion.

## Entry/exit and first quality boundary

| Maximum q_PD change per 20 ms, degrees | Standing | Arms | Move |
|---|---:|---:|---:|
| Entry, 0–0.75 s | 0.006 | 6.33 | 21.23 |
| Execute, 0.75–3.25 s | 0.005 | 2.55 | 58.99 |
| Exit, 3.25–4 s | 0.005 | 4.03 | 12.72 |
| Standing recovery, 4–6 s | 0.004 | 2.56 | 36.48 |
| Last 1 s only | 0.004 | 0.113 | 0.341 |

The move maximum occurs at branch t=3.10 s, right knee, alpha=1 (constant native z): raw action -0.1391905 -> +0.4479850. Thus it is not merely a command-switch discontinuity. Another 36.48-degree left-ankle target change occurs at t=4.00 s when z has returned to standing. These are requested target changes, not instantaneous realized joint-angle changes. An independent saved-record reconstruction of the original scaling/clipping contract agrees within 1.97e-7 rad; adjacent-target differencing also agrees. No action clipping or actual joint-limit penetration occurred. This rules out a reporting/scaling arithmetic error but does not establish whether the cause is candidate quality, model/state distribution or deployment dynamics.

Both candidates return to near-zero root translational velocity (arms 0.00562 m/s; move 0.00241 m/s). Arms' final root displacement is [0.53, 1.94, -0.14] cm. Move finishes at [63.12, -25.00, 0.10] cm, with 99.75% double-foot geometric support during recovery. Returning to standing means posture/speed, not undoing locomotion world displacement. Neither fell; native failure, nonfoot ground contact, nonfinite state, <65 cm root, >20-degree pelvis tilt and >0.02 rad actual joint penetration were absent. Torso orientation change includes yaw and is not the pelvis-tilt safety metric.

First scientific quality boundary: the mobile native command is executable but direction/speed fidelity and action continuity are insufficient evidence for a reliable whole-body contact teacher. No numerical q_PD-step acceptance threshold was preregistered, so this is an evidence-based suitability judgment, not a retroactively invented hard-stop failure. One seed/state and one z per task do not establish robustness. Arm success does not establish a semantically suitable pair for hand-guided whole-body transition.

## Resources, validation, preserved failures

After explicit user authorization to share a GPU with sufficient VRAM, all seven cards were inspected; GPU5 had the lowest sampled load among current candidates and sufficient free VRAM (~64 GiB). No other process was stopped. Inference completed in 338.58 s, peak owned RSS 27.33 GiB; physics in 162.70 s, peak RSS 6.61 GiB and sampled GPU use 582 MiB. Total numerical wall time 501.28 s within the pre-fixed 3600 s budget. Inference's 5 s nvidia-smi polling missed its brief CUDA-memory peak (ledger 0 means unobserved, not zero usage); CUDA placement and actual CUDA z assertions are saved. Process-group bookkeeping reports no remaining owned processes. No claim of server OS sandbox isolation.

Nine tests passed in three separate groups: six buffer/contract tests, two fixed-schedule tests, one physical-summary units/phase test. AST/JSON and git diff whitespace checks passed. The saved-record evidence audit also passed. Mac's system Python initially lacked numpy for the summary test; it was run successfully in the existing project environment, without installing packages. Earlier buffer alignment v1/v2 failures, loader/config fixes and all evidence remain. No inference/physics rerun or sample selection was needed; two first-attempt runs, no strategy updates.

Core changes are isolated research notes: SHA-pinned replay reader alignment and memory placement; bounded serial reward inference/supervision; original-environment fixed native-z qualification; saved-trajectory metric/audit scripts. No shared source or weights, S0/S1, C/D, teacher algorithms or old results were changed. No push or Bridge execution.

## Locations and reproduction

`P=/mnt/data/temp_user/yi.gao/projects/compliance_BFM`

- Code: `P/bridge-v2-owned/trees/issue-21-reward-stage/repo/research_notes/`.
- Run: `P/bridge-v2-owned/runs/issue21-reward-stage-20261030A/`.
- Latents, rewards, sample indices: `inference-attempt-01/outputs/` within the run.
- Full physical state/action/contact traces and original runtime contract: `physics-attempt-01/outputs/`.
- Aggregate metrics and independent audits: `analysis/physical-summary.json`, `analysis/evidence-audit.json`; tests and resource ledger alongside them.
- Replay asset: `/mnt/data-oss/users/yi.gao/compliance_BFM/native_reward_assets/62b4206d68e026de5e5dc7efb1529bccfb95164c/train/buffer.hdf5`.
- Mac source mirror: `~/Projects/compliance-bfm-diagnostics/issue21-reward-stage/`.

Offline analysis only, after sourcing `P/env.sh`:

```sh
python research_notes/reward_skill_summary.py "$RUN/physics-attempt-01/outputs" "$RUN/analysis/physical-summary.json"
python research_notes/reward_skill_evidence_audit.py "$RUN/physics-attempt-01/outputs" "$RUN/analysis/evidence-audit.json"
```

Historical protocol's `awaiting_owner_gpu_available_notice` and inference result's `physical_qualification: not yet run` reflect when those immutable artifacts were written. Current status is the completed physics result and compute ledger plus this report; old artifacts were not rewritten.

## One next research decision

Do not start Contact Projector training from this mobile candidate yet. Recommend one bounded native-locomotion deployment-consistency qualification: compare this fixed command/state/action trace with a team/author-validated same-weight locomotion z plus its original timing/PD configuration, to isolate candidate-inference quality from deployment behavior. Do not respond by inventing joint smoothing/WBC, tuning rewards until a favorable seed, or silently expanding the old Student radius. The current milestone ends here; entry into the next stage requires the user's decision.
