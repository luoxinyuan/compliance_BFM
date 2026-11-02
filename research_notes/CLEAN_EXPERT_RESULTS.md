# Clean privileged Expert — completed bounded comparison

Verified 2026-11-03 CST. Both 20-batch training runs, frozen evaluation and saved-qpos videos completed. No Student training; old M11 continuation remains stopped. This is a negative generalization result, not a reliable Expert upgrade.

## Control ownership and actual learning

Runtime `05b313e6c4a14aff1d2e3ee08432f9d231c6822f`; preparation `999494f7a25224b27780f504af797c32b6ca6879`. A learns alpha, phase rate and tangent residual; B directly outputs normalized global 256D z. Neither calls online M10 Student or CyclicFeedback force-to-motion rules. Frozen BFM/PD, 5N Tether, 200Hz physics / 50Hz BFM / 10Hz high-level policy unchanged. Input includes privileged current contact error/relative velocity, proprioception, real force history and own command memory, not future trajectory or task ID.

Both initialized with 1500 supervised updates from one valid old expert trajectory; fresh PPO optimizer, seed101, 128x128 networks. This is not random-from-scratch behavior discovery. Different parameterizations produce unequal initial physical behavior. Shared prior reward coefficient is zero because its former nominal target depended on the removed rule. Other task reward/physics unchanged. First attempts exceeded KL after an optimizer step; transactional model+Adam rollback/backtracking repaired this, five tests passed. Initial interrupted evidence retained. Subsequent accepted per-batch KL <.02 does not guarantee stable long-horizon behavior.

## Actual physical results

Metrics: active-contact 3D wrist-anchor RMS cm / actual capped-force norm RMS N / saturation fraction. No release-zero force counted as active following.

| Policy | Development error | Force | Cap | New condition error | Force | Cap |
|---|---:|---:|---:|---:|---:|---:|
| Standing |54.550|4.710|—|52.945|4.702|84.67%|
| M10 feedback expert |7.796|4.363|—|7.445|4.036|53.11%|
| Direct initial, before PPO |7.502|4.059|46.45%|6.954|4.066|44.75%|
| Direct selected batch15 |6.713|3.684|39.05%|8.291|4.200|52.51%|
| Direct last batch20 |8.541|4.261|47.93%|not evaluated|—|—|
| Structured initial |32.672|4.674|84.48%|not evaluated|—|—|
| Structured selected/last20 |29.033|4.666|81.57%|48.306|4.673|84.37%|

Development: +X82cm, onset4.25s, ramp7.8s, release14.05s, end20.05s. Predeclared heldout: +X80cm, onset4.4s, ramp8.05s, release14.45s, end20.45s. Same standing start, heading0, K80/D8/5N. Selected checkpoints fixed before final testing. Initial-policy evaluation on the same heldout was added after the selected-policy failure for attribution, not selection; condition is now exposed and cannot be reused as a new blind test.

Direct15 improves its own development initialization by 10.5% error and 9.3% force, but worsens heldout error by 19.2% and force by 3.3% versus initialization. The initialized interface can execute this condition; this is not evidence that BFM cannot walk or that no usable control representation exists. It is not yet evidence that PPO learned robust beneficial feedback. One training seed per architecture; no statistical superiority claim.

## Regression evidence and stopping

Direct15→20 on development: movement root Y displacement 4.15→11.33cm; hold 3D error 2.744→5.062cm. Hold signed anchor-minus-wrist XYZ changes from [+2.015,-0.712,+1.598]cm to [-2.364,-4.371,-0.466]cm, i.e. forward overshoot and lateral deviation. Hold Y force RMS .646→3.399N. Peak q_PD jump decreases 49.28→45.11deg, so a larger peak jump is not the explanation. First sustained >=1cm wrist divergence at7.235s. Same saved observations give command-angle change only .274–.304deg average, max .774deg; accumulated five-batch joint Gaussian KL .171–.215 mean is not a per-batch threshold violation. Small policy changes can accumulate through gait/contact dynamics; these correlations do not establish one specific causal joint.

Heldout Direct15 movement root delta [.8234,.0825,-.0120]m; tilt peak9.90deg, q_PD jump49.55deg. Initial: [.8468,.0707,-.0017]m,10.40deg,45.83deg. Structured20 root forward movement only6.15cm, q_PD jump67.09deg. All complete horizon without recorded safety boundary or total support loss. Last1s recovery root speed RMS: Direct15 .00147m/s, initial .000717m/s, Structured .000853m/s. Full half-second recovery curves, XYZ force and support/joint/saturation diagnostics in linked JSON. Release boundary residual active samples are not sustained release force.

No-contact Direct last20 root displacement .330cm versus initial .376cm; Structured20 5.339cm versus initial4.14cm. Direct15 no-contact was not separately evaluated. Last20 small-request regression error4.043cm/2.722N; do not assign last20 results to selected15.

## Identity, data, reproducibility

- Direct selected15 SHA256 `92692be758d04386c32bea3108b3b155d4c37b0c2b70679e78cf849f593112fe`.
- Structured20 SHA256 `74f2590f6c344ee96e8839c43f5bd1ecf17520e3665954c401aaec95983d00fe`.
- All evaluated checkpoint paths/SHA and final episodes: [clean_final_evidence.json](clean_final_evidence.json).
- [clean_development_diagnosis.json](clean_development_diagnosis.json): training counts, development physical phases, same-observation output differences.
- Server root `P=/mnt/data/temp_user/yi.gao/projects/compliance_BFM`; run root `R=$P/bridge-v2-owned/runs/clean-contact-expert-20261102`.
- A `structured-train-06/job`, B `direct-train-08/job`; `best.pt`, `last.pt`, numbered candidates and training/development logs retained. Final outputs `direct-final-09`, `structured-final-10`, `direct-final-12`. No asset/checkpoint/raw trajectory committed.
- Frozen evaluation command is in each final job `launch.json`; `clean_train.py --stage final --kind direct|structured --resume <exact checkpoint> --output <new directory>` through the bounded supervisor, after refreshing resources. Do not overwrite outputs or blindly relaunch from this report.
- Analysis: `evaluate_clean_saved.py` compares saved observations using only high-level MLP on CPU; no CPU BFM or simulation. `finish_clean_evidence.py` reads final trajectories. Existing GPU physics runtime unchanged.

## Video delivery

Mac `~/Projects/compliance-bfm-diagnostics/videos/clean-contact-expert/` contains only three key MP4s + INDEX. Actual saved physical qpos, 1x, fixed oblique and overhead, force/anchor/heading overlays; not target animation. Videos begin at ready t=3s and include all contact and recovery (initial settling omitted explicitly).

1. `01_initialization_teacher_structured_direct.mp4`: expert / A initial / B initial,17.08s.
2. `02_teacher_direct15_direct20.mp4`: same development expert / B15 / B20 negative result,17.08s; SHA `09557718cbc2d8863d5665a41240a9fa3b92c3e5618ed4c2736b97629b36be63`.
3. `03_unseen_teacher_initial_direct15.mp4`: same heldout expert / B initial / B15,17.48s; SHA `57d8b4394c5d00ad44511bba3050052d3446b412c8493a77bf08db1c6d570b5e`.

Server/local hashes match; full ffmpeg decode and duration checked; representative frame visually inspected. Per-video manifest keeps trajectory identities.

## Research decision

Do not promote selected15 or continue this single-condition run blindly. Keep Direct's useful initialization and independent control interface. Next useful controlled experiment is narrow-distribution PPO with predeclared multiple deterministic development conditions and preservation of initialization behavior, compared to unchanged initialization, with a fresh heldout combination. This targets update robustness/coverage rather than enlarging residual gain or changing force cap. Single-condition tuning has produced a development benefit but not transferable improvement. Structured alpha/phase learning remains an unresolved, separate execution problem, not proof that direct control is universally superior. No Student migration warranted yet.

Resource closeout: 35328 RL decisions total including 4608 interrupted; completed A/B each15360, 20 batches (A42/B57 optimizer updates). Summed supervised process wall 4879.86s against14400s ceiling (not simultaneous elapsed time). All ledger attempts finished, no remaining owned processes; video-only jobs perform no physical rollout. Shared checkout, BFM/PD/weights and other users untouched.
