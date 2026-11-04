# Transition-centered privileged latent RL — bounded closeout

2026-11-04. Issue41. Training and frozen evaluation completed; no new training or Student migration authorized by this report. The saved-data attribution is complete to the evidence limit; a unique optimization mechanism has NOT been causally established.

## Decision

Neither curriculum produced a development-selected replacement for initialization. This is not evidence that BFM cannot walk or combine behaviors. Both curricula damaged standing-to-walking initiation while retaining or sometimes improving already-moving behavior. Keep T30/S0 and the initial M39-A model. Do not extend the same training blindly.

## Fixed contract and identity

Standing-start A versus genuine source-execution takeover B; identical 1088D privileged Direct raw-to-legal-256D-z actor, 10Hz/50Hz/200Hz, frozen BFM/PD, 5N total cap, reward and raw replay .002, same seed4111, fresh Adam. B preserves full physical/control/force/z history; source-prefix physics is extra cost, not PPO decisions. No task ID, future trajectory or global clock in actor; current partner error/velocity/yaw remain privileged. No deployable Student claim.

Training runtime `a8afaff087f535124d9e1b28be7ada57caa10bdf`; preparation `8d1a02754807eda84fc2842921c9577c56a9dafb`; final evaluation/render orchestration `5fd2798185212ec65afa56753051900305bb28a5`. All existing runtime worktrees are preserved.

Initial M39-A SHA `72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a`. Both serialized selected best SHA `e1e33b521e829647e7aaac3a04b7e019f0ed99eb94883371f4437ae3d6da1fcb`, iteration0. Serialization metadata differs from the source checkpoint; this is NOT a learned improvement. A/B initial tensors matched. Selected final physical phase metrics are identical to initialization for every condition.

## Learning and development evidence

Each arm completed64 batches ×768 =49,152 RL decisions; total98,304. A219/B242 optimizer updates. Both selected iteration0 by preregistered common-source-state development score (lower better; initial1.5).

| Batch | A score | B score |
|---|---:|---:|
|16|1.56448|1.68585|
|32|2.01280|1.99775|
|48|1.56521|1.70593|
|64|1.99521|1.76255|

Selected development summaries below use the existing summary metric, not final phase RMS; different candidate rows are diagnostic, never a spliced policy.

| Condition | Initial error cm / force N | Candidate | Error cm / force N |
|---|---:|---|---:|
|standing→walk|5.34 /3.29|A32|48.03 /4.62|
|standing→walk|5.34 /3.29|B32|50.03 /4.62|
|standing→walk|5.34 /3.29|B48|14.23 /4.00|
|already walking→continue|5.37 /3.85|B48|4.84 /3.36|
|walking+left rotation|4.09 /2.84|B16|4.16 /2.57|

A32/A64 and B32 crossed the predetermined worst-cost1.75 threshold and restored initialization/optimizer, reducing actor LR. Reversed-rotation flags were false. Thus this is not an uninterrupted64-batch convergence curve; candidate checkpoints and rollback records are preserved. Severe startup loss is a physical failure, not merely a conservative score rejecting a good model.

## Saved-trajectory attribution

Reproduce without models/GPU/simulation:

`python3 research_notes/transition_saved_audit.py /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/transition-curriculum-20261104`

Compact evidence: `transition_artifacts/saved-attribution.json`. All eight evaluated trained candidates had identical initial physical qpos to the baseline. At t=4.0s, A32/B32 first executed-z differences were only0.1412°/0.1148°; first q_PD RMS differences0.0243°/0.0095°. Their root trajectories first separated from baseline by >2cm at6.545s/6.525s. This diagnostic threshold is descriptive, not a success criterion.

At14s A32/B32 root had advanced only4.54cm/1.12cm from handoff; A64 only1.04cm. B48 ultimately advanced85.83cm but was delayed (11.17cm at8s); B64 only0.70cm at8s then89.17cm at14s. Hence later catch-up and total displacement conceal initiation failure/delay. The first command is not a large instantaneous latent jump, and mismatched initial qpos is not the explanation for these paired traces. Small initial differences can amplify through feedback; no claim that one particular joint, reward term, task gradient or exploration setting is proven responsible.

Known: updates alter initiation robustness, while some ongoing-motion metrics improve; both curricula exhibit it. Hypotheses still requiring intervention: cross-task update interference, fragile startup feedback/insufficient useful exploration, mismatch between sampled reward and deterministic development. Current traces do not identify one as the unique cause. No saved-state replay is mislabeled as a causal counterfactual. No further loss tuning or force-cap training was performed.

## Frozen final evaluation

48 episodes =12 new combinations × initial/T30/A-selected/B-selected. A/B selected actors are initial, so this is24 unique model-condition comparisons, NOT48 independent evidence units or new trained-policy success. All phase results are in `final-physics.json`; full qpos/control histories preserved. No final-based reselection.

| Selected initial behavior / phase | 3D RMS cm | actual force burden RMS N |
|---|---:|---:|
|new straight moving|8.001|3.936|
|new straight connected hold|0.997|0.796|
|left rotation withdrawn, translation continuing|4.645|4.736|
|right rotation withdrawn, translation continuing|2.860|4.282|

Dual-point burden is RMS of sum of point force norms, not net force. Released recovery has no anchor-following score. The straight moving q_PD maximum step46.26° versus actual q maximum sampled step2.15° are distinct quantities. No per-model causal improvement is inferred from duplicate panels.

## Preparation and force evidence retained

32 preparation episodes; three complete source prefixes independently reproduced including next100ms. At frozen initial policy, positive combination5/10/20N error9.107/3.790/4.108cm and burden4.609/5.153/6.715N; straight5.665/6.533/6.542cm. Increasing cap helps some combination positions but worsens straight burden, and20N also strongly moves standing control. Not proof that20N fixes learning. Both trained arms remained5N.

Native left/right goal endpoints execute; tested naive walking/goal mixture fails (~44cm error). This single reference does not exhaust BFM combination ability. Backward-map preprocessing reconstruction state max difference2.98e-8; B distance remains exploratory, not calibrated arrival probability or sole behavioral metric.

## Artifacts, costs, videos

Run root: `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/transition-curriculum-20261104`.
- A-ppo-03/job and B-ppo-04/job: initial/best/candidate/last, dev trajectories, training and rollback records.
- C-final-02:32 preparation episodes; C-final-05:48 frozen final episodes.
- `final-physics.json`, `prepare-results.json`, `compute-ledger.json`, `continuation-status.json`.
- Ledger compute/render jobs all finished; wall total approximately11,296.6s, plus separately recorded saved-data CPU analysis. No training extension or force-range A/B. First failed preparation3.43s was missing materialized source (0physics), then fixed privately; failure retained.

Mac `~/Projects/compliance-bfm-diagnostics/videos/transition-curriculum/` contains01-walk_add_left_new.mp4,02-walk_add_right_new.mp4,03-stand_walk_new.mp4 (17.44/17.44/16.48s). Files SHA/duration/full-decode verified. Saved real qpos at1×; source prefix included according to render configuration. **Three panels are the same selected initialization; fourth is T30. These videos show existing capabilities, not improvement from this training. Visual QA is not claimed by this closeout.**

## Next research decision

Keep existing working models. Recommend a bounded startup-focused causal diagnosis before more broad PPO: from identical saved pre-step states compare initialization and a failing candidate's continuation/takeover, then distinguish loss of useful control from update/exploration objectives. Preserve ongoing-walk tasks as controls. This is a proposal, not a newly launched experiment or proof that startup must be solved perfectly before any further research.
