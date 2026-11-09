# DUAL_TRACK_CONTRACT_V1 — draft 1, 2026-11-09

Status: engineering draft; **not Yi-approved scientific training contract**. No formal PPO/FB run authorized by this document. Two tracks share task/quality/splits and differ only by explicitly registered model/training permissions. BC500 remains baseline; 5403b7 not promoted. Source: f9971405bad22bcff8134f226356a13eb5cba51f and read-only audit 2026-11-09. Existing FB update loop is retained; no new fork/UFO/mjlab required in this milestone.

## Common task / source / partner

Scope: right-wrist forward tether guidance only. Requests: connected zero demand, local, medium, sustained long pull, same-prefix stop/continue, holding, real release/free recovery. Distance and speed must vary independently. A current partner target is not a future-distance label. Standing, goal_arm5 arm-ready and motion_walk_task moving sources are generated through actual native BFM execution, including entry and support phase. Never reset qpos alone. Common prefixes and original source trajectories form parent groups, even across seeds/conditions. Joint source × request × support/phase coverage is counted from real executed decisions/episodes, not sampling probabilities.

Partner: continuous position/velocity; current target and velocity may be privileged. Constant-current-velocity prediction, if used, explicitly separate from true future. No future trajectory, final distance, task ID, time-to-stop or episode clock to learned actor. Natural initial states must be reached physically with original solver/history.

## Physics and controls — inherited, not newly proposed

Original frozen BFM/PD on high-level track, raw256 → norm16 z, 10Hz high-level/50Hz BFM/200Hz MuJoCo. Single reliable contact environment num_envs=1. Tether K80/D8, total cap10N, onset anchor aligned to current actual wrist; 5N historical regression separate. Spring/damper recomputed each physics tick, cap recorded before/after, equal/opposite interpretation and world/body frames explicit. Contact is bidirectional maintained tether, not unilateral pressing. Release turns off real force; no anchor-error success metric afterwards. No silent reanchoring, early disconnection or cap change. The FB track is a separate mutable copy of base network with fresh optimizer/replay, not mutation of shared BFM. Its native low-level action is 29D at50Hz, not high-level PPO's256D z.

## O_S and O_P access contract

O_S existing BC500 exact1639 path: normalized robot state64 + last low-level action29 + actor history372 =465; actual executed previous z/16 =256; per-mount body-frame ideal force history100×3slots×3=900; known installation/link/local-point encodings3×6=18. No active truth bit. Force history uses the historical implementation's present pelvis rotation for expressing past world forces; record this convention, do not silently change it. Three slots preserve checkpoint shape, only wrist mount used here. `center/scale` are loaded by exact state_dict key, no refit or concatenated normalizer replacement. Ideal-force is a simulation sensing assumption, not proven G1 hardware capability.

O_P: original current privileged1088 vector from `Task.features()` in addition to exact O_S. Contains current robot state/history/last-z, force/contact installation, current anchor error and relative/partner velocity, current heading demand. It intentionally duplicates some O_S fields. A new branch sees O_P, zero-initialized final layer; original1639 base path remains intact. O_P is Teacher-only, including any derived network feature. Actor input field audit must trace actual concatenation. Extended base FB observation is **not implemented**; current B reads robot state64+privileged463 only. Future contact-extension dimensions/normalization require separate explicit approval and migration.

## Quality and phase accounting

Per-point actual XYZ force/RMS, pre-cap demand, saturation physics-tick fraction, 3D/signedXYZ error. Wrist-relative-shoulder/torso, shoulder/elbow margins, pelvis/torso/waist, root and foot paths, loaded-slip peak/p95, support changes, actual q/dq/torque and saturation, q_PD peak/p95 jumps. Report startup, sustained, deceleration, connected holding, released recovery separately. After release: residual speeds, extra steps/displacement, stability, not old-anchor tracking. Do not treat q_PD jump as actual joint motion. No metric diluted with no-contact zero forces. Scientific pass thresholds below remain pending.

## Parent split and evaluation isolation

Store source trajectory ID, prefix ID, episode ID, source support stratum, complete state index, action/previous-z histories, simulator solver, RNG states, model/runtime SHA, task/version. Every candidate and adjacent state from one parent stays on one side. Same-prefix stop/continue stay together. Seen standing handoff4.0, arm5.6, old8/59/66/72cm, source-M1 cases, retention13 and historical regression paths are exposed; never relabel them blind. Existing exposed regression set: retention short08/normal72/stop66/none plus Issue53 short06/continue59/stop59 and source-M1 controls; these are references, not new test data.

New sealed **schema/list, no numeric instantiation yet**: standing long-A/B (different native support/prefix), arm-ready local/interior-domain/long, moving continued/stop, matched-prefix stop/continue, zero-demand/no-contact. All derive from distinct parent native episodes; exact distance/speed/onset/phase/seeds/group allocation await Yi. No sealed evaluation is run in this engineering milestone. No test-driven candidate/threshold/model changes.

Independent evaluation runs in child process with fixed independent seeds, no live training Task reset. New live branch restore includes full CrossResetState callbacks/spec bindings/MjData integration+solver/history, Python random, NumPy global, Task Generator, Torch CPU/all CUDA, explicit sampling Generators. This is in-process branch restoration; portable cross-process training resume is not claimed without serialization test.

## Counting and failures

1 physics tick=5ms;1 native transition=20ms;1 full high decision=100ms/5native controls/20physics ticks. Count reset/source steps separately.1 FB agent update means discriminator+F/B+critic+aux-critic+actor+targets, batch1024. Default16updates/1024new native transitions means15625updates per1M,78125per5M. Do not confuse replay draws with new transitions. Count formal vs disposable benchmark updates explicitly. Compile first-call, steady state, sampling, transfer, optimizer, replay/data IO, checkpoint save/read and evaluation separately; joint wall uses full path, not no-update FPS. Independent groups != candidate rows != labels. Search queries/predicted steps != real executed commands != on-policy samples.

Failure classes: correctness (probability/state/label identity → immutable erratum); source not maintainable; candidate expression/entry failure; prediction–execution mismatch; Teacher recovery invalid; deployment information insufficient; optimization/update interference; task-quality tradeoff; resource/backend/runtime failure. Unknown evidence remains unknown, no causal label from correlation alone. Original assertions3e-4 logprob/.02 KL remain unrelaxed by default. New entry centralizes all KL fields; legacy runtime unchanged.

## All numerical decisions requiring Yi approval (not defaults)

| Decision | Proposed options/reference only | Status |
|---|---|---|
| source/request mix and support strata | standing/arm/moving cross-coverage; proportions unset | pending Yi |
| distance/speed/onset/hold/stop ranges | include local/interior/long; all endpoints unset | pending Yi |
| new parent train/dev/sealed counts and exact seeds | group by source+common prefix; counts unset | pending Yi |
| serious protection and quality tolerances | M1 long10cm/6.5N, mid8cm/7N, local4cm/4N, saturation15%, slip.45,yaw20° are **historical references only**, not automatically adopted | pending Yi |
| scientific checkpoint rule and Pareto tolerances | no average-only masking; exact tolerances unset | pending Yi |
| formal PPO KL/step/std/critic schedule | inherited guard available, change values only explicitly; no formal values selected here | pending Yi |
| formal base FB contact fields, widths, normalization and migration | current robot-only base plus proposed partner/contact schema; final dimensions unset | pending Yi |
| replay composition/expert source and auxiliary reward weights | throughput replay is NOT approved learning dataset | pending Yi |
| formal budget/updates/checkpoint cadence | determine after single-card measurement; no1M/5M commitment | pending Yi |

## This engineering milestone's bounded measurements (not scientific thresholds)

GPU0 only after live preflight, personal worktree/output, no shared asset writes. High-level: one768-decision physical batch+one original6epoch PPO update in discarded copy. FB: two128 full-update segments at original1024batch/architecture; first eager, second existing torch.compile; fresh replay filled only by new physical data, no old replay resume. Max benchmark job-wall14400s/output14GiB, stop on correctness failure; no formal training or ranker. Figures are engineering test sizes, not Yi scientific approval. A failed attempt is archived separately and charged. Base checkpoint weights-only warm-start; all new scientific counters zero at start. No installation required. Isaac vectorized partner/Tether remains implementation work, not achieved throughput.
