# Transition-centered privileged latent RL — pre-registration (2026-11-04)

User-authorized new stage, Issue41. Previous force-range A/B expansion is NOT executed. Stage caps: 24000 cumulative supervised job-seconds,15GiB output,196608 RL decisions, up to3 actually idle GPUs; initiallyGPU5/6 only. Existing core contracts/runtime/assets reused. No Student, no shared BFM/PD/main changes.

## MDP and comparison
A shared1088D privileged Direct actor emits256 raw Gaussian commands, normalized to norm16 legal z. Decisions10Hz, five original spherical interpolation/BFM50Hz substeps, PD/physics200Hz. Actor/critic use identical existing current robot/force/action/last-z histories and privileged current point error/velocity/yaw demand. No source identity, future target, phase label, episode clock or motionID is appended. Privileged activation/current goals remain explicitly Teacher-only; deployment identifiability is not assumed.

A: common3s standing initialization, learns whole remaining task. B: same partner/task distribution; frozen source executes in the SAME live environment to a sampled phase, then learned policy takes over without resetting any physics, solver, observation, action,z or contact history. Source prefix is recorded separately, is not PPO reward/sample, and is separately counted as actual physical cost. Finite horizon starts before complete target hold/release/recovery, not a target frame termination. A→A (walk keep, zero/no-force), stand→walk, walk→stop, neutral→bilateral torso yaw, yaw→neutral, walking with yaw added/withdrawn included.

Same M39-A best64 SHA72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a actor/critic/normalization/std; fresh Adam both, seed4111,64×768 decisions each, evaluate every16batches. Raw reliable-domain retention .002, unchanged reward/gamma/GAE/KL/optimizer transaction. No new loss or action-space grid. Equal RL decision budget does NOT imply equal complete episodes or total physics; prefix costs and phase counts reported. Changed task proportions vs M39-B are shared, not attributed solely to takeover.

Source T30 for actual locomotion; verified initial model for yaw source; standing for neutral source. Source policies only create reset distribution, never online mode-select the learned actor. Target physical demands remain continuous original partner paths. Target return source is measured at actual attained posture, not assumed successful from name. Training phase jitter±.2s; sealed exact source phase+.37s plus new partner timing/amplitude combinations. u/goal/motion labels are not observations.

## Force and physical references
Frozen identical model/task diagnostics totalcap5/10/20N, each torso point half total. Force input remains /5 and reward denominator25, no silent renormalization. Initial learned,T30,and standing controls distinguish active effects from stronger mechanical influence, without claiming full causal decomposition. Stop on native physics safety boundary, preserve all failures. 30N not needed/used. Both main curriculum arms fixed5N, so this is not a cap-selection search; higher-force diagnostics do not change training factors. Historical5N comparisons remain.

Native motion/goal smooth direct switch is a clock/reference baseline, not learned feedback or a deployable policy. It uses existing motion and native goal interfaces, never live qpos edits. No scans or new WBC.

B encoding: only sampled native backward-map diagnostics from actual state/velocity; verify reconstruction/normalization. Test association with physical error/velocity/yaw and whether single-pose closeness conflates standing/walking. Not reward, probability or sole success criterion; no F/B updates.

## Evaluation and selection
Common fixed source-state development for both models, and held-out combinations locked in transition_protocol.json before runs. Selection same physical mean relative cost +.5 worst; all phase and severe failures reported. Candidate boundary/worst>1.75initial triggers matched rollback policy. No new endpoint-yaw sign rejection for return-to-neutral episodes. Initial/best/every16candidate/last retained. No final-based reselection or retraining.

Active contact: per-point XYZ errors/forces, cap, net wrench/work, root/torso/pelvis/waist, latency, accumulated lag, entry/hold/withdrawal; support/slip, q_PD vs actual joint changes separately. Released anchor never task error. Observe5s recovery. Prefix/target state mismatch must be reported rather than repaired by replacing with a frame.

Expected reservations: preparation≤2400s; training2×6500s; final≤3000s; rendering≤600s; remaining for bounded engineering failures/analysis. Output ceiling enforced on stage root. Source-prefix steps not hidden from cost ledger. No automatic extension beyond64batches per arm despite larger user ceiling.
