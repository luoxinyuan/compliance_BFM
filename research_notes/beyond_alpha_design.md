# Beyond fixed forward alpha: bounded capability qualification

Frozen before any new model/physics execution. Inherits private 5cadad3, not a new Bridge or controller.

## At most three candidates

1. **Upward right-wrist guidance (+Z30cm), selected.** Hand raising/repositioning is a meaningful directional requirement independent of desired forward reach magnitude. Existing forward native goal moves the wrist forward/down; existing same-weight raisearms-l-m physically raises the right wrist ~39cm with some waist/shoulder coordination. It is not yet a qualified vertical-contact teacher. A2's unseen-Z failure alone would only show a learned mapping limitation. Therefore include the original forward path with full prearranged progress, giving it the same timing information as the raise path. Exact +Z30cm is fixed as a round capability probe, not fitted to reward endpoint. No claim it necessarily requires large whole-body motion or a discrete selector.
2. **Same +X demand from already-forward/raised posture.** Proprio/history can encode remaining range; potentially requires a posture-dependent path. Existing endpoints are trustworthy only from standing, and no evidence currently proves a different response is needed/qualified from these states. Do not manufacture inability by clamping arm joints. Deferred pending candidate1 evidence.
3. **Lateral/backward wrist guide with body/support relocation.** Signed force and proprio/history are plausible deployable cues, but current movement z has known lateral error and q_PD jumps; other move/spin/crouch reward names are definitions, not qualified skills. Raw LaFAN clips/goal-frame references are not independently executable standing-entry skills. No new candidate sampling or rollout here.

## Assets / interface

Use original FBcprAuxModel checkpoint 33f410c..., standing norm16 z, exact native goal from pinned source, A2 SHA3cc125a..., reward raise JSON SHA3a542d7e... from the prior official matching replay inference. No reward re-inference. Existing replay raw SHA46eac53f..., HF revision62b4206d...; original150k/seed20261030 weighted reward inference with pinned legacy successor-physics mapping. Skill provenance != universal tracking guarantee.

Goal encodes a full privileged target pose offline; constant z deployment needs no future trajectory. Tracking exports per-reference z sequences and official script resets to that clip's first pose/history: availability of40 LaFAN references does not qualify safe transitions from this standing state. Reward only qualifies saved z tied to this checkpoint; many task names have no saved/validated z. No source/PD changes.

## Fixed physical test

Four branches: standing; exact frozen A2; forward_clock; raise_clock. Same full t3 standing state/history, once-aligned unforced wrist anchor at3.8s, +Z30cm in4s smoothstep,1s hold, release8.8s, end10.8s. Original K80,D8,5N,seed0,50Hz actor/PD,200Hz physics. Actual force endogenous in each branch.

Both clock arms use identical alpha=smoothstep((t-3.8)/4), then1-smoothstep((t-8.8)/.75) at release. This is privileged skill qualification, not contact-conditional selection. A2 input/learned release unchanged. No blending S1, no IK, no Jacobian corrections, no PD smoothing.

Measure full XYZ contact error/force, force cap, wrist/shoulder/root and wrist-relative-root, waist/torso, real foot loads, same-observation additional q_PD over standing, saturation and continuity,2s recovery. No noncontact geometric error counted as following. Operational useful-candidate screen: >=10% full-contact error improvement over standing, force<=1.05x, no fall/support loss, recovery<=.02m/s. It is not statistical significance, and full/hold XYZ must still be reported.

Stop at original unsafe physics boundary (height<.65m, pelvis tilt>20deg, native failure, undesired ground contact, actual joint penetration>.02rad), model mismatch, or resources. At most4 completed branches, no post-result tuning or repeats.8CPU/32GiB RSS/8GiB GPU/1800s cumulative budget, GPU required, exact owned process cleanup. Shared free disk is not personal quota. Expected outputs <200MiB; no downloads.

## What conclusions are allowed

A2 poor response does not prove the path cannot work. Fixed-clock poor response plus native endpoint semantics is limited physical evidence for a missing direction; it does not exhaust all possible alpha schedules. Raise vertical displacement without improved XYZ/force is an execution capability, not a good contact teacher. Active body evidence requires same-state extra BFM q_PD plus motion, not passive root displacement.

If raise is useful, a direction plus amplitude latent surface is as plausible as a selector: evaluate signed force/proprio observability and finite intermediate directions before assuming categories. If raise fails contact utility, stop here; ask team/authors for a same-checkpoint right-hand raise/reach primitive with paired init/history/PD/latent/physical trace and neutral-return evidence. No new controller or random latent search.
