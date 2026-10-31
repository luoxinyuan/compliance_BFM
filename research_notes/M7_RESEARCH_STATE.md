# M7 — Contact-conditioned continuous transitions

User milestone authorization: same-episode forward/unload/reverse, frozen M6 baselines, queryable feedback/action projection, bounded aggregation if physically justified. No new controller/PD/weights, no PPO/large assets/push. Prior temporary gates are not global criteria.

Independent source branch gaoe05/m7-continuous, base M6 2d3ed08bfc859223c59e2fa132d7349cb177a161. Issue28 information record, not queue. Run /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m7-continuous-20261101A.

Existing .venv/env.sh and immutable model-view/standing latent/original ComplianceEnv reused. GPU1 selected after live all-GPU check: >65GiB free; user permits sharing if adequate memory. Single owned serial model process,8CPU,32GiB RSS,8GiB VRAM,5400s cumulative. Shared CPFS4.5TiB free is not personal quota; expected new outputs <=6GiB. Only exact-owned process groups cleaned.

Protocol fixed before execution in m7_protocol.json. Task trajectory is changed; Tether force law/apply,PD and BFM unchanged. Segment wrapper calls original Tether.measure with temporary segment origin/delta/activation/ramp/connection, restores task config; no fake sensor observation, no resetting robot at reversal. Whole episode same absolute anchor; true disconnect during gap. Students get only current force/proprio/action history.

Official upstream fetched privately: c002f394f0b77d2f8a85356667fec5eda0f4a1af,6files249add5del vs6023c6e; accumulated latent changes environment/policy execution semantics, deliberately not merged into M6-pinned study. Inherited repository history has a missing old parent ab5602f; fetch succeeded despite ancestry warnings, no history rewrite attempted. Source tree/weight pins control runtime identity.

FIBER https://arxiv.org/html/2610.24675v1 Appendix B/C: LPP aligns same-state physical joint targets; semantic B(s) anchoring is distinct. SF adapter interaction reward+B(s)^T zN is skill acquisition, not universal behavior-quality oracle. We do not add AH-BFM/Flow Planner/SF objectives merely for module completeness.

Current initial qualification: two precommitted development episodes x native/force-rule/original teacher/frozen M6 Alignment/Pz. No training started. Full events, physical XYZ, actual wrench, joint targets/actual q, support and8s release stored. Teacher may not exit under persistent negative force; test first, any algorithm correction explicitly versioned.

Qualification interim evidence: short-gap old teacher reverse error8.460cm/hold11.756cm,forcehold5N,alpha1; frozenM6 Alignment/Pz reverse1.867/1.995cm,alpha0. Old teacher is not a reliable supervision source for reversal. Proposed bounded supplementation reuses frozenM6 Alignment causal progress for native base+original bounded feedback. This is a teacher algorithm change, no new actuator authority or Student privilege. Handwritten signed-gate alternative was prepared locally but never executed or used for results; no scan.
