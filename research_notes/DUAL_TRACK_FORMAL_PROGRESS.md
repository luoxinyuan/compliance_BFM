# Dual-track formal stage: preparation checkpoint, not research closeout

2026-11-09 · [Issue56](https://github.com/GaoE05/robot-research-bridge/issues/56) ·
[approved-scope contract](DUAL_TRACK_CONTRACT_V1.md) ·
[remaining decisions](DUAL_TRACK_SCIENTIFIC_DECISIONS.md) ·
[small measured evidence](dual_track_formal_artifacts/preparation.json).

## Actual state

The independent stage and versioned contract are created. Original #55 draft is
preserved byte-for-byte. Master six approvals plus the1M limit and this session's
three explicit decisions are recorded separately from proposals.

**Formal Expert PPO=0 decisions; new Student PPO=0; formal FB=0 transitions,
0 agent updates. No formal run is queued.** The remaining applicable scientific
decisions still have pending status; entry admission rejects a launch rather
than importing benchmark defaults. This is not a completed dual-track milestone
and there are no new learning curves or behavioral claims.

## Implemented and actually verified

|Work|Evidence / result|Limit|
|---|---|---|
|97D contact extension|actor/F/B/critics and their targets extended; discriminator robot-only; old robot normalization kept/frozen, new fixed scales|Online collector/replay/full modified-model restore not yet completed|
|Warm-start migration|Real33f410c1… checkpoint; six public inference paths bitwise equal, max error0; fresh optimizer membership includes all9 new online matrices|Synthetic observation parity, not contact-policy learning|
|LayerNorm trap prevention|Minimal counterexample proves naive zero padding changes old normalized coordinates; old LayerNorm domain retained, zero new matrices injected before existing activation|A pre-training engineering hazard caught and avoided; not evidence of a bug in past unextended #55 outputs|
|New weight gradients|Both ordinary and parallel residual input branches have nonzero contact gradient while their new weights start exactly0|Gradient computation only,0 optimizer steps|
|Real physical parity|standing/goal_arm5/motion_walk_task; same complete state and native commands;25 native controls per original/migrated branch, six branches total; low action, actual qpos and complete physics/history digest exact|Short.5s branches, not full task qualification or90 independent scientific source groups|
|Returns|Legacy genuine-terminal numerical equivalence; real-successor time-limit bootstrap, no cross-reset GAE trace; .99/100ms equals(.99**.2)^5|New entry only; old GAE files and historical finite-episode contracts unchanged|
|Data admission|Reject predicted rows as online transitions, reset-crossing successors, missing terminal successor, and shared parent IDs across splits|Pure validation tests; not proof a future collector already saves every field|
|Single-card supervision|Independent immutable jobs/outputs, live GPU0 occupancy check and serial lock, hard timeout, explicit engineering/formal purpose|This is a small supervisor, no Bridge; formal entry still needs complete approved config|

### Exact runtimes / identities

-97D model/admission regression: `82e84807779c2bf295b3b3f63e44c81960cdbe3d`.
- Real physical branch regression: `d3b137f5d23e08845c8db439b74e91013e75c607`.
- Returns regression: `7d1ae5f38cb7a278874021e266fc587d1e20d4af`.
- Source BFM unchanged: `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`.
- Original BC500 retained: `a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e`.
- No new scientific checkpoint exists. Models created for tests are discarded,
  not renamed step0 of a formal learning run.5403b7 remains non-promoted.

Public source uses an independent detached worktree based on a097774, no new
research branch, no modification of original runtime/main/shared assets.

## Resources and honest throughput boundary

Real preflight: GPU0 RTX PRO5000 72GB idle, others untouched;210 shared CPU quota
cores, about679GiB available system RAM, negligible I/O PSI. This is not210
exclusive cores. Shared CPFS had about257GiB free and was99% used; formal replay
capacity and checkpoint storage must be guarded, not inferred from host RAM.

Measured whole jobs (startup/wait included): migration59.09149s,
physical parity130.10831s, returns20.75822s: **209.95802 experiment-job seconds**.
Additionally charge3600s as a clearly labeled conservative preparation/publication
reserve (not measured GPU compute). Conservatively carry #55's4076.20309s into
the common32h allocation to avoid losing prior dual-track engineering cost.
Total conservative common charge7886.16111s; not a new wall-clock throughput
measurement. Private jobs/ledger preserve exact PIDs/commands and raw evidence.

The paired physical test contains150 original/migrated branch native controls;
source/prefill/pre-contact controls were not separately tallied in that engineering
test, so no invented full-transition count is given. Every formal scientific count
is still0. No new Teacher queries, BC updates or F fitting.

**Contact-expanded online collection+FB update throughput remains unknown.** The
#55 unextended estimate1M≈6.3–6.5h must not be re-used as a measured completion
time for the new97D model, input-normalization wrapper, replay or task reward.
GPU0 was released after the tests. No real behavior change warrants a new video.

## Next actual work after the remaining choices are resolved

Implement the resolved generator/parent manifests and shared native-tick reward;
the frozen KL pair uses exactly one locked configuration. Finish FB50Hz online
collection, pre-reset terminal successors, self-contained extended-model and
six-optimizer/replay/RNG restore, then measure the complete online loop and its
physical regression. Only after those concrete checks does the formal1M start.
250k is observational;1M is a hard stop;5M is unapproved.

The four-cell physical table currently has explicitly missing trained cells; see
the decision packet. Do not invent PPO results, qualifications, sealed groups or
Student migration to fill it. Issue56 stays open; this preparation checkpoint is
not a conclusion on which research track wins.

