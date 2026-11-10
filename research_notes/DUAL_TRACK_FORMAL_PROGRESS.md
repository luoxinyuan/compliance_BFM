最终状态（2026-11-10）：本文件保留历史执行快照；正式结项、真实曲线与缺失格子见[DUAL_TRACK_RESULTS](DUAL_TRACK_RESULTS.md)。两KL完成且均选初始化；FB按用户授权提前停，末轮存盘缺口另有勘误。

# Dual-track formal stage: preparation checkpoint, not research closeout

2026-11-09 · [Issue56](https://github.com/GaoE05/robot-research-bridge/issues/56) ·
[approved-scope contract](DUAL_TRACK_CONTRACT_V1.md) ·
[approved P1–P9](DUAL_TRACK_SCIENTIFIC_DECISIONS.md) ·
[small measured evidence](dual_track_formal_artifacts/preparation.json).

## Actual state

The independent stage and versioned contract are created. Original #55 draft is
preserved byte-for-byte. Master six approvals plus the1M limit and this session's
three explicit decisions are recorded separately from proposals.

**Formal Expert PPO=0 decisions; new Student PPO=0; formal FB=0 transitions,
0 agent updates. No formal run is queued.** Yi has explicitly approved the entire P1–P9 packet7354e0a. Entry admission
checks the resolved authorization record rather than importing benchmark defaults.
Necessary runtime validation is in progress, not a new permission gate. This is not a completed dual-track milestone
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


## Subsequent approval and entry validation

Yi explicitly adopts the complete P1–P9 packet7354e0a. Configuration is resolved.
Stage-smoke-03 passed at1df24ee0ef214478b809b8a02c744b30486317bf:
three actual native prefixes, true terminal successor, deferred auto-reset,
truncation and complete state/history plus independent physics RNG.
The first two stage smoke failures were engineering defects before any formal
training (reset keyword forwarding; test CPU/CUDA comparison), retained in
private immutable jobs/ERRATUM. No old research artifact is reclassified.
A disposable full768-decision high-level entry test is currently running; it
uses no sealed parents and its updated model will not initialize formal training.
FB online collection, extended full model/six-Adam/replay/physical-prefix restore
are being completed. Formal counters remain zero until separately launched.

## Approved formal entry now validated (before formal launch)

See [engineering summary](dual_track_formal_artifacts/engineering.json) and
[new-runtime errata](DUAL_TRACK_FORMAL_ERRATUM.md). Formal counters still0 at
this snapshot; the queue will create new, independent initializations.

- High full768-decision disposable rollout/update:4 accepted actor updates,
  logprob maximum error0.0001220703125 < unchanged0.0003 assertion;
  critic completes18 updates after100 first-batch value-only updates.
  Mean legal-z changes0.04958degrees; same-state five-substep action/q_PD probes
  recorded. This is not evidence of useful contact learning.
- FB extended online collector:256 actual50Hz transitions in4.3683s in this
  short partial-episode sample. Not a full-run/update-inclusive training FPS.
- Full eager updates3.9363/.3419s; compiled updates58.0505/18.5217s still include
  cache/graph warmup. **No steady compiled update time is established from two
  calls**, and no formal1M ETA is claimed from these numbers.
- Full modified model/six-Adam/replay/normalizer and actual-action physical
  resume passed. Save+read+physical reconstruction53.3545s. Peak allocated
  15.917GB, reserved19.975GB. Entire successful engineering job262.3471s.
- Same-state64-row diagnostic actor-gradient norms:CPR184353.86,
  auxiliary6577.55,FB2366.01. These are diagnostic actual-z states, not the
  relabeled optimizer batch, and not causal proof that a coefficient should
  change. Approved coefficients remain fixed; watch actual learning/physics.
- Measured jobs including failures total2140.3241s; conservative carry/reserve
  remains7676.2031s. Common charged total9816.5272s (<32h). No failures erased.

The shared physical-time target now also discounts the five native rewards
inside a100ms high-level command; an analytic test proves equality with the
native20ms reward stream. Undiscounted physical costs remain separately saved.
No physical cost weight, force cap, anchor or PD is changed.

The formal serial schedule is two independent KL arms (98304 each), then the
fresh1M FB probe; original BC500/BFM remain frozen. Fixed-runtime detached
worktree and per-job ownership/resource guards protect the running source.
Student continuation remains conditional, not automatically launched by queue.

## Formal queue launch,2026-11-09 15:15UTC

Runtime pinned at147b22b684b0c03a613559fa6d5e5881d380ab7f in a separate detached
worktree. QueuePID380408; first KL.02 supervisor380634/worker380646 actually
running. Initial development evaluation is in progress; at this snapshot no
formal actor/critic training update or FB sample is claimed. Later status must
come from private per-job learning/selection/sidecar files, not this snapshot.
Both KL arms and1M FB have a fixed serial queue, each live-admitted separately.

Additional CPU course/provenance test:12000 generated episodes (not physical
experiences), fractions near.1935/mid.1945/long.25367/stop.20092/zero.104/none.05342;
2411 paired parents, identical pre-branch task parameters, equal hold lengths;
uint8 parent/time provenance round-trip exact. No RL or physics in this test.
Training sampler preserves approved episode marginals through paired macro-groups;
full pending-pair curriculum state is saved for resume. Reserved source/request
parents are rejected from train sampling. Provenance stays outside actor inputs.

Latest real smoke at147b22b validates actual wrapped truncation vs genuine
terminal as well as3 native sources, replay metadata and full history restore.
Original default native time limit is2250 controls=45s; this course fits within it.
On a real wrapped timeout, legacy execution used to mark native_termination;
the new entry clears that mistaken failure marker, keeps the true successor and
bootstraps without absorbing-fall tail. Formal training had not started when fixed.
No historical result is withdrawn without evidence of an actual timeout event.

The four-cell table remains partially empty: frozen privileged KL run has started;
mutable privileged1M is queued; historical restricted BC500 is retained; new
restricted Teacher migration/Student RL has not run. Missing cells are not failures.
