# Formal E28k restore interruption: first difference and bounded repair

2026-11-16. User authorized repair and continuation; all scientific gates remain.

## Actual formal state

E attempt05 reached28000 real50Hz transitions. Last committed optimizer block ended27360,
107 accepted Actor updates. 640 pending rows are unchanged on-policy data; the next original
block ends28384. P2/P3 first-update and4k/8k/12k/20k protections passed;28k protection was
NOT executed because restoring its qualification source failed. I not started. Original
progress.json RUNNING is stale after the nonzero exit; formal-stop.json/sidecar take priority.
No claim of active task success, no Dev/Sealed evaluation.

## Strict first difference

Frozen new-process node28 diagnostic fails on efc_J_rowadr and efc_J_rownnz only.
Shapes both61 ints; different elements57/51; max integer difference34/64 respectively.
Original assertion remains. The diagnostic also shows the training node itself encounters
this comparison; it is not evidence of a learned policy's physical failure.
Local MuJoCo3.3.3 mjdata.h:391–392 declares row nonzero counts / colind row addresses,
each nefc×1. Current configuration Newton+dense, no islands; this does not by itself prove
that changing them is harmless. Original three replay-restored index arrays omitted these two.

## Physical qualification before changing production restore

Independent frozen intervention checks each array separately: archived original, zero,
and integer perturbation within archived range, three real qualification states (none,
local FOLLOW and actual stop crossing), at least3s per branch. Same full state/RNG/controls;
require exact physical rows, actual controls, scores and continuation RNG. At least one
state must have real partner force and nonzero constraints. Any physical difference fails
this proposed repair. No optimizer updates or formal PPO rows; all costs/ticks charged.

If qualified, include both archived arrays in exact restoration, without relaxing or masking
any comparison. All effective arrays including both copied arrays must match at final restore.
Keep raw artifacts, failing assertions and logs. Rerun fullnode28 protection before continuation.
Resume schedule CPU check preserves all32 original optimizer block boundaries; no redoing P2,
no repeating107 updates, no extra samples to replace640 pending rows.

Status at this draft: intervention IN_PROGRESS, no production change/through-gate claim.

## Preserved intervention-entry failure

Intervention01 stopped at the frozen-source agent digest: OR1Agent adds OR1_counts
to state(), whereas the trusted frozen fixture is ContactGoalAgent. No intervention
rollout or training occurred. Cost142.447377s/2441ticks retained. Intervention02 uses
the exact original frozen reader class and empty-optimizer counter schema; this does
not modify production Actor, data or scientific gates. CPU patch quoting failure
produced no remote action and is retained in orchestration history.

## Actual intervention PASS and restoration rule revision

Intervention02:380.500490job seconds,14141 realphysics ticks, zero optimizer/formal rows.
Three full real states, five branches per state, each3s. Cases0/4/8 had actual partner
force max0/4.532877/10N; constraints were formed in all three. Each array separately
zeroed and randomized within archived range. Physics, control records, scoring and
continuation RNG hashes match original exactly for all branches. See machine summary.
This is evidence only for the current fixed dense/Newton configuration and these states.

Comparison-rule change: NO tolerance or canonical mask change. Restore the archived
bytes for efc_J_rowadr and efc_J_rownnz alongside the existing three replay-restored
Jacobian index workspaces. Pre-copy replay comparison still requires every other
canonical array exact; final restored full canonical state includes both arrays and
must be exact. The arrays are not deleted or asserted zero-valid-length: each is
nefc entries, pointer-backed, associated with sparse Jacobian row indexing. Local
header evidence alone was not treated as a dynamics proof. Any changed simulator
configuration continues to fail closed in existing metadata/canonical checks.

Resume source remains fullnode28000; model107 accepted updates equals last committed
P3 block15. Rebuild complete state and two12-row sampled continuations before collecting.
Rerun actualnode28 protection, then preserve pending640 rows and target28384. Original
32-block/32k schedule unchanged. The two12-row probes are not new PPO samples.
