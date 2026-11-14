# G17-QA1 immutable execution protocol

Yi approved the source amendment in the current conversation on 2026-11-15 (Asia/Shanghai). Source SHA256: `726d2baf5160b0e973a8ef15729230385c0727477804d69248470718cea53dc9`. The archived source's REVIEW ONLY status is superseded solely by the separate signoff; historical FAIL is unchanged.

## Fixed experiment

- E round14 / Q56 / 3500 actual rows, manifest `2068e10d41e6cace8dfc5bef840e444562ad395af2767b06d996fb911bcc21de`.
- U seed636413; C tie seed636414. Fixed 456 batches64 per branch. Total nodes56/128/256/512. Original `ApprovedLearner.q_update` reused; only its batch-index provider changes. LR1e-4, EMA.01, original actual reward/duration/termination, actual action/z/zi remain unchanged.
- Physical phases derive from each actual row's four archived5ms ticks: disabled connection; inactive before actual onset; inactive after actual onset; mixed transition; holding speed≤1e-8m/s; decreasing speed; remaining moving. The tolerance is only metadata classification. Parent×phase quotas match U slot-by-slot, including historical56 exposures in C counts. Stop/continue share their true parent; no additional family balancing.
- Schedule, original agent/Adams/targets/replay digests, parent/phase map and four prospective specifications are locked in `r2_ei_qa1_schedule_lock.json`; exact indices in the hash-addressed private schedule packet. Schedule fixtures include quota identity, least-exposure choice, RNG reproduction and corrupted-source/stratum rejection.
- I paused. Actor/U/Goal and all their targets, original base/normalizer/PD frozen. No new main rollout or expert data; old40 prediction branches never enter replay. No Dev/Sealed reads.

## Target semantics and endpoint gate

Before any optimizer step, compare actual frozen target mean with every archived physical mean continuation at20ms and100ms boundaries, and write per20ms r/d/Q backups and1s n-step residuals. Skip the deliberately intervened first action in non-mean branches when comparing policy outputs. A difference is reported, never hidden by changing tolerance.

Hidden-future opposite-optimum evidence requires genuinely identical complete past under two futures. If absent from archives, mark UNMEASURED. Nonpositive deterministic pathwise reward provides a prefix value upper bound only conditional on that exact future; it does not by itself prove a conditional-expectation Q bug.

Old40 keep exactly58 comparisons≥1e-4 /four effective parents /70%. Nodes128/256 are descriptive;512 is the fixed endpoint. Additionally report actual Q versus realized negative1s prefix on mean mid-hold round5 and stop-hold round9. As a conservative prospective trigger, require both registered mean values no more optimistic than their realized prefixes; otherwise calibration is not cleared. This check grants no standalone mathematical Q qualification and cannot relax the70% gate. No arbitrary new numerical tolerance is introduced.

Only if endpoint ranking and calibration clear, use the four prelocked unseen-in-replay train parents, five first20ms actions each, shared frozen mean to1s. Candidate draws use the oldstd.01/clip.06; all source/prefix/solver/history retained. No prospective data enters training. Even a prospective PASS authorizes no Actor/U/I continuation.

## Storage, identity and budget projection

Dedicated stage `r2-g17-qa1-20261115`; original replay/state remain immutable references. Eight full Q-agent packets at nodes56/128/256/512 (two branches) retain Q/targets/Adams, frozen-function identity, external replay/physical references, schedule cursor and all RNG. Each packet is hash-checked and read back before commit. No original checkpoint is overwritten.

Prior R2 cost15241.192158s /last2.372GiB. QA1 subcap3600 job seconds,1GiB new output,18000 full-flow5ms ticks, all inside R2≤57600s/12GiB and project240h. Initial CPU/source/archive/preparation upper charge600s, including the preserved empty-round0 inspection IndexError (no update/physics). GPU main job limit1500s; publication reserve300s; conditional prospective reserve≤1000s, actual remaining budget checked first. CPU schedule job measured9.894s is included in600s, not double-counted.

Prior cold startup150–167s. Two456-group Q-only branches have no new sampling; allow≤600s updates/audits and≤400s packet IO, within1500s supervisor. Required completed snapshots estimated below512MiB; monitor actual per-packet growth. Prospective four full prefixes plus20×1s cost approximately9000–12000 ticks; startup/source reset allowance remains within18000 and is checked against actual counters. These are conservative projections, not achieved throughput.

Physical counter uses actual completed plus unexecuted reserved ticks: `reserved` is cumulative committed requests and already includes completed `actual`; they must not be added twice. Both values are reported. No physics budget is silently enlarged.

Single live-verified GPU0; shared diskfree≥100GiB. The owned supervisor stops only its exact process group. Any invalid hash/RNG/optimizer/quota/save/finite-Q/hardphysics/budget error preserves first evidence, allows at most one same-input read-only diagnosis, and stops pending decision. No automatic redesign or rerun of optimizer experiment. Prelaunch projection or guard failure also stops.
