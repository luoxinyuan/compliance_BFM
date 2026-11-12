# Issue62 R0 + R1-T0 results — complete, stopped

2026-11-12. User authorized R0+T0 only; [Issue62](https://github.com/GaoE05/robot-research-bridge/issues/62), startup6814319591. **T0-A/B/C all pass for this small eager engineering implementation. No behavioral qualification, A/B/C Pilot, new PPO or Student has run.** Original BC500, BFM/PD and historical conclusions unchanged.

## What was actually implemented

Independent `research/interaction_fb/` six-network path: frozen original robot B_R/F_R/actor/normalizers; train new32D B_I/F_I, deployable Guidance, zero-initialized29D execution adapter, physical Q and robot-only logistic preservation discriminator. Six fresh Adam; targets Polyak.01. This is a signed residual successor-density hypothesis, not proof of identifiable factorization. Mathematical losses/scales/permissions in [method](INTERACTION_FB_METHOD_SPEC.md); [data](INTERACTION_DATA_PROTOCOL.md); [ablations](INTERACTION_FB_ABLATION_PROTOCOL.md); [second mechanism](INTERACTION_SECOND_MECHANISM_DESIGN.md).

Deploy input is explicit proprio/action/history, known mount, ideal force/history, wrist relative to own root and actual own previous z. No partner truth, future path, task ID or clock. Public goal prototype fixed in T0; B_I never reads current hidden partner state to create an online actor prompt. Hidden-truth mutation invariance passed exactly. F/Q use actual own held command/substep memory and permitted current training privilege. Original robot normalizer retained; fixed contact scales. Six trainable modules total947,679 parameters (per-module counts in interface JSON). Original large F is frozen; these timings **are not full original FB finetuning throughput**.

## Three gates

**T0-A pass.** Canonical control04:3×250 actual50Hz transitions,2 full joint updates/round(batch32),750 transitions/6 updates. All six modules have finite nonzero real gradients, target/Adam updates, original frozen-weight guard unchanged. Real action and interpolated z agree with actual execution; five50Hz controls per10Hz command, four5ms physics ticks/control. No hard physics termination occurred. Existing immutable source/PD reused through an explicitly isolated single-Task child; process-local callback hooks listed in interface JSON, not hidden as a pure backend.

Actual coverage: one standing native source group, one completed8cm episode plus partial24cm episode.489/750 active-contact rows=65.2%;261 inactive rows;3000 online5ms rows. One true time truncation at row479 records its pre-reset physical successor, no terminated flag; all261 inactive rows preserve bootstrap. Release is not termination. Real hard-terminal path was not induced; source/code mask is `gamma^duration × not terminated`. Stop/continue/zero/none are configured but **not reached**; no claim of six-condition coverage or behavioral quality.

**T0-B pass.** Full round times20.095/34.171/21.346s include collector, source reset where present, six-optimizer groups and transaction IO.750/75.612=**9.919 transitions/s**,6/75.612=**0.07935 updates/s**. Only updates sum0.311s; checkpoint IO6.331s. Cold process198.485s→**3.779 transitions/s**,0.03023 updates/s; cold imports/model/assets/initial prefix count. CUDA peak allocated~800.8MB (not device utilization or reserved-memory claim), host process HWM~7.66GiB; shared disk remained>100GiB. round3 package46,861,391 bytes, entire replay included. Long-run IO grows with full snapshots; no linear large-run promise.

**T0-C pass.** Fresh process resume02 from control04 round1 rebuilt source/callback and replayed exact actually executed actions. F1 qualified MuJoCo3.3.3/Newton/dense/island-off valid-array and BFM-history checks passed before state/RNG restoration. Rounds2 and3: every model/target/normalizer/six-Adam tensor, sampled indices, replay/native rows, commands/actions, RNGs/counters, full5ms stream, final qpos/qvel/history match exactly; no tolerance relaxation. See [evidence](interaction_fb_t0_evidence.json). control runtime `e86dced8eda3ce5027fc8c0248858e656a3f4ff0`; resume/injection `e9f8a7b1510b8a79018e6fc9a64dc02742df1339` adds only signal-boundary handling/host metrics, numerical files unchanged.

Injected exception after15 new transitions in round2 produced separate uncommitted evidence, no round2 package; latest pointer still hash-verified immutable round1 with250 transitions/2 updates. Incoming package unchanged. The exact same package was used for successful two-round recovery. Normal SIGTERM/INT handler implemented (finish/commit current round); external signal and SIGKILL/power-loss were not separately injected. Sudden loss guarantees last committed round only, not volatile partial work. No compiled qualification.

Checkpoint full state includes all new models/targets/six Adam/normalizers/prompt/zbuffer/replay/native rows/indices, actual actions/commands, portable physics/history and callback reconstruction/spec identity, RNG/counters/runtime. Original frozen full BFM is an immutable SHA-verified dependency, not duplicated inside each small package. Per-round manifest + sidecar job ledger identify IO and cost; no completed flag before fsync/hash/readability/atomic publication.

## Identities, failure history and costs

Base SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`. Canonical initial full-state SHA `cdd17dcb50adad3aa1fe6ab06595fe1a31168fff88ac79b180cc4532b672faa4`; round3 full-state SHA `4709b2cae69230b729ecd5a4a62b0804285141b95ea1639c4300cf5e82bc992f`. These are engineering state bundles, not promoted behavioral models. Private index: stage `interaction-fb-r0t0-20261112`, runs/control04 is actually named `control-04`; jobs, checkpoints, raw failure evidence retained there.

| Job | Entire process s | All actual physics ticks | Exit |
|---|---:|---:|---:|
| control-01 | 106.368 | 1 | 1 |
| control-02 | 142.470 | 2401 | 1 |
| control-03 | 206.505 | 5201 | 0 |
| control-04 | 198.485 | 5201 | 0 |
| inject-01 | 162.846 | 3261 | 0 |
| resume-01 | 182.511 | 3201 | 1 |
| resume-02 | 210.525 | 6001 | 0 |

[Erratum](INTERACTION_FB_ERRATUM.md): missing layout sidecar; metadata tensor collation; nested restored tensor devices. All failures preserved, no old gate retroactively rewritten. control03 earlier successful loop retained but control04 is canonical full-stream test. All jobs: 25267 physics ticks,2265 newly executed online transitions,16 joint updates including repeated engineering controls/recovery; only750 unique canonical transitions. Reset/reconstruction ticks are not new training samples.

Measured experiment jobs 1209.711s; explicit conservative R0 preparation reservation3600s and T0 CPU/IO/publication1200s (not claimed measured stopwatch times). Stage charged 6009.711s; project cumulative 48.1510h/240h. Private unique-inode output before final reports 0.531GiB/8GiB. [Ledger](interaction_fb_t0_ledger.json), [round metrics](interaction_fb_t0_rounds.json), [interface](interaction_fb_t0_interface.json).

## Limits and next decision

Engineering success does not establish useful B_I, contact quality, stopping, composition or A/B/C advantage. No sealed parent/outcome accessed. Only a small eager single-environment loop is qualified. [Pilot proposal](PILOT_BUDGET_AND_SPLIT_LOCK.md) needs Yi approval plus its explicitly pending implementation/split locks; no automatic launch. All owned heavy jobs finished, single GPU released.
