# M41 follow-up — standing startup causal diagnosis

2026-11-04; Issue #41. Completed frozen diagnostics, no new RL or Student training. Old A/B closeout and all failures remain valid. The main result is a history-dependent startup failure, not loss of all walking capacity.

## Decision and evidence strength

**Prioritize C (closed-loop state/history accumulation), with B (protect the effective startup feedback function) as the next intervention.** Early policy changes push the system into a poor startup region. After the initial policy has established the appropriate evolving state/history, the failed models can walk. Conversely, switching to initialization too late cannot recover this particular source prefix. These are real physical policy-swap interventions; they do not separately identify mechanical state versus executed-z/action history, nor establish which individual joint causes the failure.

A (course allocation) is a secondary hypothesis: actual stand_walk decisions are 14.96%/16.69%, not12%; only a small fraction of all decisions lies in its early-contact window. D (stochastic/mean mismatch) is present but not sufficient: stochastic failed models remain worse and highly variable. No unique optimization root cause is claimed.

## Fixed identity and protocol

Original training runtime `a8afaff087f535124d9e1b28be7ada57caa10bdf`; new diagnostic runtime `9e3771a8db036763152895748564bb1b302d597b`. Direct1088D privileged actor,256D raw Gaussian→norm16 z,10Hz high level/five50Hz BFM substeps/200Hz physics. Frozen BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`, original PD and5N wrist Tether unchanged. No online rule controller.

I = M39-A best64 SHA `72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a`.
A32 SHA `39ba9ee3a39020992358f3eefd1251fa79de7c372b771ae65f5be4477da433d5`.
B32 SHA `3ca49d5cecab308b6e405aa70a0aff6a7f1c35440867f8f2dd274e3c3c1329de`.
T30 SHA `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`.
All A16/32/48/64 and B16/32/48/64 paths, file SHA, embedded runtime/iteration and actual std are in identities.json. Both old selected best remain iteration0; final duplicate initialization runs are not new learning successes.

Task: original exposed development stand_walk, right wrist+X82cm, onset4.25s, ramp7.8s, release14.05s, horizon19.05s, standing source to4.0s. These are diagnostics, not new sealed generalization tests.I/A32/B32 full new qpos/qvel/force trajectories exactly match their original saved baselines (max difference0), confirming runtime parity.

41 complete episodes =4 mean baselines +24 full policy swaps +4 single100ms swaps +9 sampled executions. All had no safety boundary.

At each swap, the original Task instance continues without ANY reset or state assignment. Independently regenerated prefixes are checked exactly by section hashes: MuJoCo integration state including warmstart/applied loads, solver quantities, complete saved physical/control prefix, BFM observations/history, previous action, actual previous z/q_PD, force/contact histories, support, current input, control memory and source prefix.28/28 passed all sections. The code never substitutes a qpos frame for the live prefix. This verifies recorded full prefix/state consistency; it does not claim a generic serializer of every Python object.

## First divergence and same-state queries

Existing traces: raw differences begin at4.0s, before contact. A32/B32 actual joint differences exceed1° continuously for100ms around5.040/4.960s. First5N saturation occurs6.000/6.020s (I5.945s). These descriptive thresholds locate events; they are not success criteria or proof of causality. Saturation follows an already developing state difference, so it cannot alone explain the first divergence.

Compare commands on the **same** observed physical state/history and previous z:

| State prefix/time | Model vs I | Target-z angle | Mean RMS q_PD difference over five fixed-observation substeps |
|---|---|---:|---:|
|I /4.0s|A32|0.706°|0.073°|
|I /4.0s|B32|0.574°|0.028°|
|I /5.5s|A32|1.391°|0.072°|
|I /5.5s|B32|1.180°|0.057°|
|B32 /5.5s|B32|1.315°|0.050°|
|B32 /6.5s|B32|1.881°|0.147°|

The full target angle differs from the first interpolated executed z (~0.14/0.11°). Five-substep query holds the observation fixed; it is not five real physics steps. On distinct closed-loop trajectories, B32 vs I reaches~30° executed-z and~12.5° q_PD RMS difference at6.5s. That large difference mixes policy and different visited state/history; it is NOT a same-state causal action effect. Full joint/substep evidence is retained in physical-01/same-state.json and saved-audit.json.

## Physical causal swaps

Metric: all active-contact3D wrist error RMS cm / actual force RMS N. Free recovery is excluded from anchor tracking. I baseline5.344/3.295; A32 48.026/4.622; B32 50.031/4.615; T30 5.833/3.489.

| Switch time | A32→I | I→A32 | B32→I | I→B32 |
|---|---:|---:|---:|---:|
|4.0s|5.344/3.295|48.026/4.622|5.344/3.295|50.031/4.615|
|4.5s|5.035/3.196|50.819/4.627|10.520/4.185|30.649/4.616|
|5.0s|6.307/3.819|23.147/4.604|24.605/4.497|11.162/4.328|
|5.5s|12.756/3.814|9.225/4.024|50.076/4.617|7.753/3.696|
|6.0s|50.517/4.622|6.693/3.962|50.132/4.615|6.666/4.033|
|6.5s|48.405/4.622|5.231/3.234|50.135/4.615|5.342/3.503|

At6.5s I→A32/B32, root reaches79.79/80.66cm by14s. At6s A32/B32→I, it reaches only0.57/0.85cm. At5s B32→I eventually reaches90.32cm, but does not cross5cm forward until9.075s (I6.765s); late catch-up must not hide initial lag. For this prefix, B32 recovery loses effective startup between5.0–5.5s; A32 between5.5–6.0s. Earlier4.5s recovery is clearly better but B32 still has10.52cm error. These are observed windows, not a universal irrecoverability boundary.

Single100ms substitution at5.5s then returning to the prefix policy:
A32→I→A32 remains48.194cm; B32→I→B32 remains50.028cm.
I→A32→I gives6.075cm; I→B32→I gives5.214cm.
Thus a single tested command neither rescues failure nor reproduces severe failure; persistent feedback/state evolution matters. No conclusion about all possible single-command times is made.

I/A32/B32 cap fractions22.72/82.24/82.03%. I q_PD max20ms jump43.75°, actual q max5ms change2.05°; B32 q_PD2.42°, actualq0.28°. The smaller B32 command peak accompanies failure to walk, not superior motion quality. All phase forceXYZ/errorXYZ, root velocity, support, slip/tilt, target versus actual changes and free-recovery displacement are retained. Different sampling intervals are explicit; q_PD jumps are not actual joint motion.

## Exploration, reward and critic

Actual raw std=.008 for every coordinate; seeds4101/4102/4103 use matched epsilon sequences, same initial state/task. No parameters updated.

| Model | Mean error/force | Sampled error mean±sample SD cm | Sampled force mean±SD N |
|---|---:|---:|---:|
|I|5.344/3.295|7.939±0.859|3.889±0.292|
|A32|48.026/4.622|24.630±21.285|4.257±0.662|
|B32|50.031/4.615|23.388±21.056|4.344±0.280|

A32 samples10.45/49.11/14.33cm; B32 47.42/14.58/8.16cm. Noise can change which physical trajectory is entered, but does not provide reliable feedback or restore initial performance. Three seeds are a bounded diagnostic, not a statistical generalization claim; no seed is selected as the new model.

Reward recomputation: I return−7.497, A32−26.035, B32−26.615. Force costs6.380/12.555/12.519; tracking0.105/8.472/9.194; uncapped demand0.643/4.691/4.785. Continuity savings in stationary failures are tiny by comparison. The objective ranks this failure worse, not better.

At4s critic values I/A32/B32=−3.288/−3.171/−2.142, versus deterministic discounted returns−5.058/−13.934/−14.206. Sampled discounted returns average−6.714/−9.943/−9.790. Failure critics are optimistic even relative to this small sampled diagnostic, but these frozen evaluations are NOT training-time old-GAE labels. No saved per-batch obs/actions/advantages remain, so task-conditioned actor gradient cosine and exact task GAE decomposition cannot honestly be reconstructed. High aggregate EV does not establish startup credit accuracy.

## Actual curriculum/update audit

| Statistic | A | B |
|---|---:|---:|
|All decisions|49152|49152|
|stand_walk decisions|7352 (14.96%)|8204 (16.69%)|
|Completed stand_walk episodes|45|53|
|All released-recovery decisions|14717 (29.94%)|18693 (38.03%)|
|Completed-episode first2s-contact stand_walk decisions|900|1060|

The last row is reconstructed from logged actual handoff, condition and completed episode decision counts; it accounts for those completed episodes only. All completed episodes account for48345/48309 decisions; partial episodes are explicitly missing from that cross-tab. First2s contact is an operational time window, not detected stepping. Exact global family and phase marginals are preserved; their joint table was not logged for partial episodes.

Existing reliable retention contains T30/D2 source states, not a comprehensive set of this I policy's startup/failed states. Weighted replay/PPO actor-gradient norm median0.002751/0.002689 (~0.27%), range~0.20–0.36%; gradient cosines mixed. This motivates checking effective retention strength/coverage, not a causal proof of task conflict. Batch jointKL is bounded, but averageKL does not guarantee stable nonlinear startup. Full64 update diagnostics and preserved rollbacks remain in original logs.

## Force-range evidence reused, not rerun

Prior5/10/20N positive-combination error9.107/3.790/4.108cm, direct-start error5.665/6.533/6.542cm;20N also moves standing mechanically. New exact-prefix recoveries succeed under unchanged5N while failures can persist under the same cap. No additional cap matrix or force-range PPO was necessary. This does not prove5N is optimal for every future task, only rejects it as a sufficient explanation for the observed update-induced startup loss.

## Next matched training decision — proposal only, NOT launched

1. **Primary B+C test: protect the startup feedback function on its validated real state/history domain.** Both arms start I, preserve current architecture, normalization, raw action/std=.008,5N, reward/gamma/GAE, original B curriculum including stand_walk0.12 and fixed task seed. Fresh identical Adam; inherited identical critic.32×768=24576 decisions per arm, evaluate0/8/16/24/32. Control retains original reliable-domain raw replay. Treatment adds same-state raw-distribution retention to I on a fixed balanced startup bank: I actual4–6.5s histories and early A32/B32 prefixes with demonstrated effective I recovery. Use measured own history, no fabricated teacher prefix. Include failed late prefixes as evaluation, NOT I-correct supervised targets. This is one composite startup-preservation intervention, not separate proof of coverage versus coefficient.
2. Before optimization, calibrate its additional coefficient on one shared fixed batch to median startup-retention gradient/PPO-gradient ratio0.05; freeze coefficient thereafter. This is a proposed test setting, not an established optimum. Keep full five-substep q_PD diagnostics. Do not freeze legs/body, require tiny global MSE, or constrain failed combination regions to old answers. Report whether the extra constraint blocks gains in already-walking tasks.
3. Predeclare development startup onset4.13/4.25/4.37 and original walk/combination/zero/regression set; new untouched combinations e.g.onset4.41/ramp8.03/distance.84 and onset4.07/ramp7.63/distance.79 are proposed for sealing before any future run. Evaluate mean and three fixed-noise seeds, first2s/startup decisions and state-dependent takeover recovery. Preserve initial/best/last, select by multi-task physical quality with severe startup failure disclosed, never final-test reselection. Do not splice directions/candidates.
4. **Secondary A test only if the primary does not suffice:** same-budget original0.12 vs0.24 stand_walk probability, rescale all other probabilities proportionally; keep action/std/reward and retention fixed. Log actual family×phase×episode/version, values, advantages and sampled raw actions to compressed bounded rollout snapshots. This changes curriculum, not force. Do not simultaneously introduce a std schedule; current stochastic evidence warrants monitoring but does not justify treating noise as the sole fix.

No new training is authorized by this report itself. A later run needs its explicit bounded execution decision. The central claim remains shared state/contact-conditioned behavior, not motion-pair rules.

## Artifacts, reproduction and resources

Run root `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m41-startup-causal-20261104`.
Private runtime worktree `bridge-v2-owned/trees/m41-startup-causal/repo`, branch`gaoe05/m41-startup-causal`.
`physical-01/` includes41 genuine qpos trajectories, same-state queries, identity file,28 prefix comparisons and original model-integrity checks; no BFM update. `causal-summary.json`, `saved-audit.json`, `coverage-summary.json`, `completed-phase-coverage.json`, `historical-parity.json`, `ledger.json`, `videos/ledger.json` retain evidence. Source scripts `m41_causal_audit.py`, `m41_causal_stage.py`, `m41_causal_report.py`; protocol`m41_causal_protocol.json`.

Recompute existing results without physics: `python research_notes/m41_causal_audit.py <old-transition-run> <fresh-analysis-output>` and `python research_notes/m41_causal_report.py <new-run> physical-01`. The physical launcher command is captured in ledger; do not rerun into an existing output directory.

GPU6 physical evaluation973.987job-s; GPU5 saved-qpos rendering24.196job-s; aggregation28.187CPU-s, plus small read-only audits separately described. 6191 evaluation commands,123492 suffix physics steps plus8200 source-prefix and24600 standing warmup steps; these are physical evaluation costs, not RL training decisions. No RL or supervised updates; <1GiB stage outputs against3GiB cap. Total compute remains far below5000job-s, max2 concurrent GPUs. Both owned jobs exited0; GPU5/6 returned to1MiB, shared checkout clean at6023c6e257fab8fe58d13d9207fb5e85c32ee9c3. No other-user process, shared BFM/PD/weight, main, or runtime was modified.

Mac: `~/Projects/compliance-bfm-diagnostics/videos/m41-startup-causal/`, only2 MP4s+INDEX. Both15.08s,25fps,1× actual4.0–19.05s qpos (common standing warmup precedes video).01 contrasts B32→I vs I→B32 at5.5s;02 contrasts A32 failure vs early4.5s recovery. Side and overhead views, real force/anchor/contact/root track. SHA, full decode and representative visual frames checked. No best-seed resimulation for videos; they are the registered deterministic interventions, not newly trained models.
