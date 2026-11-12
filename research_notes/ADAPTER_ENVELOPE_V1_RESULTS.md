# Adapter Envelope V1 — numerical repair succeeds; frozen physical qualification fails

## Decision

**STOPPED under Issue62 comment6827917070 §4/§9. No B/C retraining, no further qualification branches, no five-way dev/probe.** Yi's inward numerical amendment6828464806 resolves representation only; it does not waive slip/physical criteria. A128, B19, BC500 and all historical failures remain unchanged. No sealed conditions read.

## Locked execution and identities

Physical runtime `e4c2d3ff08fc0a29152dc7b1f0ee00201b077012`; source/state/threshold lock committed and pushed BEFORE new physics. Reports: [protocol](ADAPTER_ENVELOPE_V1_PROTOCOL.md), [full time projection](ADAPTER_ENVELOPE_V1_TIME_PROJECTION.md), [full storage projection](ADAPTER_ENVELOPE_V1_STORAGE_PROJECTION.md). Old strict inverse blocker c13cc69 remains historical, not retroactively passed.

Frozen B19 core SHA `0cd3e3684d764cad42caaa83773b885868caeed0557dba1780ec36a3d03a3984`; agent/targets/Adam digest `54002fb0161de37bff3e2e00c781b216e8fc294e829bd88b60cf053a68222f79`. Original BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361` unchanged. Parent `8dde57eac4325d7701fafe97d48f57cb962cd8ff121a9e4770a1bbdffa10db4e`, exact archived time11.8s. This parent released contact10.8s; external force stays0 during the comparison. No released-anchor tracking score is used.

Full canonical MuJoCo arrays, BFM history and RNG match saved checkpoint. Against archived off sample: identical model/RNG/task/first BFM observation/first actual interpolatedz. Exact executedaction==replayaction==controlaction, and original g(action)==physical q_PD assertions passed for all150controls. Actual sample trace SHA `2fb2feacb607e4c73227daeb69d4a5790174a132beb9703d070d071819e4a95b`. Complete3s interval11.8→14.8,600×5ms ticks; not compared with original on's0.515s failure as if equal length.

## Physical gate (same 3s, sampled execution)

| Metric | Archived adapter-off | New Envelope | Frozen rule / result |
|---|---:|---:|---|
| Actual joint excess max, rad | 0.000000 | 0.000000 | B19 must be0: pass for sample only |
| Loaded slip peak, m/s | 0.606027472 | 0.919526089 | ≤0.727232967: **fail** |
| Loaded slip p95, m/s | 0.021672103 | 0.150267713 | ≤0.041672103: **fail** |
| No-support ticks | 0 | 0 | no additional: pass |
| Minimum root height, m | 0.724670853 | 0.724363616 | no more than.03m loss: pass |
| Final world pelvis yaw, deg | -164.857148 | 96.831128 | absolute-yaw relative worsening rule passes; **both fail absolute20° reference** |
| Root XY displacement in window, m | 0.095695 | 0.430761 | descriptive recovery metric, not new gate |
| Final root speed, m/s | 0.004686 | 0.202657 | descriptive recovery metric |
| Actual external force max, N | 0 | 0 | release: no anchor-follow score |

Neither branch passes the original absolute.45m/s slip reference. World-yaw absolute score must not be mistaken for small actual rotation or good recovery; the sign changes substantially. We do not redefine yaw or relax a gate after seeing this result. Root travel/speed also show continuing post-release motion; these are supporting descriptive observations, not substituted success criteria.

Envelope correction median RMS29=0.10269844rad, maxabs=0.12000000rad, per-joint trigger fraction=74.643678%. This release window is NOT evidence for the two-active-parent adaptation gate; those branches were not run. Avoid saying the entire adapter was zeroed or all effective feedback retained.

## Numerical amendment and engineering failure are separate

Offline CPU/CUDA test1024×29 entries: max inward representational correction4.76837158203125e-7rad; actual execution stays within original bounds, zeroresidual output exact, gradients nonzero outside genuine clamp saturation. Actual physical trace maxnumeric correction `2.384185791015625e-07`rad. Physics/replay use actual values; no allclose replaces original exact checks. All four constants, std.01/clip.06, reward, PD, forcecap and actualq.02rad hardboundary unchanged. First nextafter-only implementation failed on offset cancellation and is preserved under prep/envelope_attempt01.py/jobs/rounded-operator.log; finite ordered-float bisection fixes only representation.

The qualification job then failed in **test harness gradient context**, after full sample physics and exact execution/replay/qPD checks, before writing qualification.json: outer m10 physical driver uses no_grad, and disposable audit failed to re-enable it before Qloss.backward. Trace was already saved. No completed optimizer step; no original model updated. Failed job/stack retained; posthoc evidence JSON is explicitly reconstructed from existing trace and frozen checkpoint, not fabricated original job success.

Minimal harness correction now explicitly uses torch.enable_grad for the disposable update and writes physical-only summary before gradient audit. Static compilation passes; **full update path regression remains unverified**, because the independent physical slip gate already requires stopping. Current/target/critic gradient audit must not be claimed complete. A future approved runtime must relock changed source hashes. Old failed runtime remains e4c2d3f; no rerun is hidden in these numbers.

## Missing cells and evidence grade

- **Confirmed:** deterministic inward representation within≤1e-6rad; same-state sample branch eliminates prior actual joint excess over3s; new slip peak/p95 fail locked off-relative gates; full frozen physical/control/replay consistency verified.
- **Compatible, not established as unique mechanism:** bounding joint targets alone is insufficient to preserve post-release coordination; large distributed residuals remain. No new cap/margin/gain sweep.
- **Unproven / not run:** B19mean; near/mid active effectiveness; no-contact recovery qualification; complete differentiable path audit; B/Cfresh128rounds; five-waydev; BIincrement. No claim new package learns better or is ready to retrain.

The single next scientific decision is whether to redesign execution-adapter behaviour retention/optimization under a new matched contract. Further floating-point edits cannot repair this measured physical tail; no automatic secondEnvelope or loss is tried.

## Cost and retained assets

Before physics the complete forecast was17.9636h/12.7414GiB; conditional18h/16GiB authorization recorded, no historical reset. Actual failed process `277.474380s`, including coldload, prefix/replay, branch, IO and attempted audit. Physicalcounter7641 actual=reserved ticks (7041 reset/prefix/replay,600 qualification);150 frozen50Hztransitions; formalRL/Studenttraining0; completednewAdamsteps0. Source-state prior B19updates38 unchanged.

Cumulative charged upper bound **10.279462h /18h** = inherited34028.588967 + preparation/offlineanalysis upper1800 + job277.474380 + read-onlyanalysis/report/publication upper900. Measured output approximately **3.434030GiB /16GiB**, includes historical nine-stage inode-dedup plus root ledgers/scripts; not just current output. Sharedfree at final read364.383GiB≥100GiB. GPU0 released; no owned process running. Nothing deleted.

Stage `experiments/interaction-envelope-v1-20261112/`: jobs/qualification-B19-01.log and qualification-job-qualification-B19-01.json (failed exit1), runs/qualification-B19-01/envelope-sample/qualification-trace.pt, qualification-B19-01-partial-evidence.json, qualification-B19-01-gate.json, ledger-envelope-final.json. Public compact evidence under [adapter_envelope_artifacts](adapter_envelope_artifacts/); raw trajectories/checkpoints stay private on server.
