# CURRENT_STATE — F1 closed; Issue61 S0 complete, S1 awaiting Yi

Last checked2026-11-12. Read [overview](RESEARCH_OVERVIEW.md), [F1 final](F1_RESULTS.md), [S0 audit](S0_PORTABILITY_AND_ARCHITECTURE.md), [S1 draft](S1_PRE_REGISTERED_PROTOCOL.md), [Issue61](https://github.com/GaoE05/robot-research-bridge/issues/61), [Master18](https://github.com/GaoE05/robot-research-bridge/issues/18).

## Completed / frozen conclusions

F1 L0–L3 complete, Issue60 closed. L3 matched3 windows/2 parents: future O-F improved stop6.205→5.766cm,5.076→4.702N; continue10.085→9.344cm but6.960→7.026N; tail-hold commands/physics exactly unchanged. Continue/tail still fail joint quality; future knowledge alone does not remove the problem. C-S planner remains a missing cell, not evidence of observation aliasing. Runtime61b4371; final report commit a7cba60. BC500 unchanged; F-RL-v2/v3 not promoted. No training/sealed-data use.

S0 source/config/interface audit complete; no new physics or training. Original wrapper42 calls/21 archived substeps exactly reproduces action/q_PD, runtime16e103e. Final helper/config audit168a91bf1446dec7346a9fff725e28d74fef0030:8 contract tests pass,32 P velocity cases exact; I sphere-antipode helper fails original1e-5 criterion (error3.5731e-5), preserved in [S0 erratum](S0_ERRATUM.md). This is not a physical I-BFM failure.

## Steering decision and boundaries

Continuous velocity/progress + validity, later heading, is preferred. Original BFM has no native ERI: delivered T is a legal but physically unqualified progress/bank modulation reference. Source bank71 rows/1.42s includes historical offline arm/waist corrections, no old actor online. Goal identity does not guarantee style preservation.

P-BFM has real native ERI and public~166MiB inference bundle metadata, but depth/history, wrist PD/action scale and simulator differ. No weights downloaded. I-BFM model~3.166GiB has LOGO current-B/privileged-object dependency and different PD, also not downloaded. Both are source-qualified candidates, not locally physically validated replacements. [Three-backend table](S0_PORTABILITY_AND_ARCHITECTURE.md).

S1 proposal:7 exposed windows/6 parents, same original physics, BC500 vs C-S admittance→T vs current-velocity C-P→T; no-contact command controls and release extensions. ≤2h/2GiB, no training; concrete parameters/counts/hashes in draft. **Not authorized or started.** Await Yi confirmation; no automatic P download, S2/S3, Student, FB or PPO. Contact Adaptation necessity remains unproven; first separate progress-interface failure from contact execution-quality failure.

## Exact identities / costs / handoff

Frozen BFM SHA33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361. Old BC500/other candidates: [unchanged model manifest](f1_model_identities.json). Original action map705fa1d4…b36dd; source/state hashes in [S0 bank](s0_source_bank.json), [S1 draft states](s1_state_draft.json). F1 L3 cost3427.130s; completeF1 14799.703s/12h. S0 cost is separate [ledger](s0_cost_ledger.json), within2h; no runtime currently scheduled by this milestone.

F1 full real-qpos1× videos verified on Mac at `~/Projects/compliance-bfm-diagnostics/videos/f1-l3-20261112/`; see [video index](F1_VIDEO_INDEX.md). S0 has no new physical video because no new physics ran.

Keep originalBC500, T30/S0/M38 and historical successful ideal-force transfer, failures and errata. New code is `research/interaction_steering/`, no old research_notes import or training autorun. Physical adapter integration still needs S1 qualification; no qpos-only reset. Original MuJoCo recovery proof is solver-specific, not transferable to external backends.

Start next session from this state/Issue61, verify live GPU0/storage≥100GiB and approved scope. Only personal research ordinary fast-forward; no main/sharedBFM/PD/other tasks. Library2026-10-28 files are historical snapshots, not live state. Previous state preserved as CURRENT_STATE_S0_PREVIOUS.md.
