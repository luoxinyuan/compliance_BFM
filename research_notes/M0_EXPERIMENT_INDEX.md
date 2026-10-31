# M0 experiment index

Milestone authorization: user requests autonomous tracks A/B and conditional C. No per-probe approval.

| ID | Purpose | Status |
|---|---|---|
| A-half | Frozen raised path alpha0.5 under original+Z30cm | completed; see milestones below |
| B-collect | Actual P0/expert dev rollouts and exact same-state expert labels | completed; see milestones below |
| B-fit | Representability; one paired1000-update alignment/latent-distillation aggregation | completed; see milestones below |
| B-final | Frozen final checkpoints, newonset3.55 and no-contact controls | completed once, final step1000 only |
| C | Conditional multi-behavior fit only if two useful qualified behaviors | not started: qualified second interaction teacher missing |

Budget oneGPU/8CPU/32GiB/8GiB VRAM/3600s cumulative. Owned PID cleanup only. Exact initial t3 standing state, physics200Hz/actorPD50Hz. Source/weights/oldcheckpoints immutable.

Architecture review: Issue3 comment6437697428 + Issue14 comment6458416428 and current Issue18 v3.14. Publicact no_grad, use frozen normalized _actor mean for gradient; physical action/qPD matching is essential. Independent expert local correction may provide action target without latent labels. Per-state z fitting is only feasibility and a direct-distillation comparator. LPP claim requires actual Projector rollout, expert requery on those states, update and new physical rollout. This is minimal FIBER-inspired LPP, not full FIBER architecture reproduction.

## Execution amendments before final evaluation

- Half probe completed; hold wristZ+38.48cm for+30cm target, hold3D9.488cm,100%5N with downwardFz-4.374N. Stop native-alpha calibration; no further alpha search. Existing bounded local expert remains TrackB source, not a claimed qualified+Z30cm expert.
- Collect completed:228decisions/1140controlframes; actual expert action parity<1.5e-8 and qPD<6.5e-8, including original50Hz release fade. Student inputs are465normalized proprio/action-history + last THREE XYZ force samples (9D),474total; manifest wording “plus3 measured forces” means three vectors, not three scalar features.
- Fit completed at fixed1000steps each, sameP0/samebatches. Frozen BFM unchanged. No second fitting round authorized by this protocol.
- Final onset3.55/release6.95 remain sealed.50Hz simulation completes its lasttick at8.96 rather than nominal8.95; recovery is evaluated at EXACT200Hz physical time8.95. This is grid rounding, not an extra tuning/recovery allowance.
- New read-only summary respects2s ramp (old generic script assumed4s). No physics/method changes. Sharedrepo'sorigin/main is available only in shared object database; a read-only ls-tree in private object database failed “Not a valid object name origin/main”, retried correct source. No fetch/merge/write involved.

## Milestone disposition (one aggregation round only)

A-half: completed; partial controllability but persistent downward5N hold resistance. No alpha scan. Asset audit m0_asset_audit.md.
B-collect: completed;4real branches + old expert,228decisions/1140frames. Exact expert requery verified.
B-fit: completed;160temporary updates per batch for15batches; representability passes. Alignment/distill each1000Projector updates, sameP0/schedule. Final hashes in fit-attempt-01/outputs/checkpoints-before-evaluation.json.
B-final: completed13branches; unseen onset3.55 and negative controls. Both learned models pass operational qualification, but alignment does not outperformP0 orA2. New onset full3D cm:standing2.0314,expert1.3884,P0 1.4081,alignment1.4277,distill1.4982,rule1.6037,A2 1.2316. Retain Y/Z regression and improved overall action continuity without claiming global superiority.
C: not run. No independently qualified new upward teacher. Effective local/forward behavior alone did not establish a needed second skill.
Stop: single-skill method loop qualified; multi-behavior milestone remains incomplete at teacher capability gate. No further training, latent scan, Bridge, background job or push.

Runtime commits6779140 /32f36cc /1ef17d4; four model/physics launches, no failures or reruns. GPU0 cumulative582.118s,616MiBownedVRAM,6.691GiBRSS,allownedPGIDs gone.18branches106.96s +54s prefixes=160.96s. Full evidence:m0_results.md and run/analysis/{half-summary,lpp-summary,evidence-audit}.json. SharedHEAD6023c6e remains clean; corehumanoidverse diff zero.

Read-only action/physics audit: qPD independent reconstruction<1e-6rad, actual force=wrench exactly, observed force=last completed physical frame exactly, all original timestep intervals and norm/oldradius/zero-gate contracts pass. Common stateSHA0060a963481d9f34dbd398d555b8abc18b2bec924dad5fecb710d56d37d5bb8d identical for18branches. Old/newforcehistory inspected against actual50Hz frames (no cadence change).

## Research interpretation correction — 2026-10-31

M0_RESEARCH_DIRECTION.md supersedes the earlier recommendation that a second upward expert is mandatory. The historical TrackA failure/TrackC nonexecution and all numerical results remain unchanged. Priority returns to SAME forward contact with different physical reachability/support states: arm→trunk/root→stepping only where justified. One unified queryable teacher may cover these responses; no required two-expert architecture or discrete selector. Audit credible existing body/support-transition capability before new qualification; fixed-foot forward goal does not establish stepping. This documentation-only correction starts no experiment and changes no preregistered protocol/checkpoint.
