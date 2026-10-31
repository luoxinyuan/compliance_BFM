# M0 experiment index

Milestone authorization: user requests autonomous tracks A/B and conditional C. No per-probe approval.

| ID | Purpose | Status |
|---|---|---|
| A-half | Frozen raised path alpha0.5 under original+Z30cm | preregistered |
| B-collect | Actual P0/expert dev rollouts and exact same-state expert labels | preregistered |
| B-fit | Representability; one paired1000-update alignment/latent-distillation aggregation | preregistered |
| B-final | Frozen final checkpoints, newonset3.55 and no-contact controls | sealed |
| C | Conditional multi-behavior fit only if two useful qualified behaviors | gated |

Budget oneGPU/8CPU/32GiB/8GiB VRAM/3600s cumulative. Owned PID cleanup only. Exact initial t3 standing state, physics200Hz/actorPD50Hz. Source/weights/oldcheckpoints immutable.

Architecture review: Issue3 comment6437697428 + Issue14 comment6458416428 and current Issue18 v3.14. Publicact no_grad, use frozen normalized _actor mean for gradient; physical action/qPD matching is essential. Independent expert local correction may provide action target without latent labels. Per-state z fitting is only feasibility and a direct-distillation comparator. LPP claim requires actual Projector rollout, expert requery on those states, update and new physical rollout. This is minimal FIBER-inspired LPP, not full FIBER architecture reproduction.

## Execution amendments before final evaluation

- Half probe completed; hold wristZ+38.48cm for+30cm target, hold3D9.488cm,100%5N with downwardFz-4.374N. Stop native-alpha calibration; no further alpha search. Existing bounded local expert remains TrackB source, not a claimed qualified+Z30cm expert.
- Collect completed:228decisions/1140controlframes; actual expert action parity<1.5e-8 and qPD<6.5e-8, including original50Hz release fade. Student inputs are465normalized proprio/action-history + last THREE XYZ force samples (9D),474total; manifest wording “plus3 measured forces” means three vectors, not three scalar features.
- Fit completed at fixed1000steps each, sameP0/samebatches. Frozen BFM unchanged. No second fitting round authorized by this protocol.
- Final onset3.55/release6.95 remain sealed.50Hz simulation completes its lasttick at8.96 rather than nominal8.95; recovery is evaluated at EXACT200Hz physical time8.95. This is grid rounding, not an extra tuning/recovery allowance.
- New read-only summary respects2s ramp (old generic script assumed4s). No physics/method changes. Sharedrepo'sorigin/main is available only in shared object database; a read-only ls-tree in private object database failed “Not a valid object name origin/main”, retried correct source. No fetch/merge/write involved.
