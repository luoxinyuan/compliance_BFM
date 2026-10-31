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
