# S1 APPROVED execution supplement

User approval2026-11-12: “可以，按协议继续推进”. The exact draft at4df6b0763b6e871ee28ed775ac267c6bf4962386 is approved, retained unchanged.

All listed constants, 7 windows/6 parents, ≤7200s/2GiB and45,000 experimental physics ticks apply. Source-reconstruction/reset ticks are separately counted as correctness overhead and included in job time; no added scientific states. No S2/S3/download approval.

Implementation boundary: `legacy_worker` is a dedicated process reusing archived F1 reset/solver/physics code. Its imports/configuration never enter the pure Guidance/executor package or other processes. Historical reset order is retained to reproduce exact complete states; unselected archived prefixes are qualification overhead, not additional candidate experiments.

Known coverage limitation: draft fixed-.18 vs0 test is allocated only to native-standing, whereas its stronger response criterion names two demand parents. Do not expand the approved branches to fill that gap; report that clause unproven if no matched evidence.

C-S force history:25 samples from preceding20ms intervals, current measured force replaces the latest sample; all rotated by current yaw. Causal action/proprio history uses last4 actual BFM control records. Guidance consumes only latest force and own velocity. C-P reads only currently exposed partner velocity (the physics-sensor sampling timestamp is retained); no spec/future access. T own phase starts0 because these BC500/native-standing states do not carry this new controller's phase. All controls use original five-substep interpolation from actual previous-z.
