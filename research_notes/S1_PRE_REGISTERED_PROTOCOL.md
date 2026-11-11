# S1 draft — matched Steering interface physics, NOT STARTED

2026-11-12; requires Yi confirmation under [Issue61 §5](https://github.com/GaoE05/robot-research-bridge/issues/61). No S1 allowance is consumed by this draft. No training or new search-Teacher data. Proposed first tranche: **original frozen backend only, ≤2 job hours, ≤2GiB output**. P-BFM download/port is a separate optional request, not included or automatically activated on failure.

## Question and fixed contracts

Can a causal, explicit forward-progress command control the existing frozen BFM's progress and stopping under the same wrist task, without changing contact/PD or accessing future partner trajectories? Goal is a fixed versioned forward-motion intent, not a fixed execution z. T may modulate z; it cannot claim to preserve exact goal/style merely because its bank ID is fixed.

Freeze original weight33f410c1…e361, original robot/PD/action map705fa1d4…b36dd, MuJoCo3.3.3/F1 solver,10/50/200Hz, wrist K80D8/10N, partner spec, reward and all scientific quality definitions. No waist/ankle actuator edits, force cap changes, reward scans or synthetic qpos. All conditions are previously exposed, never new sealed validation.

## States / strict matching

[s1_state_draft.json](s1_state_draft.json) locks six complete-state records and hashes from the F1 manifest, plus the existing continue branch of the stop parent: **7 windows /6 parent identities**, including one no-contact native source. P1-1 and P1-13 are distinct sustained parents; near5f62…, stop/continue5f638…, tail97ff… and native-standing are distinct controls. Preserve complete solver, source, BFM/history, prior action/z, force history and RNG. State timestamps are explicit (stop5.0s, observed branch5.5s); never rebuild by qpos alone.

First verify fingerprint and canonical BC500 replay. State-prefix callback/reset sequence is identical to F1-L3; all new worker adaptations are isolated processes. If exact matching fails, stop affected comparison and retain erratum; do not alter tolerance/masking. No different model state's memory may be copied into T. Its new own phase is initialized identically from the saved native source reference, or zero when no phase is present; this choice and resulting mismatch are logged, not treated as an expert phase.

## Controllers / permissions

Three controllers per window, same initial state and actual partner:

1. **Frozen BC500**: unchanged actual-action baseline, original input/normalizer.
2. **C-S admittance → T**: robot/ideal wrist-force history and known forward mode only. Fixed implicit update `v_next=(10*v+0.1*F_x)/(10+0.1*30)`, clipped[0,.22]m/s; mass10kg, damping30Ns/m, decision0.1s; v_init0. No force threshold, skill classification or future phase. These numbers are proposed for approval, not inferred optimal.
3. **C-P current partner-velocity → same T**: only current true partner velocity projected to the same yaw frame, clipped[0,.22], no future or final distance. This is an execution/Guidance diagnostic upper-information control, not deployable Guidance. It does not read anchor to optimize control, nor use a physical search.

T is the delivered transparent, bounded prototype: phase advances by requested velocity divided by the source nominal velocity/period; normalized interpolation of adjacent native motion latents, blended with standing by min(1,v/v_ref). Fixed bank and source SHA, no optimization. Source bank is71 rows, period1.42s, canonical z SHA `f3bb1f490aa5f90a4a78653e3f3f78af79d47ddadc92d502632deb8c5550e907`; full reference SHA `211a192ddf6e3d534a1ef273c22e9cf977510ccb235e1c88dc2f0b24e8d6f2c0`. Proposed v_ref=.214m/s is a fixed command-to-phase calibration motivated by historical native execution, **not a measured equality for this bank**; nominal source-motion speed metadata is.346828m/s. No post-outcome retuning. The stored cycle closure has19.84° joint and35.55° latent gaps; continuity risk must be measured, not assumed solved. See [source identity](s0_source_bank.json).

This is a deliberately falsifiable baseline. Zero phase speed or a standing blend may fail to stop; the bank may retain lateral/yaw drift. It is not learned U, native ERI or a final force-to-motion controller. No lateral/yaw dimensions are tested by this prototype. Existing B64/B128/v3/C-P/O-F windows are archival context only unless exact same state/partner permits direct comparison; do not pool mismatched scores as matched controls.

## Minimal execution plan and bounded counts

- Every one of7 windows ×3 controllers ×2 exact repeats =42 branches, each3s =126 simulated seconds,1,260 high-level decisions,6,300 BFM steps,25,200 physics ticks. Second repeat verifies determinism; not a new independent group.
- Add three **no-contact command qualification** branches from native-standing: valid velocity0/.09/.18m/s ×2 repeats ×3s =18 simulated seconds,180 decisions,900 BFM steps,3,600 physics ticks. This changes only the explicit command in an already no-contact source.
- For stop/continue/near, extend each controller's first repeat from the same ongoing actual trajectory through actual release+2s, maximum12s total per branch. Up to9 extensions ×9s =81 extra simulated seconds (≤810 decisions,4,050 BFM steps,16,200 physics ticks). The evaluator may know release; actor/Guidance may not. If release falls outside12s, mark recovery unobserved, do not move release or claim complete recovery.
- Hard total proposed ≤225 simulated seconds /2,250 high-level execution decisions /11,250 BFM steps /45,000 physics ticks. **RL decisions0**, labels for training0, search queries0. No candidate-pool ranking or latent scan.

The two repeat branches have identical command inputs and RNG until real history differs. For stop/continue, complete C-S input, command and physical prefix must be equal before the observed split. C-P has identical current velocity until the same split. Any earlier difference is a permission/state failure, not useful anticipation.

## Metrics and proposed decision rules (awaiting approval)

Retain all F1 physical boundary checks and original slip three-way semantics; no changed normal-force threshold. Report active-window XYZ force/error/RMS, saturation, root/wrist-relative-shoulder progress, phase and u, foot load/slip peak and duration, yaw, q/qdot/q_PD, support and post-release recovery. Anchor error after actual release is descriptive only. Keep original score components; **no checkpoint or controller selection by average return**.

Proposed operational qualification:

- Valid u=.18 versus0 increases sustained forward root speed by≥.03m/s on each of two non-same-parent demand states; ordering reproducible, not forced drag only (compare no-contact command control).
- At least two independent nontrivial parents meet their unchanged F1 task/force/slip/yaw references, including stop or continuation; no requirement to first fully extend arm or to never step.
- Near error/force ≤5% worse than matched BC500, and no new physical-boundary failure. Stop metrics measured over the actual hold window, not final root speed only. Guidance decrease must precede or accompany reduction of sustained root progress; residual slip/yaw remains explicit.
- F1 loaded slip peak≤.45m/s and |yaw|≤20° are reported unchanged alongside continuous quality; retained task-category bounds come from F1/M1 (local4cm/4N, middle8cm/7N, long10cm/6.5N, stop hold8cm/6.5N). Not every short window spans the whole task: window references and complete episode claims are separated.
- If progress is not controllable, conclude **T interface insufficient in this scope**, not BFM universally incapable. If progress responds but quality fails under load while no-contact works, record evidence compatible with targeted Contact Adaptation; do not automatically train it.

## Cost projection and remaining approval

L3 measured roughly17–19 wall seconds per8×2s candidate query implies~1.1 wall seconds per simulated second including query overhead; this is only a local projection, not guaranteed S1 throughput.225s simulation projects~250s hot physics, use2× contingency~500s; cold model/hash/reset/replay verification≤600s allowance, implementation/inspection/report/video≤5,500s, remaining600s reserve. **Proposed hard total7,200s**, not a claim all compute lasts2h. IO/output allowance2GiB; single GPU0, live100GiB storage guard. Re-measure first short branch; stop at hard budget if slower. No external backend download included.

Before physical launch Yi must confirm: this state/controller/metric set, proposed admittance constants and response threshold,2h cap, and the explicitly proposed .214m/s command-to-phase calibration (not a guaranteed physical speed). Optional separate P-BFM qualification would require explicit authorization for174,004,428-byte inference bundle, its own PD/depth/backend, and a new exact cost plan. Current S0 has not installed/downloaded it. S2/S3 remain unauthorized.
