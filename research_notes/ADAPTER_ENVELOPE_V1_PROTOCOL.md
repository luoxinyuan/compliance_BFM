# Adapter Envelope V1 — implementation preflight blocked

Approval: Issue62 comment6827917070, 2026-11-12. This version records the approved constants and mechanical table; **it is not a completed physical qualification/launch lock**. No new qualification physics or training has run.

Approved nominal-anchored rule (no change): q0=g(clamp(base,-1,1)); margin=max(.02,.05*(qmax-qmin)); lo=qmin+margin; hi=qmax-margin; dminus=min(.12,.25*max(q0-lo,0)); dplus=min(.12,.25*max(hi-q0,0)); raw proposal includes original adapter and exploration; dq_safe=clip(g(raw)-q0,-dminus,dplus); q_requested=q0+dq_safe. A remains unmodified. B/C would use the same transform on current/target/noisy action, critic bootstrap and replay. Real q allowance .02rad, scientific criteria and original task/reward/PD all remain unchanged.

Original g is exactly `offset + clamp(action*5,-5,5)*scale/5`, not a reassociated affine expression. Domain [-1,1], all 29 scales positive, finite; mechanical margins ordered for all joints. The real-number map is invertible on this domain. This does not imply every separately rounded FP32 q_requested has a FP32 action preimage.

The approved exact inverse gate is tested before implementing this operator into any active Agent. CPU and NVIDIA eager checks use archived B19 inputs; no model sampling/rollout/training. For a genuinely constrained proposal, binary search over all ordered float32 actions finds two adjacent actions whose outputs strictly bracket q_requested. This proves no exact preimage in the existing dtype/operator contract. No tolerance, rounding repair, state exclusion or constants search is used.

The machine file `adapter_envelope_lock_v1.json` includes all 29 mechanical rows/constants, precise source hashes and counterexamples. Its status is BLOCKED. The full state manifest, full time/storage projection, five-path integration/gradient/recovery qualifications and subsequent B/C/dev/probe queue remain unexecuted. They cannot be claimed locked/passed. No conditional budget increase has been used.

See [actual preflight report](ADAPTER_ENVELOPE_V1_PREFLIGHT_RESULTS.md). Yi's comment §3 explicitly requires pause if inverse mapping cannot be verified exactly, and §9(1)/(3) covers this correctness/premise boundary. Any finite-precision executable definition needs a specific contract clarification before qualification; this does not reject the nominal-anchored method or establish physical failure.
