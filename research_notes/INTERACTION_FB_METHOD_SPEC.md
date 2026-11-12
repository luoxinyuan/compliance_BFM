# Issue62 R0 — Factorized interaction method v1

Approved scope: R0 design + T0 engineering trial only, user 2026-11-12. Base9fc9c6d; original BFM33f410c1… unchanged. This is a proposed residual successor-density model, not a demonstrated factorization benefit.

## Fields and authority

Robot O: original state, last_action, history_actor with original frozen normalizers; privileged_state is robot simulator state used by original B_R/F_R and training Q, excluded from U/adapter. Additional deployment87: known installation6, measured ideal force3 /5N, wrist-relative-root3 /1m, 25 true past force vectors75 /5N. Append actual previous high-level z256 /16; no counter, source ID, task clock, partner, future or active to deployment layers. All robot fields retain source checkpoint contract; ideal force remains unvalidated hardware proxy.

Training O_P adds privileged_state and current error3/.25m, relative velocity3/.5m/s, partner velocity3/.5m/s, physical active1. B_I input is successor robot state plus successor force/error/relative velocity/wrist-relative-root/partner velocity/active16. It does not receive reward, time, ID or future Task spec. Robot normalization unchanged; contact scales are fixed, clipped±10 by existing physical observation builder.

## Model and loss, executable T0 choice

Original B_R and F_R, Actor, targets and normalizers remain frozen. Six new two-hidden-layer128 tanh networks: B_I32, F_I32, U256, adapter29, scalar Q, robot-only discriminator. U/adapter final weights and biases zero; other layers standard initialization. Online raw robot Goal is original standing norm16; U adds a free256D correction, then norm16 projection. No force-to-skill rules. Guidance10Hz, spherical five-substep execution50Hz, physics200Hz unchanged. Adapter adds once in original normalized29D action space, clipped[-1,1], original PD unchanged. Native source has no adapter/noise; online Gaussian action noise std.05, absolute clip.3 then[-1,1], actual action stored. This is off-policy FB, not Gaussian PPO or on-policy data.

Interaction prompt is an explicit desired outcome prototype: original standing robot state plus zero force/error/relative velocity/partner velocity and desired connected=1; fixed at initialization, never current hidden state. z_I=target B_I(prototype), norm sqrt32, stop-gradient. It is an openly declared generic cooperative-follow goal across all T0 episodes, not a task-family answer. Later arbitrary interaction-goal inference remains unqualified.

Let b_R=B_R/16, f_R=mean(original F_R ensemble)/16, b_I=B_I/sqrt32, f_I=F_I/sqrt32. M=f_Rᵀb_R+f_Iᵀb_I. Target uses one true successor and target policy, gamma=.99^(dt/.1), zero only on genuine termination; release and truncation bootstrap. FB loss is half off-diagonal squared Bellman residual minus mean diagonal current occupancy, plus .1 whitened B_I Gram deviation from identity. The fixed robot component is a retained prior; new component is a signed residual, not an identifiable probabilistic partition. Cross-component Bellman errors remain in total M. No claim that subspaces are independent without later heldout probes/ablation.

Q learns actual unchanged integrated physical reward by TD MSE. Actor objective=-Q-.01(F_I·z_I/32)+.1 native q_PD-MSE/.01+.01 robot discriminator softplus. CPR discriminator uses actual native source vs online robot state/z, logistic classification, no physical active or contact fields. This is a new explicit logistic CPR-like preservation objective, NOT silently claimed identical to the archived GP CPR algorithm. No GP term inherited or removed from an existing run. Six fresh Adam lr1e-4; no scheduler (constant), no mixed precision/compile; all target networks Polyak.01 each update. B_R retention exact; native motion retention supervised only on actual new source rows. Reward, representation and preservation losses are separate.

T0 uses batch32, two complete update groups per250 real50Hz transitions, three rounds; engineering ratio differs from archived16/1024 and cannot be called formal Pilot recipe. Required nonzero finite gradients per all six groups; freeze reference originals. Per-row own command memory records substep index/previous/targetz only to reconstruct policy distribution, not a future task signal. At new high-level boundary U recomputes; interior native substep holds its actual issued target. Differentiable interpolation agrees with executed current-state contract.

## B/C comparability planned

A only U and independent Q; frozen base all original. B U+same adapter/Q/robot preservation, no B_I; replace interaction32 adapter input by fixed public constant so capacity can be matched. C adds B_I/F_I and explicit goal composition; incremental capacity/loss reported. B/C auxiliary goal bias and state normalization must be identical except declared interaction branch. Three-arm science not yet approved or run. T0 only implements C engineering path, no behavior comparison. Frozen F_R rather than additional robot-F learning is a deliberate minimal first version.

## Legacy reuse boundary

A dedicated child process owns exactly one Task. Read-only physics/source/normalizer/action implementations reused; entry explicitly records native contact installer, clean_env.advance Task override, backend prefix calibration and m10 driver callback. They cannot affect another process/runtime. No old high-level actors, searched Teacher labels, model updates or old replay used. New learner/checkpoint/replay lifecycle resides research/interaction_fb. F1 validity mask only accepts MuJoCo3.3.3/Newton/dense/island-off; raw arrays saved, proven NULL views excluded only from comparisons.
