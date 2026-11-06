# Issue48: native capability priors → shared contact feedback

This is a new authorized stage, not Issue47 continuation. New actors, critic, normalizer and optimizer never inherit I/C0/T30 weights. They are independent final-evaluation references only. The frozen original BFM actor/normalizer/backward-map and PD remain unchanged.

## Sources and initialization

At most six logical candidates: verified standing; same-checkpoint 5/15cm goal-derived reaches; one goal-derived modest waist-pitch/reach pose; traced native forward motion (`task` variant from M10 offline reference correction, not M10 Student); old reward `move-ego-0-0.3`, with its published drift/jump failure retained. Goal target construction and motion arm/waist edits are priors, not force-only learning or a guarantee of natural execution. The trusted replay contract and original reward inference are reused, not rerun with external contact rewards.

Representative real source episodes provide optional **continuation initialization**: stable holds and actual ongoing motion. Clock-selected entry/exit frames are excluded from BC, because a hidden source schedule can prescribe different next actions at the identical initial observation. This does not prove every nearby label identifiable. Entering a behavior in response to contact is subsequently learned from physical returns; no online source controller replaces the PPO actions.

## Exact exploration comparison

Both arms have the same fresh 1088→128→128→256 actor, native-initialized raw mean `mθ(x)`, full256D freedom, normalization, curriculum and initial weights.

A executes `raw = mθ(x) + .008 ε`, with ε standard256D Gaussian.

B executes `raw = mθ(x) + L ε`, where `LLᵀ = .008² I + .15² DᵀD`. Four orthonormal rows of D are constructed once by Gram-Schmidt from the standing-tangent directions of goal_arm5, goal_arm15, goal_lean30 and the mean traced walking command. The originally specified unweighted SVD was rejected before the guided arm started:71motion rows dominated3static rows and captured only9–13% of static tangent directions. See native_balance_directions.py and the private input-version manifest; old basis bytes are retained. D is computed once, independent of current force, future task, labels or old high-level outputs. This is **native-direction covariance exploration**, not a learned planner, skill selector, mixture override or a temporally correlated proposal. Random samples can still be poor or temporally inconsistent; actual safety and deployment mean quality are reported.

For compatibility with the existing diagonal-Normal PPO implementation, store/sample `u ~ N(L⁻¹mθ(x), I)` and execute `raw = L u` **once**, then original `z =16 raw/||raw||`. L is fixed/invertible per arm. `log p_raw(raw)=log p_u(u)-log|detL|`; the constant cancels from ratios and KL. The original analytic joint-Gaussian KL, clipping,3e-4 pre-logprob and.02 accepted-KL thresholds, Adam transactional rollback and actor gradients remain valid in u space. Old logprob records the actual sampled u. The executed policy is not retroactively relabeled as an unshifted Gaussian. Actual raw std/norm, legal z and q_PD are reported separately; `log_std=0` in u is not zero physical exploration.

At deterministic evaluation, both use `raw=mθ(x)`, without a candidate query, old actor or exploration sample. This comparison changes exploration covariance only; both receive native initialization and the same transition curriculum, so cross-stage improvements cannot be attributed solely to guided exploration.

## Transition MDP and evaluation

Source prefixes physically execute standing, qualified reach or native walking in the **same MuJoCo instance**, at variable source stages. At takeover no qpos/history/solver reset occurs; integration state, BFM observation/action/last-z and contact history are saved. Source identity, future distance, globalclock and switch times are curriculum/evaluation metadata only. Teacher features use current privileged partner error/relative velocity, actual per-point forces and proprio/action history. Future Student needs separate restricted-observation training; ideal force is not a hardware claim.

Self-hold, local, expanding/long demand, stop/continue, released recovery and moving-source demand changes are present from the first version. K80D8,total10N,5N regressions, original geometry reward/GAE/controls are fixed. Native execution and source-prefix costs are separate from actual on-policy decisions. Candidate qualification and BC do not count as PPO samples.

Selection includes iteration0 and uses fixed multi-family development physical costs, worst-task protection and safety. New final combinations are sealed until selection. Results report continuous task utility and motion, not latent distance or root displacement alone. Released anchor error is descriptive, not follow success.


## Scope of causal claims
The guided arm adds variance as well as changing covariance orientation. It also changes the raw-coordinate probability/trust-region metric. Therefore this first contrast does not isolate direction from total exploration energy or optimization geometry. A future trace-matched isotropic control would be needed for that claim. Training samples stop/continue from matching parameter distributions; strict same-physical-prefix/same-checkpoint paired equality is measured in the sealed evaluation, not asserted for every independently randomized stochastic training episode.


### Pre-guided source-balance correction
The ordinary arm is unchanged: it never reads the direction array. Before any guided rollout, a single deterministic source-balanced basis replaced the unweighted basis, with rank4/std.15/isotropic floor.008/total covariance trace unchanged. No candidate/reward/seed scan, no old high-level outputs and no heldout results were used. New fitted-real-input jointlogprob max1.220703125e-4<3e-4; mean whitening inverse max6.41e-7. Original and new basis files are retained with SHA256; runtimec1a8cc3 reads the explicitly versioned derived asset. This is a documented pre-arm design correction, not evidence of better physical behavior.
