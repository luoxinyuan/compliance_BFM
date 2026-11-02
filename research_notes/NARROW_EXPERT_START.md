# Narrow independent privileged contact Expert — preregistration and start evidence

2026-11-02; [stage Issue33](https://github.com/GaoE05/robot-research-bridge/issues/33). M11 continuation is stopped and is not resumed. A160/B114 is not a matched-budget final experiment.

## Method and authority

New actor, critic and Adam state. No M11 checkpoint or optimizer initialization; no mandatory online M10 Student. Same frozen BFM and original PD/Tether. The nominal latent uses M10's corrected native `task/straight` cyclic reference, 71 frames / 1.42 seconds, SHA256 `211a192ddf6e3d534a1ef273c22e9cf977510ccb235e1c88dc2f0b24e8d6f2c0`. Its alpha/phase evolve from actual filtered force using the existing native feedback rule. Phase sin/cos, alpha, filtered force and previous residual are explicit own memory; no partner future or task ID. Only the separate M10 comparator loads the old Student.

Expert observations include current privileged partner error/relative velocity, true applied force, normalized proprio/action history, support, native command and own memory. Initial mean residual zero; fixed raw Gaussian std .02. Raw tangent residual is projected onto norm16 sphere; tangent norm cap16 corresponds to at most45deg from nominal. No per-component tanh cap. This covers the demonstrated command envelope rather than inheriting a standing-only radius. It is not proof every latent within the envelope is a safe behavior; original physical termination checks remain.

Single task: right wrist +X .82m, onset4.25s, 7.8s smoothstep ramp, release14.05s/end20.05s, K80/D8/5N. Standing initialization uses the unchanged verified reset/history. Fixed single condition before any neighborhood expansion. No turning task, new contact or multi-motion distribution is added.

## Completed targeted diagnostics, not new training success

GPU1 audit attempt01 (runtime95f1e82) failed while serializing NumPy integer counters in newly added reward logs, before a PPO update. It remains intact. Attempt02 (f6b387e) fixed only serialization and completed seven physical episodes plus one768-decision PPO diagnostic. All completed physical comparisons were stable under the existing monitor.

|Controller|3D error cm|Actual force RMS N|5N cap fraction|Undiscounted PPO episode return|
|---|---:|---:|---:|---:|
|Standing|54.550|4.710|.8504|-29.178|
|Native straight / independent zero mean|13.856|4.704|.8520|-16.257|
|M10 Student comparator|12.662|4.610|.7887|-15.435|
|M10 feedback expert comparator|7.796|4.363|.5717|-12.870|

Independent reconstruction from actual 200Hz force/positions, 50Hz control and summed10Hz PPO rewards: max total-return discrepancy <7e-10; component discrepancy <1.6e-8. Initial continuity cost uses its logged initial value; all subsequent continuity terms reconstructed. Force accounts for13.004 of16.257 native cost, raw demand2.247, tracking.705, continuity.0333, prior0. Expert has force11.187 of12.870 and prior.0091. Thus soft-prior/continuity did not dominate the contact objective. No force cap, anchor or reward weight was changed. Discount .99 per.1s, fractional exponent for partial control interval; GAE .95 and absorbing failure penalty inherited. No failure termination occurred in these seven rows, so empirical validation of a new failure case is not claimed.

A paired ±.4 raw coordinate0 perturbation at7–7.5s changed downstream state and force vectors, with identical pre-intervention state. − perturbation error13.545cm and force4.700N; +13.848cm/4.704N. This establishes a physical control effect, not a strong optimal direction or proof of easy learnability. All XYZ differences are retained in `control-perturbation.json`.

The actual diagnostic minibatch update at actor LR3e-5 overshot jointKL to.0493 and clip fraction.665 despite only one step. Same saved real batch, independent single full-batch update: LR3e-5 gives KL.05323/clip.6927; LR1e-5 gives KL.005914/clip.2448; positive-advantage logprob increases.0572 and negative decreases.0721. Formal training starts fresh with LR1e-5, critic3e-4, independently clipped gradients. No diagnostic weights/optimizer are inherited. The update check is policy-only CPU arithmetic, not BFM CPU execution or new sampling.

A proposed cap4 (14deg) was removed before formal training: actual effective M10 comparator command deviations reach42.44deg relative to native straight (including feedback steering and arm projection). Cap16/45deg is evidence-based coverage, not a standing-only assumption. Reference phase/turn distinctions are retained; this does not claim the whole deviation is arm correction. Initial sampling remains small; new representation, reward and nominal source are recorded separately.

## Fixed budget and evaluation

One GPU, eight CPUs,32GiB owned RSS,8GiB VRAM,8GiB outputs,28,800s cumulative supervised compute including failures/evaluation/rendering. Up to200,448 high-level RL decisions including768 diagnostic decisions: primary120×768, independent-repeat80×768, neighborhood60×768. Conditional allocations are ceilings, not obligations to spend remaining budget.

Mean execution every10 batches; sampled execution also logged. Select on the fixed single task using force RMS +3*error_m +cap fraction +20*failure; retain initial/best/last. No-contact and small-request regressions at0/40/80/120. No claim that selection score replaces physical metrics. Repeat/neighborhood follow only after actual single-domain learning benefit; no repeated final-test selection. New withheld combinations are in `narrow_protocol.json` before fitting. Original M10/M9B/M11 artifacts remain unmodified.

Sources: `research_notes/narrow_{env,policy,worker,train,supervisor,reward_audit,update_check}.py`. Two targeted tests cover legal/zero-residual commands and correct advantageous-sample policy update with nonzero actor/critic gradients. No full foundational interface re-audit.

Data root: `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/narrow-contact-expert-20261102/`; inspect ledger and individual `launch.json` for exact runtime. `audit-attempt-02/outputs/job/audit-physical` has real trajectories, independent reward audit and control perturbation; `same-batch-update-check.json` is in its parent. No new Expert benefit is established yet. Future videos must be actual saved qpos at1×, not reference animation.
