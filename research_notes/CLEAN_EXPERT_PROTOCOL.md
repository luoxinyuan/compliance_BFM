# Clean privileged Expert: learned motor control, not residual cancellation of a force rule

2026-11-02. User supersedes the proposed three reward/initialization groups x two seeds. That plan is NOT executed. Old narrow and M11 runs/checkpoints remain immutable. This protocol is fixed before new physical execution.

## Source audit and causal boundary

`narrow_env.Task.prepare` calls `CyclicFeedback.step` every20ms. `m9_cyclic` low-pass filters force with .25s time constant, uses clip((body_Fx-.25)/2,0,1) as walking target, alpha slew +.6/-.9 per second, and advances phase at 1/period only while alpha>.05. Privileged turn is also computed from guide, but narrow `prepare` actually samples ONLY straight: turn changes its effective phase period, not a turn latent. PPO controls only256D tangent residual, not alpha/phase/stop. At old hold, negative Fx drives alpha0; that explains nominal exit mechanically but does not prove it originally caused wrist overshoot. Old trajectory and all-time gain tests cannot isolate the counterfactual. A small residual has to compensate for nominal state changes. An algebraically broad residual does not make this an efficient learning representation.

## New interfaces and inputs

A structured: network outputs alpha target[0,1], native phase rate[0,1.5], and256D tangent command. Alpha retains the already-used command slew(.6/-.9 per second); phase integrates the NETWORK rate even at alpha0. Neither reads force except through network input. The qualified straight latent is sampled at this own phase and slerped from standing; tangent command is scaled16, projected orthogonal, capped norm16, sphere16. Limits are initial physically grounded domain, not a global BFM limitation.

B direct: network outputs256D absolute normalized global latent. Sphere16 projection; zero-vector numerical fallback standing. Network command at10Hz is slerped from the previous executed latent across5 actor ticks(50Hz); no motion phase or force controller runs online. Prior available through initialization data, not an online Student or rule. A/B therefore differ in parameterization and interpolation, not a supposedly identical action space.

Both read normalized original465D proprio/action history, current privileged contact error/relative velocity/force, body/support, actual last0.5s force history expressed in current body frame, own previous executed z and own alpha/phase/rate memory (zeros for direct). No future partner, episode ID, global clock, teacher mode or teacher internal memory inputs. Privileged current contact-active flag is allowed for Expert; it is not a deployment-Student claim.

Bootstrap: ONE same full-task M10 reference+projected-feedback expert trajectory per interface; runs are physically matched, only feature/label coordinates differ.1500 supervised minibatch updates,128x128 actor; endpoint command labels describe the next100ms actual teacher action, not future observations in input. Direct supervised z loss, structured alpha/rate/tangent losses. This is optional prior initialization, not proof of learned feedback. Fresh PPO actor/critic optimizer, no M11 optimizer. Initial before/after bootstrap physical curves retained. Teacher inverse tangent boundary checked; no silent range expansion. No online teacher during PPO.

## Fixed task, reward and training

Same82cm straight right-wrist task: ready3s, onset4.25, ramp7.8,release14.05,end20.05; K80/D8/5N unchanged, actual physical force and all XYZ logs. Original BFM/PD and200Hzphysics/50Hzactor/10HzPPO unchanged.

All existing task/recovery/continuity costs unchanged EXCEPT soft_prior=0 for both interfaces: its original nominal target used the removed gate and is architecture-dependent. This is an explicit objective change, not a bug fix or a force-weight experiment. Caps/anchor/termination unchanged. Reward sorting and original numerical audit retained, bootstrap physical return recorded under new reward.

First checkpointed run:20x768decisions per interface, seed101, mean+sample development every5batches, no-contact and15cm local regression every10. Same verified PPO update, actorLR1e-5,critic3e-4,clip.15,analyticKLstop.02; raw distribution probability and actual transformed command logged. Structured raw std .10/.15 for participation/rate, .008per residual coordinate; direct .008per normalized-z coordinate. Fixed std excluded optimizer. Neither claims matched physical exploration: measure actual commands and effects.

Selection mean F_RMS+3*error_m+cap_fraction+20*boundary, then final once. No final reselection. New final(.80m,ramp8.05,onset4.4,release14.45,end20.45) sealed until candidate fixed. Evaluate first20 before continuing; do not launch six long jobs. Further rounds only with interpretable mean learning/diagnostic evidence, within total cap. No gain/reward scan.

## Finite resources / isolation

At admission GPU1/5/6 idle;2/3 occupied and avoided. Shared210logicalCPU,~911GiB availableRAM,96–98%idle,pressure0. Snapshot is not a future reservation. Per job8CPU,32GiBRSS,8GiBVRAM. Initially two jobs on1/5, distinctCPU8–15/16–23. Third card only for an informative check or seed after evidence. Stage sum of supervised process wall<=14400s (concurrent time counted twice), max122880new RL decisions, output12GiB total. Preparation reservations1800s each, first training2400s each. Admission atomically reserves stage time/decisions; exact owned process groups tracked and released. Shareddisk free is NOT personal quota. No daemon/watch, no downloads.

Source in independent personal worktrees; immutable runtime commits before launches. Outputs `bridge-v2-owned/runs/clean-contact-expert-20261102`, separate kind/stage/seed paths and ledger. Record failures, init/best/last model SHAs and complete physical videos. Shared checkout/main/BFM/PD/weights and all other tasks untouched.

## Decisions

If initialization/physical command expression fails, diagnose that specific control/learning boundary before PPO. If representable but RL fails, inspect actual mean vs sample, exploration, credit/reward/gradient; never quietly restore force-to-motion rules. The stage delivers what is actually validated, not a claim that supervision alone learned autonomous feedback or that PPO failure disproves BFM.

## Evidence-driven update correction before continuation
First three batches at999494f preserved: first accepted joint KL A=.3830/B=.03768 exceeded planned.02. The old implementation only stopped AFTER the oversized Adam step. Both owned runs interrupted cleanly, all children released, 2304decisions each retained. New optimizer transaction backs up parameters and Adam moments; over-limit candidate is rolled back and actor LR halved, at most7 candidates, no old-logprob changes, fixed std. Accepted batch jointKL<=.02; stop near.015. Nominal LR resets at each new rollout batch, so backtracking near the boundary cannot silently drive all later batches to zero LR. Both restart from same archived post-supervision initial.pt and fresh empty optimizer, not from damaged candidates. Reward/interface/physics/noise unchanged. This is an optimization-design correction, not proof that it caused all historical failures. Five tests include exact weight+Adam rollback. Failed/aborted work counts in resource ledger.
