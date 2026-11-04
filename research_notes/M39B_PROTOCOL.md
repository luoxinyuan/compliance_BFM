# M39-B — registered matched retention comparison

2026-11-04; [stage Issue40](https://github.com/GaoE05/robot-research-bridge/issues/40). M39-A is closed; old results/checkpoints and shared checkout remain unchanged.

Common initial: seed3901 best64, SHA256 `72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a`. Both arms inherit identical actor, critic, center/scale and fixed Gaussian std; fresh Adam. Seed3911; each64 batches×768 decisions, evaluations every8. No automatic extra seed/extension; remaining ceiling reserves diagnosis and evaluation. Current stage maximum196608 decisions,24000 summed job-wall seconds,15GiB output,3 GPUs; these are ceilings. Calibration uses768 additional physical decisions, no training.

A uses existing raw Gaussian command retention, coefficient.002. B uses mean squared q_PD radians, all29 joints and all5 valid50Hz substeps. Data and replay minibatch generator are identical. Same recorded highlevel input, actual previousz, actual per-substep BFM observation, original slerp and qPD mapping. Freeze BFM parameters; retain derivative to highlevel output. No physical differentiable rollout claim.

Before optimization, fixed8 replay minibatches (seed3939) calibrate B coefficient to match median A actor-gradient norm. Log PPO gradients and ratios/cosines on common initial real768-decision batch. No test-driven coefficient changes. Verify all saved source actions/qPD and analytic gradient; positive raw rescaling mean invariance does not imply stochastic equivalence with fixedrawnoise. Original raw logprob, .15 clip, analytic joint KL/transactional Adam backtracking retained. Replay losses do not change probability coordinates.

Replay dataset SHA `4e72f76a476b43746de90922539df989d60e6dbffd79eac165a9d22472401095`; existing reliable T30 source trajectories and actually-loaded D2 rotations. No failed combination labels and no claim of broad recovery-domain coverage.

Shared curriculum56% combination/entry/withdrawal, other44% old tasks/zero/noforce/puretranslation/rotation. Exact probabilities, bounded continuous variation,16 dev and13 new sealed combinations in m39b_protocol.json. Realized episode/decision/stage coverage logged. Increasing curriculum relative to M39-A is common to both arms, cannot be attributed to loss difference.

Reward, gamma .99/highlevel, GAE .95,5N force budget and physical control remain unchanged. Full finite horizon ends are terminal; sampler chunk boundaries retain next-value bootstrap. Environment failures retain existing absorbing-cost handling. No stage/task labels added to actor input.

Development selection: mean relative cost+.5worst over16 conditions. All phases/worst tasks still reported separately. Same rollback criterion (boundary, reversed developed±15, worst>1.75initial); preserve failed candidates, restore devbest+Adam and halve actorLR. No arm-specific performance early termination before64; physical/resource boundaries stop and save. New sealed results never select/extend training. One pairedseed supports a bounded mechanism comparison, not population-level statistical superiority.

Report actual contact entry, translation delay/cumulative lag, rotation entry/hold/withdrawal, post-withdrawal translation, and released recovery. Released-anchor distance is not active task failure. Include old baselines and all failures; actual qpos1× videos only. No Student training, shared BFM/PD edits, online hand-written force-to-motion, force-cap change, Bridge, main push or real robot.
