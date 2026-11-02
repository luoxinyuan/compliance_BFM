# Independent Expert: why gain16 training still did not reduce force

2026-11-02. Saved-trajectory audit plus three predeclared frozen-policy physical interventions; no PPO/Student updates or final-set evaluation. This is development diagnosis, not a new qualified Expert.

## 1. What actually finished

Primary120: runtime fc8dc3b9cb92132b6c49d2c4df0d2e6d97fa22b1. Gain16 comparison: runtime51c653fa6b9cab0dd87a788394d842ddd9419d7a,80 batches/61,440 decisions/1100 optimizer steps,4452.97 supervised seconds,all workers exited. Selected last80 SHA256 `017ac2ada5677277baf4cb07a69fe201055cb969b64afa269aae579d4f68774b`. Same-first80 comparison is valid;80 vs120 is not equal budget. Mean13.856→12.368cm; force4.704→4.703N. No meaningful force improvement. Initial/local regression3.877→3.948N retained as degradation. No_contact has zero applied force but this alone does not prove no unsolicited action.

## 2. Gain, output and authority are different

Contact-window mean latent angle2.194deg, tangent norm.613/16, q_PD residual RMS.3299deg. No residual cap hit. Stochastic execution angle17.823deg, tangent5.145, q_PD RMS1.895deg. Thus random physical changes are much larger than the learned mean. Effective M10 expert on its own trajectory: angle17.294deg mean,42.443deg maximum; equivalent tangent norm maximum14.632<16. Its commands are algebraically representable inside the current interface on those states. This is NOT a proof of arbitrary Student-state reachability or BFM general capability.

The output limit is not currently binding. Gain1 was too weak; gain16 makes physical exploration effective, but the actor still learns a small deterministic correction. Do not infer that larger force weight, a wider latent sphere, or more random noise alone solves feedback.

## 3. XYZ and overshoot reveal the physical failure

|Policy|contact XYZ error RMS cm|hold signed XYZ error cm (anchor-wrist)|contact force RMS N|demand norm median N|
|---|---|---|---|---|
|Native|7.90 /10.23 /5.00|−6.85 /−5.32 /+.26|4.704|10.159|
|gain16 trained mean|8.44 /7.55 /4.98|−9.49 /+.62 /+.71|4.703|8.700|
|M10 feedback expert|4.43 /5.56 /3.21|−4.45 /−.81 /−1.06|4.363|5.811|

Improvement is predominantly less lateral error, not uniformly better tracking. In hold the learned wrist is about9.5cm AHEAD of the anchor, actual Fx≈−4.961N. It is being pulled backward, not failing to walk forward enough. `CyclicFeedback.step` maps filtered forward body force to alpha; alpha=0 throughout this hold, leaving standing nominal plus a weak residual. Mean hold q_PD residual.257deg is insufficient to remove the observed overshoot. The learned policy changes gait/feedback interactions but does not yet organize deceleration and wrist recovery effectively. This is a behavior-source/control-learning interaction, not evidence that no BFM action can recover.

## 4. Frozen mean amplitude falsification: real full episodes

Same frozen checkpoint, no sampling, same full reset, actual Tether and partner. Apply gain multipliers only inside legal tangent projection; retain raw own-command memory in original policy coordinates. Tangent16 cap and radius16 unchanged. All3 choices fixed before execution. Baseline scale1 is saved original evaluation.

|Mean multiplier|3D error cm|force RMS N|5N saturation|contact latent angle|q_PD residual RMS deg|
|---|---|---|---|---|---|
|1|12.368|4.703|84.94%|2.19deg|.330|
|4|11.296|4.459|54.11%|8.78deg|1.140|
|16|18.180|4.993|96.02%|34.04deg|4.147|
|−4|14.479|4.762|85.91%|7.99deg|.928|

All complete, no recorded physical boundary. Scale4 lowers force5.18%, error8.67% vs current mean; hold force3.862N and no capped samples, signed X overshoot4.57cm. Scale16 worsens task and control continuity: max q_PD step94.45deg vs47.33deg original mean; absence of fall is not acceptable motion quality. Negative4 degrades, supporting some useful learned direction. These are single exposed deterministic conditions, not independent training repeats or generalization evidence.

Important confound: multiplier acts before contact too. Scale4 precontact q_PD residual1.174deg, root Y shift−1.59cm; scale16 precontact posture changes more. Hence this establishes total closed-loop scaling effect, not isolated same-contact-state feedback causality. Do not call scale4 a deployed replacement. Mean residual angle is2.30deg precontact/2.19 contact/2.31 recovery; scalar magnitude alone cannot prove constant direction, but unnecessary no-contact correction is a real concern.

## 5. Reward and PPO: checked facts vs hypotheses

- Reward reconstruction previously matched to<7e-10 total. No raw logprob/GAE temporal-index failure found in the inspected path. Done masks stop both bootstrap/trace; workers preserve actual sampled raw coordinates. These checks do not prove an optimal RL design.
- `narrow_env.Task.step` force cost is1.5*||F||²/25, not a Gaussian force reward with sigma. At the5N cap it is exactly1.5 irrespective of excess spring demand. ~85% contact saturation makes the dominant term locally nondiscriminating. Multiplying its weight cannot distinguish two already-capped states.
- Native/gain16 mean contact-integrated force costs13.0039/12.9984; uncapped2.2472/2.0526; tracking.7050/.5617. Most measured objective gain comes from auxiliary terms, while force remains flat. Uncapped term is only.15*log1p(||demand||²/25), so improvements within saturation have a weak but nonzero signal. This explains limited discrimination; it does not prove a reward change alone succeeds.
- Prior/continuity do NOT dominate: gain16 mean contact costs.000162/.03245 vsforce12.998. Increasing residual authority by lowering the soft-prior weight is unsupported.
- Same-first80 PPO mean jointKL≈.021; final clip fraction roughly.57–.64 for gain16. Positive-advantage raw actions gain logprob, negative ones lose it. Per-coordinate mean change≈.000255; with gain16 this is about.064 total tangent L2 per batch (roughly.23deg locally). Conservative joint trust region plus256D exploration plausibly makes useful mean learning slow/noisy; high clip fraction is a diagnostic, not proof of wrong logprob or an instruction to remove trust limits.
- Actor preclip norms are large, but Adam largely normalizes a uniform gradient scale. Do not claim clipping alone reduced the physical update100x. Critic old-value EV≈.878 late does not establish actor success.
- .99 gamma at10Hz, GAE lambda.95 gives TD-error trace e-fold≈1.63s, discount≈9.95s. This may weaken delayed gait/recovery credit, but has not been isolated experimentally.

## 6. Decision

Partial support for insufficient MEAN residual magnitude, not insufficient legal range. One useful learned direction exists but more gain also amplifies precontact bias, overshoot and continuity defects. Do not launch another unchanged long run or adopt scale4 as reliable.

Recommended next minimal step: distinguish useful contact feedback from prepositioning using a fixed contact-window-only scale4 check from the original mean's identical precontact state (diagnostic, not a hard deployment gate), with original scale1 and existing all-time scale4 evidence. If the benefit survives, calibrate learned-mean actuation independently of stochastic exploration, preserve no-contact/stop regression, then one bounded same-task optimization comparison. If it does not, prioritize seeding the independent actor with the already-valid same-weight feedback expert's legal latent/action behavior, then RL fine-tune current error/force feedback; this does not reintroduce online M10 Student or reset the whole control system. Do not simultaneously change gain, force reward, distribution and network.

No new training started this analysis. Current problem is not established to be BFM expressiveness or lack of GPU. It is a combination of insufficient effective mean correction, imperfect correction direction/overshoot recovery, and saturated/noisy learning signal. The contribution of each remains partially causal, partially inferred as marked above.

## Reproduction, failures and resources

`narrow_residual_analysis.py` reads saved physics/control only; `narrow_residual_summary.json` contains phase aggregates, not raw assets. Paths rooted at `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/narrow-contact-expert-20261102/`: original `train-attempt-02`, gain16 `train-attempt-03`, amplitude `amplitude-attempt-02/outputs/job/amplitude`, aggregate `residual-audit/summary-with-amplitude.json`.

Amplitude runtime fc650c2984c8022b4baacd1951defabdbc9c77c4. First amplitude attempt failed before model/physics loading because the new worktree inherited sparse checkout and lacked pinned environment.py. Materialized tracked files using GIT_LFS_SKIP_SMUDGE sparse-checkout disable; no weight downloads or source-pin bypass. Original23.78s failed attempt retained. Successful supervisor181.47s,0RL decisions/updates,all workers exit0, GPU1 released. Same cumulative resource ledger; no changes to shared BFM/PD, no other task intervention. Source integration/reporting does not overwrite runtime or results.
