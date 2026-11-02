# Narrow Expert: control-scale diagnosis, 2026-11-02

Primary120 completed at fc8dc3b9cb92132b6c49d2c4df0d2e6d97fa22b1, 92,160 RL decisions. Initial mean 13.856cm/4.704N; selected80 13.507cm/4.703N; final120 13.765cm/4.702N. No material force benefit. All prior data retained; no heldout evaluation or Student training.

## Evidence and inference
Saved same-observation q_PD differences from nominal: learned120 mean .02324deg RMS, sampled .11239deg; effective M10 feedback expert 2.46557deg (includes heading feedback and arm projection). Mean latent angle .176deg and sampled1.156deg are far below existing45deg cap. This supports action-unit/learning-signal miscalibration, not a proven sole cause or a probability bug.
Reward independently recomputed to <7e-10 total discrepancy. Actual force contributes about80% of nominal integrated cost; prior/continuity do not dominate. ~85% saturation limits force discrimination, but uncapped-demand/error auxiliaries rank bad behavior worse. Reward remains unchanged in this comparison.

## Real physical scale probe (zero learned mean, no updates)
Runtime3958bbf4ffcf9b33c03a0eb5fb10c8a67fa58fa1. Same epsilon seeds per gain; identical episode resets, not identical subsequent states. Full actual trajectories, unchanged tether/cap/partner.

|gain|seed|error cm|force RMS N|cap fraction|
|---|---|---|---|---|
|1|501|13.227|4.689|.789|
|1|502|13.853|4.705|.852|
|1|503|14.066|4.704|.853|
|4|501|13.513|4.712|.858|
|16|501|18.127|4.712|.858|
|16|502|11.487|3.830|.461|
|16|503|15.689|4.734|.867|

All seven completed without physical boundary. Gain16 opens substantial beneficial AND harmful response variation; it is not an improved controller. Three seeds are not a robust performance estimate. Do not select the favorable seed as success.

## One controlled training amendment
Multiply raw tangent residual by16 before the unchanged norm16 cap and sphere16 projection. Raw Gaussian std=.02 and PPO log probabilities remain in raw coordinates. This changes physical mean sensitivity and exploration together, not reward or control authority. Preserve gain1 default and reject checkpoint resume across gain identities. Fresh seed101 actor/critic/optimizer; same network, LR1e-5/3e-4, task, reward, nominal reference, PPO settings. Compare matched early budgets with primary; do not call80 vs120 equal budget.

Predeclared run:80 batches=61,440 RL decisions, evaluation every10, original development selection; original sealed final conditions remain unused. Existing cumulative28,800s/200,448decision ledger applies, one GPU/8CPU/32GiB RSS/8GiB VRAM. No new global resource allowance. Remaining after probe~21,292s and107,520 decisions. No automatic second run. Success requires deterministic physical improvement, not random samples or critic EV. If mean still remains ineffective or becomes unstable, inspect measured action changes and learning signal rather than continue blind gain scans.

Tests: AST all changed files; test_narrow_gain.py verifies exact gain1 compatibility, zero identity, gain16 physical equivalence to same-noise probe, sphere/cap and untouched raw probabilities. Runtime PPO independently checks old logprob each batch. First train launch/results recorded in Issue33 and CURRENT_STATE.

Artifacts: /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/narrow-contact-expert-20261102/{train-attempt-02,probe-attempt-01,train-attempt-03}/outputs/job. Probe has0training decisions/updates; physical diagnostics are separately counted and consume wall budget.
