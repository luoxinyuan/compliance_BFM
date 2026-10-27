# Strong-force whole-body training

`whole_body_strong.json` increases the tangent latent residual scale from 0.3 to
3.0. The low-level BFM remains frozen. Each episode selects one of the same ten
surface contact points and one nominal force level (10, 20, 30, 40, 60 N) uniformly.
The force direction is sampled uniformly on the sphere.

The nominal level determines the anchor excursion as `level / stiffness`.
With stiffness 200 N/m, excursions are 0.05, 0.10, 0.15, 0.20, 0.30 m.
Damping remains 8 Ns/m and the applied force is capped at 60 N. These levels
are reference spring loads for an unmoving endpoint at the end of the ramp,
not guarantees of measured force. Anchor velocity, damping, and robot movement
change the actual load. The anchor remains independent of the robot so following
it can reduce the force to zero; no constant-force controller is used for training.
The existing two-second ramp follows a one-second attachment period.

Run a fresh high-level policy because the force distribution and residual scale
have changed. Do not use exact `--resume` with a round-one checkpoint. Preflight
checks the frozen no-force and pulling baseline before optimization. Inspect
`preflight.json`, measured RMS forces and failure rates in `training.jsonl`, and
paired evaluation files; completing the budget does not establish convergence.

```bash
.venv/bin/python -m humanoidverse.force_compliance.parallel \
  --model-folder pretrained/official/new_model_for_training_code_inference \
  --config humanoidverse/force_compliance/whole_body_strong.json \
  --output artifacts/force_compliance/whole_body_strong_r3_round2_20261027 \
  --gpus 5 6 7 --workers-per-gpu 4 --rollout 128 \
  --total-decisions 1000000 --max-hours 12 \
  --checkpoint-every 5 --evaluate-every 50
```

Validation: `python -m unittest discover -s tests -p 'test_force_compliance*.py' -q`.
The strong-force test verifies all five reference forces on a stationary endpoint,
complete unloading after following the anchor, and invalid force configuration
rejection. Existing default configurations retain displacement-based sampling.
