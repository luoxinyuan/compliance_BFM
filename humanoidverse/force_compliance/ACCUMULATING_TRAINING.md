# Accumulating latent compliance

`whole_body_accumulate.json` keeps the BFM frozen and changes only the high-level
latent update semantics relative to `whole_body_strong.json`. The residual scale
remains 3.0. Every episode initializes `z_0` to the saved standing latent. At each
10 Hz high-level decision:

```
base = z_t
residual = tanh(high_level_action)
tangent = residual - dot(residual, base) / dot(base, base) * base
gate = clamp(norm(measured_force) / 5 N, 0, 1)
z_next = project_z(base + 3.0 * gate * tangent)
```

At exactly zero measured force, return `base` unchanged (bitwise), rather than
reprojecting or returning to standing. The next low-level actions use `z_next`.
The last measured force is used; after physical release the gate sees zero on
the next high-level decision. Holding a latent does not guarantee holding a pose
or zero robot velocity. The existing observation already includes `z_t`, so no
observation dimensions change. Action likelihoods remain those of the Gaussian
raw residual. Tangent projection is relative to the current command.

The default `latent_update_mode="standing"` preserves old checkpoint semantics.
New training uses `"accumulate"`, recorded in config/checkpoints. Start a fresh
high-level policy: an old policy learned offsets, not temporal increments.
The low-level BFM is not reinitialized or optimized. Standing baseline evaluation
continues to use the fixed standing latent for an honest paired comparison.

All other settings remain those of the strong-force run: ten randomly selected
single contact points, nominal spring loads 10/20/30/40/60 N, 200 N/m stiffness,
8 Ns/m damping, 60 N cap, six-second episodes, and the existing reward. Nominal
loads are not guaranteed measured loads; moving toward the fixed anchor unloads
the spring. Both preflight and periodic evaluation cover all five nominal levels.

Launch:

```bash
.venv/bin/python -m humanoidverse.force_compliance.parallel \
  --model-folder pretrained/official/new_model_for_training_code_inference \
  --config humanoidverse/force_compliance/whole_body_accumulate.json \
  --output artifacts/force_compliance/whole_body_accumulate_r3_round3_20261027 \
  --gpus 5 6 7 --workers-per-gpu 4 --rollout 128 \
  --total-decisions 1000000 --max-hours 12 \
  --checkpoint-every 5 --evaluate-every 50
```

Validation includes sequential accumulation on the current tangent plane,
100 exact zero-force holds, mixed-force batch gating, missing-state rejection,
and existing physics/PPO regression tests. A real BFM/MuJoCo release diagnostic
and a two-update parallel smoke run are saved under
`artifacts/force_compliance/accumulate_r3_smoke/`.
