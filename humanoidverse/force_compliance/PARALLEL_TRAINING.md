# Whole-body random single-contact training

This extends the right-hand prototype to **ten candidate points, exactly one
maintained tether per episode**. It does not apply ten simultaneous forces.
The original single-contact configuration and CLI remain supported.

## Contact catalog and observations

`contacts.py` and `whole_body.json` define left/right palms, left/right upper arms,
chest, pelvis, left/right thighs and left/right shins. Offsets were checked against
the actual G1 mesh vertices (within 1 mm). These are local points, not joint
origins. A per-environment seeded RNG uniformly selects a point on reset. An
independent anchor RNG samples direction and displacement. The point remains
fixed throughout the episode. The grasp/tether transmits tension; this is not a
unilateral pushing-contact model.

The policy receives the existing normalized BFM actor observations, force history,
contact flag, actual local point, body encoding, previous latent, and a 10-element
contact one-hot. It receives neither anchor position nor its future trajectory.
The ten-point policy has a different input dimension from old right-hand policies;
old checkpoints must not be loaded into it.

Default force parameters remain K=80 N/m, D=8 Ns/m, anchor displacement 3–10 cm,
60 N force cap, 1 second motion delay, 2 second smooth ramp and 6 second episode.
A substep force cap is a safety bound, not the distribution of applied forces.

Render and verify all points:

```bash
.venv/bin/python -m humanoidverse.force_compliance.inspect_contacts
```

## Parallel training architecture

`parallel.py` uses spawn processes, each owning one native MuJoCo environment and
one frozen BFM model. The original adapter permits only one environment instance
per process. Each worker sees just its assigned physical GPU through
CUDA_VISIBLE_DEVICES, set before importing torch. GPUs are the `nvidia-smi` indices.
CPU threading is limited to one thread per worker.

All workers collect on-policy rollouts for **one shared high-level actor/critic**.
A CPU learner updates this small network and broadcasts its state at each rollout
boundary. Frozen BFM inference runs on the assigned GPUs. This implementation uses
concurrent independent inference processes, not one batched GPU simulator or DDP.
Rollout versions are checked; GAE is computed separately along each worker's time
axis before flattening. The interval-discount convention remains gamma^k, and
terminal transitions do not bootstrap across resets.

Actor learning rate defaults to 3e-5, critic to 3e-4; PPO uses clipped ratios,
entropy regularization, gradient clipping and a KL early stop. High-dimensional
Gaussian likelihoods make the actor sensitive to large first updates.

## Launch and monitoring

From the BFM-Zero repository root:

```bash
.venv/bin/python -m humanoidverse.force_compliance.parallel \
  --model-folder pretrained/official/new_model_for_training_code_inference \
  --config humanoidverse/force_compliance/whole_body.json \
  --output artifacts/force_compliance/whole_body_round1 \
  --gpus 2 3 4 --workers-per-gpu 4 --rollout 128 \
  --total-decisions 1000000 --max-hours 12 \
  --checkpoint-every 5 --evaluate-every 50
```

Use a fresh output directory. The first run automatically validates zero-force
standing and all ten frozen-policy contact baselines under two anchor seeds (20
interaction episodes). It refuses to optimize if a baseline falls or produces no
interaction. This preflight is repeated when resuming.

- `launch.json`: arguments, parent PID and GPU selection.
- `workers.json`: worker PIDs, GPU assignment and observation/action dimensions.
- `preflight.json`: baseline results for every contact.
- `training.jsonl`: throughput, ETA, PPO diagnostics and per-contact force RMS,
  completed episodes, failures and return sums.
- `status.json`: current state and latest progress; `final_status.json` exists only
  after exit. A stale progress timestamp should be checked against process health.
- `initial.pt`: policy before training.
- `policy.pt`: atomically replaced latest policy and optimizer checkpoint, every
  five updates and on normal/time-budget/interrupted exit.
- `evaluation_XXXXXX.json`: deterministic baseline/learned comparisons for each
  point using the same absolute anchor trajectories and a held-out anchor seed.
- `best.pt`: checkpoint with best mean evaluation episode return (including
  failure penalties). Inspect per-contact force/failure metrics as well.
- `worker_XX.log`: native environment startup logs and errors.

`--total-decisions` counts transitions summed over workers. One high-level decision
normally executes five low-level controls and twenty physics substeps. A round of
one million high-level decisions is a sample budget, **not a convergence claim**.
The time limit is checked between rollout/update cycles.

Resume into a fresh directory using the same config and model:

```bash
.venv/bin/python -m humanoidverse.force_compliance.parallel \
  --model-folder pretrained/official/new_model_for_training_code_inference \
  --output artifacts/force_compliance/whole_body_round1_resume \
  --resume artifacts/force_compliance/whole_body_round1/policy.pt \
  --gpus 2 3 4 --workers-per-gpu 4 --total-decisions 1000000
```

Resume restores policy, optimizer, decision count and learner RNG. Simulator
states are deliberately reset; it is not an exact trajectory-level continuation.
SIGTERM or Ctrl-C requests a checkpoint and worker cleanup. A hard kill may lose
updates since the last checkpoint. Worker failure stops the learner rather than
silently training on missing or stale data.

## Validation and interpretation

Run `python -m unittest discover -s tests -v`. Additional tests cover uniform seeded
point sampling, unchanged catalog config, one-hot observations without anchor
leakage, actual MuJoCo wrench application at all ten points, worker-separated GAE,
and finite PPO updates.

Compare force reduction and falls **per contact**, not just pooled reward. The
fixed standing latent is the control baseline. Small initial improvements or zero
force without moving anchors do not demonstrate learned active compliance. Neither
multi-contact interaction nor low-level BFM fine-tuning is enabled.

The first actual launch and throughput measurements are recorded in [WHOLE_BODY_RUN_REPORT.md](WHOLE_BODY_RUN_REPORT.md).
